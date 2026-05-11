import argparse
import json
import os
import re
import random
from collections import defaultdict
from typing import Dict, Iterable, List, Optional

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from ..config import load_yaml_config, normalize_eval_config
from ..data.loaders import DATASET_LOADERS
from ..train.rewards import reward


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _parse_step(name: str) -> int:
    match = re.search(r"(\d+)$", name) or re.search(r"(\d+)", name)
    return int(match.group(1)) if match else 0


def _is_model_dir(path: str) -> bool:
    return os.path.isfile(os.path.join(path, "config.json"))


def _is_lora_adapter_dir(path: str) -> bool:
    return os.path.isfile(os.path.join(path, "adapter_config.json"))


def resolve_full_ft(cfg: dict) -> bool:
    return bool(cfg.get("full_ft", cfg.get("do_full_ft", False)))


def resolve_use_lora(cfg: dict) -> bool:
    if "use_lora" in cfg:
        return bool(cfg["use_lora"])
    return not resolve_full_ft(cfg)


def _matches_whitelist(name: str, whitelist: Optional[Iterable[str]]) -> bool:
    if whitelist is None:
        return True
    return any(name == item or name.startswith(item) or item in name for item in whitelist)


def _get_json_path(cfg: dict) -> str:
    if cfg.get("json_out_path"):
        return cfg["json_out_path"]
    model_short = cfg["model_name"].rstrip("/").split("/")[-1]
    return os.path.join("outputs", "eval", model_short, "evaluation_results.json")


def _discover_target_dirs(algo_dir: str, eval_steps: bool, use_lora: bool) -> List[dict]:
    is_target_dir = _is_lora_adapter_dir if use_lora else _is_model_dir
    ckpt_root = os.path.join(algo_dir, "ckpts")
    targets = []

    if os.path.isdir(ckpt_root):
        ckpts = [
            name for name in os.listdir(ckpt_root)
            if os.path.isdir(os.path.join(ckpt_root, name)) and is_target_dir(os.path.join(ckpt_root, name))
        ]
        ckpts = sorted(ckpts, key=_parse_step)
        if ckpts:
            chosen = ckpts if eval_steps else ckpts[-1:]
            for ckpt in chosen:
                targets.append(
                    {
                        "ckpt": ckpt,
                        "path": os.path.join(ckpt_root, ckpt),
                        "step": _parse_step(ckpt),
                    }
                )
            return targets

    if is_target_dir(algo_dir):
        return [{"ckpt": os.path.basename(algo_dir.rstrip("/")), "path": algo_dir, "step": 0}]

    child_targets = []
    if os.path.isdir(algo_dir):
        for name in os.listdir(algo_dir):
            child = os.path.join(algo_dir, name)
            if os.path.isdir(child) and is_target_dir(child):
                child_targets.append(
                    {
                        "ckpt": name,
                        "path": child,
                        "step": _parse_step(name),
                    }
                )
    child_targets = sorted(child_targets, key=lambda item: item["step"])
    if child_targets and not eval_steps:
        child_targets = child_targets[-1:]
    return child_targets


def discover_eval_targets(cfg: dict) -> Dict[str, List[dict]]:
    use_lora = resolve_use_lora(cfg)
    eval_steps = bool(cfg.get("eval_steps", False))

    if cfg.get("model_path"):
        label = cfg.get("model_label", cfg.get("full_ft_label", "model"))
        return {
            label: [
                {
                    "ckpt": os.path.basename(cfg["model_path"].rstrip("/")),
                    "path": cfg["model_path"],
                    "step": int(cfg.get("model_step", 0)),
                }
            ]
        }

    model_short = cfg["model_name"].rstrip("/").split("/")[-1]
    model_root = cfg.get("model_root") or os.path.join(cfg.get("output_root", "outputs/models"), model_short)
    whitelist = cfg.get("algorithms")
    whitelist = None if whitelist in (None, "all") else {str(item) for item in whitelist}

    targets_by_algo = {}
    if not os.path.isdir(model_root):
        print(f"Model root not found: {model_root}")
        return targets_by_algo

    include_baseline = bool(cfg.get("include_baseline", False))
    if include_baseline:
        baseline_dir = os.path.join(model_root, "baseline")
        baseline_targets = _discover_target_dirs(baseline_dir, eval_steps=eval_steps, use_lora=use_lora)
        if baseline_targets:
            targets_by_algo["baseline"] = baseline_targets

    for algo_name in sorted(os.listdir(model_root)):
        if algo_name == "baseline":
            continue
        if not _matches_whitelist(algo_name, whitelist):
            continue
        algo_dir = os.path.join(model_root, algo_name)
        if not os.path.isdir(algo_dir):
            continue
        targets = _discover_target_dirs(algo_dir, eval_steps=eval_steps, use_lora=use_lora)
        if targets:
            targets_by_algo[algo_name] = targets
        else:
            print(f"No eval target found for {algo_name}: {algo_dir}")

    print("Auto-discovered eval targets:")
    for algo, targets in sorted(targets_by_algo.items()):
        for target in targets:
            print(f"  {algo}: {target['path']}")
    return targets_by_algo


def resolve_torch_dtype(cfg: dict):
    requested = str(cfg.get("torch_dtype", "auto")).lower()
    if requested == "auto":
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            return torch.bfloat16
        if torch.cuda.is_available():
            return torch.float16
        return torch.float32
    if requested in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if requested in {"fp16", "float16"}:
        return torch.float16
    if requested in {"fp32", "float32"}:
        return torch.float32
    raise ValueError(f"Unsupported torch_dtype: {requested}")


def load_eval_model(model_path: str, cfg: dict):
    tokenizer_source = cfg.get("tokenizer_name") or model_path
    if not os.path.exists(tokenizer_source):
        tokenizer_source = cfg["model_name"]
    elif os.path.isdir(tokenizer_source) and not os.path.isfile(os.path.join(tokenizer_source, "tokenizer_config.json")):
        tokenizer_source = cfg["model_name"]

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model_kwargs = {
        "torch_dtype": resolve_torch_dtype(cfg),
        "trust_remote_code": True,
    }
    device_map = cfg.get("device_map")
    if device_map:
        model_kwargs["device_map"] = device_map

    use_lora = resolve_use_lora(cfg)
    if use_lora:
        try:
            from peft import PeftModel
        except ImportError as exc:
            raise ImportError("use_lora=true requires peft to be installed.") from exc
        base_model = AutoModelForCausalLM.from_pretrained(cfg["model_name"], **model_kwargs)
        model = PeftModel.from_pretrained(base_model, model_path)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs)

    if not device_map:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
    model.eval()
    return model, tokenizer


def generate_completions(model, tokenizer, prompt_text: str, cfg: dict, seed: int) -> List[str]:
    inputs = tokenizer(prompt_text, return_tensors="pt")
    device = next(model.parameters()).device
    inputs = {key: value.to(device) for key, value in inputs.items()}

    k = int(cfg["k"])
    generation_batch_size = int(cfg.get("generation_batch_size", min(k, 16)))
    max_new_tokens = int(cfg.get("max_new_tokens", cfg.get("max_tokens", 768)))
    completions = []
    produced = 0

    while produced < k:
        chunk = min(generation_batch_size, k - produced)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed + produced)
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                do_sample=True,
                temperature=float(cfg.get("temperature", 0.3)),
                top_p=float(cfg.get("top_p", 0.95)),
                max_new_tokens=max_new_tokens,
                num_return_sequences=chunk,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        prompt_len = inputs["input_ids"].shape[1]
        generated = output_ids[:, prompt_len:]
        completions.extend(tokenizer.batch_decode(generated, skip_special_tokens=True))
        produced += chunk
    return completions


def quick_eval(model, tokenizer, cfg: dict, eval_dataset: list, seed: int):
    n_total = 0
    n_acc_first = 0
    n_pass_at_k = 0
    is_correct = []

    for item in eval_dataset:
        prompt_text = tokenizer.apply_chat_template(
            item["prompt"],
            add_generation_prompt=True,
            tokenize=False,
        )
        responses = generate_completions(model, tokenizer, prompt_text, cfg, seed)
        correct_responses = [reward(response, item["answer"]) for response in responses]
        is_correct.append(correct_responses)
        if correct_responses and correct_responses[0]:
            n_acc_first += 1
        if any(correct_responses):
            n_pass_at_k += 1
        n_total += 1

    accuracy = (n_acc_first / n_total) if n_total else 0.0
    pass_at_k = (n_pass_at_k / n_total) if n_total else 0.0
    return float(accuracy), float(pass_at_k), is_correct


def load_eval_datasets(cfg: dict) -> Dict[str, list]:
    datasets = {}
    limit = cfg.get("eval_limit")
    for name in cfg["datasets"]:
        if name not in DATASET_LOADERS:
            raise ValueError(f"Unknown dataset {name}. Available: {sorted(DATASET_LOADERS)}")
        ds = DATASET_LOADERS[name]()
        if limit is not None:
            ds = ds[: int(limit)]
        datasets[name] = ds
    return datasets


def save_results(json_out_path: str, k: int, per_seed_records: list) -> None:
    key2rows = defaultdict(list)
    for record in per_seed_records:
        key = (record["algo_type"], int(record["step"]), record["dataset"])
        key2rows[key].append(record)

    aggregates = []
    for (algo, step, dataset), rows in key2rows.items():
        p1s = np.array([row["pass_at_1"] for row in rows], dtype=float)
        pks = np.array([row["pass_at_k"] for row in rows], dtype=float)
        aggregates.append(
            {
                "dataset": dataset,
                "algo_type": algo,
                "step": int(step),
                "n": int(len(rows)),
                "pass_at_1_mean": float(p1s.mean()) if len(p1s) else 0.0,
                "pass_at_1_std": float(p1s.std(ddof=1)) if len(p1s) > 1 else 0.0,
                "pass_at_k_mean": float(pks.mean()) if len(pks) else 0.0,
                "pass_at_k_std": float(pks.std(ddof=1)) if len(pks) > 1 else 0.0,
            }
        )

    payload = {
        "k": int(k),
        "seeds": sorted({int(record["seed"]) for record in per_seed_records}),
        "datasets": sorted({record["dataset"] for record in per_seed_records}),
        "per_seed_records": sorted(
            per_seed_records,
            key=lambda r: (r["dataset"], r["algo_type"], int(r["step"]), int(r["seed"])),
        ),
        "aggregates": sorted(
            aggregates,
            key=lambda r: (r["dataset"], r["algo_type"], int(r["step"])),
        ),
    }
    dirpath = os.path.dirname(json_out_path)
    if dirpath:
        os.makedirs(dirpath, exist_ok=True)
    with open(json_out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"Saved JSON to {json_out_path}")


def load_existing_records(json_out_path: str, k: int) -> list:
    if not os.path.isfile(json_out_path):
        return []
    with open(json_out_path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    if int(payload.get("k", k)) != int(k):
        print(f"Existing JSON uses k={payload.get('k')}; current k={k}. Starting fresh.")
        return []
    records = payload.get("per_seed_records", [])
    print(f"Loaded {len(records)} existing per-seed records.")
    return records


def eval_all(cfg: dict) -> None:
    set_seed(int(cfg.get("seed", 42)))
    targets_by_algo = discover_eval_targets(cfg)
    if not targets_by_algo:
        print("No evaluation targets found.")
        return

    datasets = load_eval_datasets(cfg)
    json_out_path = _get_json_path(cfg)
    per_seed_records = load_existing_records(json_out_path, int(cfg["k"]))
    completed = {
        (record["dataset"], record["algo_type"], int(record["step"]), int(record["seed"]), int(cfg["k"]))
        for record in per_seed_records
    }

    for algo_type, targets in sorted(targets_by_algo.items()):
        for target in targets:
            print(f"\nEvaluating {algo_type}: {target['path']}")
            model, tokenizer = load_eval_model(target["path"], cfg)
            try:
                for ds_name, ds in datasets.items():
                    if not ds:
                        continue
                    for seed in cfg["seeds"]:
                        key = (ds_name, algo_type, int(target["step"]), int(seed), int(cfg["k"]))
                        if key in completed:
                            print(f"Skipping completed {algo_type} / {ds_name} / seed={seed}")
                            continue
                        p1, pk, is_correct = quick_eval(model, tokenizer, cfg, ds, int(seed))
                        per_seed_records.append(
                            {
                                "dataset": ds_name,
                                "algo_type": algo_type,
                                "ckpt": target["ckpt"],
                                "step": int(target["step"]),
                                "seed": int(seed),
                                "pass_at_1": float(p1),
                                "pass_at_k": float(pk),
                                "is_correct": is_correct,
                            }
                        )
                        completed.add(key)
                        save_results(json_out_path, int(cfg["k"]), per_seed_records)
            finally:
                del model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    if per_seed_records:
        save_results(json_out_path, int(cfg["k"]), per_seed_records)
    else:
        print("No evaluation records were produced.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    cfg = normalize_eval_config(load_yaml_config(args.config))
    eval_all(cfg)


if __name__ == "__main__":
    main()
