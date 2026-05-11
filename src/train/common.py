import argparse
import gc
import importlib.util
import os
import random

import numpy as np
import torch
from huggingface_hub import create_repo
from transformers import AutoModelForCausalLM, AutoTokenizer

from ..config import load_yaml_config, normalize_train_config
from .rewards import math_verify_reward

try:
    import wandb
except ImportError:
    wandb = None


reasoning_start = "<start_working_out>"
reasoning_end = "<end_working_out>"
solution_start = "<SOLUTION>"
solution_end = "</SOLUTION>"

system_prompt = f"""You are given a problem.
Think about the problem and provide your working out.
Place it between {reasoning_start} and {reasoning_end}.
Then, provide your solution between {solution_start}{solution_end}"""

CHAT_TEMPLATE = \
    "{% if messages[0]['role'] == 'system' %}"\
        "{{ messages[0]['content'] + eos_token }}"\
        "{% set loop_messages = messages[1:] %}"\
    "{% else %}"\
        "{{ '{system_prompt}' + eos_token }}"\
        "{% set loop_messages = messages %}"\
    "{% endif %}"\
    "{% for message in loop_messages %}"\
        "{% if message['role'] == 'user' %}"\
            "{{ message['content'] }}"\
        "{% elif message['role'] == 'assistant' %}"\
            "{{ message['content'] + eos_token }}"\
        "{% endif %}"\
    "{% endfor %}"\
    "{% if add_generation_prompt %}{{ '{reasoning_start}' }}"\
    "{% endif %}"


def build_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    return parser


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_torch_dtype() -> torch.dtype:
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    if torch.cuda.is_available():
        return torch.float16
    return torch.float32


def get_precision_kwargs() -> dict:
    if not torch.cuda.is_available():
        return {"bf16": False, "fp16": False}
    if torch.cuda.is_bf16_supported():
        return {"bf16": True, "fp16": False}
    return {"bf16": False, "fp16": True}


def load_model_and_tokenizer(model_name: str, max_seq_length: int):
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    tokenizer.model_max_length = max_seq_length

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=get_torch_dtype(),
        trust_remote_code=True,
    )
    model.config.use_cache = False
    if tokenizer.pad_token_id is not None:
        model.config.pad_token_id = tokenizer.pad_token_id
    return model, tokenizer


def resolve_num_rollouts(cfg: dict) -> int:
    candidates = (
        cfg.get("context_grpo", {}).get("num_rollouts"),
        cfg.get("grpo", {}).get("num_rollouts"),
        cfg.get("rl", {}).get("num_rollouts"),
        cfg.get("rl", {}).get("num_generations"),
        cfg.get("num_rollouts"),
    )
    for candidate in candidates:
        if isinstance(candidate, int) and candidate > 0:
            return candidate
    return 8


def resolve_vllm_flag(cfg: dict) -> bool:
    requested = bool(cfg.get("use_vllm", False))
    if not requested:
        return False

    if not torch.cuda.is_available():
        print("use_vllm=True but CUDA is unavailable. Falling back to use_vllm=False.")
        return False

    if importlib.util.find_spec("vllm") is None:
        print("use_vllm=True but vllm is not installed. Falling back to use_vllm=False.")
        return False
    return True


def resolve_optimizer_name(requested: str, stage: str) -> str:
    if "8bit" not in requested.lower():
        return requested

    if not torch.cuda.is_available():
        print(f"{stage} optimizer={requested} but CUDA is unavailable. Falling back to adamw_torch.")
        return "adamw_torch"

    if importlib.util.find_spec("bitsandbytes") is None:
        print(f"{stage} optimizer={requested} but bitsandbytes is not installed. Falling back to adamw_torch.")
        return "adamw_torch"

    return requested


def resolve_optimizer(cfg: dict) -> str:
    requested = str(cfg.get("optimizer", cfg.get("rl", {}).get("optim", "adamw_torch")))
    return resolve_optimizer_name(requested, stage="GRPO")


def resolve_sft_optimizer(cfg: dict) -> str:
    requested = str(cfg.get("sft", {}).get("optim", "adamw_torch"))
    return resolve_optimizer_name(requested, stage="SFT")


def resolve_full_ft(cfg: dict) -> bool:
    return bool(cfg.get("full_ft", cfg.get("do_full_ft", False)))


def apply_lora_if_needed(model, cfg: dict, full_ft: bool):
    if full_ft:
        print("Running full fine-tuning mode (LoRA disabled).")
        return model

    try:
        from peft import LoraConfig, get_peft_model
    except ImportError as exc:
        raise ImportError("Default training mode is LoRA. Install peft or set full_ft: true.") from exc

    rank = int(cfg.get("lora_rank", cfg.get("lora", {}).get("rank", 8)))
    target_modules = cfg.get(
        "lora_target_modules",
        cfg.get(
            "lora",
            {},
        ).get(
            "target_modules",
            ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        ),
    )
    lora_cfg = LoraConfig(
        r=rank,
        lora_alpha=int(cfg.get("lora_alpha", cfg.get("lora", {}).get("alpha", rank * 2))),
        lora_dropout=float(cfg.get("lora_dropout", cfg.get("lora", {}).get("dropout", 0.0))),
        bias=cfg.get("lora_bias", cfg.get("lora", {}).get("bias", "none")),
        task_type="CAUSAL_LM",
        target_modules=target_modules,
    )
    model = get_peft_model(model, lora_cfg)
    print(f"Running LoRA fine-tuning mode (rank={rank}). Set full_ft: true for full fine-tuning.")
    return model


def resolve_hf_upload_config(cfg: dict):
    hf_user_id = cfg.get("hf_user_id") or cfg.get("hf_username")
    hf_token = (
        cfg.get("hf_pat")
        or cfg.get("hf_token")
        or os.environ.get("HF_TOKEN")
        or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    )
    hf_upload = cfg.get("hf_upload")
    if hf_upload is None:
        hf_upload = bool(hf_user_id and hf_token)
    else:
        hf_upload = bool(hf_upload)
    hf_private = bool(cfg.get("hf_private", False))
    return hf_upload, hf_user_id, hf_token, hf_private


def push_model_to_hub_if_configured(model, tokenizer, cfg: dict, algorithm_name: str) -> None:
    hf_upload, hf_user_id, hf_token, hf_private = resolve_hf_upload_config(cfg)
    if not hf_upload:
        print("hf_upload=False, skipping Hugging Face upload.")
        return
    if not hf_user_id:
        print("hf_user_id is missing, skipping Hugging Face upload.")
        return
    if not hf_token:
        print("Hugging Face token is missing, skipping Hugging Face upload.")
        return

    repo_name = algorithm_name.replace("/", "-").replace(" ", "-")
    repo_id = f"{hf_user_id}/{repo_name}"
    create_repo(repo_id=repo_id, token=hf_token, private=hf_private, exist_ok=True)
    model.push_to_hub(repo_id=repo_id, token=hf_token)
    tokenizer.push_to_hub(repo_id=repo_id, token=hf_token)
    print(f"Pushed model to Hugging Face: https://huggingface.co/{repo_id}")


def _cfg_value(cfg: dict, key: str, default=None):
    if key in cfg:
        return cfg[key]
    return cfg.get("rl", {}).get(key, default)


def guide_name_from_cfg(cfg: dict):
    sage = cfg.get("sage", {}) or {}
    flags = {
        "random": bool(cfg.get("use_random", False)),
        "token": bool(cfg.get("use_token", False)),
        "branch": bool(cfg.get("use_branch", False)),
    }
    if "guide" in sage:
        guide_name = sage.get("guide")
        return None if guide_name in {"", "none", None} else guide_name
    enabled = [name for name, value in flags.items() if value]
    if len(enabled) > 1:
        raise ValueError(f"Only one SAGE guide can be enabled at a time, got: {enabled}")
    return enabled[0] if enabled else None


def build_algorithm_name(cfg: dict, use_sage: bool) -> str:
    explicit = cfg.get("algorithm_name")
    if explicit:
        return str(explicit)
    if use_sage:
        guide_name = guide_name_from_cfg(cfg)
        if guide_name is None:
            raise ValueError("train_sage requires sage.guide or one of use_random/use_token/use_branch.")
        algorithm_name = f"SAGE_{guide_name.capitalize()}"
    else:
        algorithm_name = "GRPO"

    return algorithm_name


def configure_model_for_training(model, use_gradient_checkpointing: bool) -> None:
    model.config.use_cache = False
    if not use_gradient_checkpointing:
        return

    if hasattr(model, "base_model"):
        model.base_model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
    else:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )

    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()


def format_sft_example(row: dict):
    expected_answer = str(row["expected_answer"])
    problem = row["problem"]
    thoughts = str(row["generated_solution"]).replace("<think>", "").replace("</think>", "").strip()
    final_prompt = reasoning_start + thoughts + reasoning_end + solution_start + expected_answer + solution_end
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": problem},
        {"role": "assistant", "content": final_prompt},
    ]


def run_sft_if_needed(model, tokenizer, cfg: dict, baseline_save_dir: str):
    if bool(cfg.get("instruct", True)):
        return model

    from datasets import Dataset, load_dataset
    from trl import SFTConfig, SFTTrainer
    import pandas as pd

    print("Model is not instruction tuned. Starting SFT before RL.")
    sft_cfg = cfg.get("sft", {})
    sft_chat_template = CHAT_TEMPLATE\
        .replace("'{system_prompt}'", f"'{system_prompt}'")\
        .replace("'{reasoning_start}'", f"'{reasoning_start}'")
    tokenizer.chat_template = sft_chat_template

    dataset_name = sft_cfg.get("dataset_name", "unsloth/OpenMathReasoning-mini")
    split = sft_cfg.get("split", "cot")
    dataset = load_dataset(dataset_name, split=split)
    dataset = dataset.to_pandas()[["expected_answer", "problem", "generated_solution"]]
    is_number = pd.to_numeric(pd.Series(dataset["expected_answer"]), errors="coerce").notnull()
    dataset = dataset.iloc[np.where(is_number)[0]].copy()
    dataset["Messages"] = dataset.apply(format_sft_example, axis=1)
    dataset["N"] = dataset["Messages"].apply(lambda x: len(tokenizer.apply_chat_template(x, tokenize=True)))
    max_seq_length = cfg.get("max_seq_len", cfg.get("max_seq_length", 2048))
    dataset = dataset.loc[dataset["N"] <= max_seq_length / 2].copy()
    max_sft_samples = sft_cfg.get("max_samples")
    if max_sft_samples is not None:
        dataset = dataset.iloc[: int(max_sft_samples)].copy()
    dataset["text"] = tokenizer.apply_chat_template(dataset["Messages"].tolist(), tokenize=False)
    dataset = Dataset.from_pandas(dataset)

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=dataset,
        args=SFTConfig(
            dataset_text_field="text",
            **get_precision_kwargs(),
            per_device_train_batch_size=sft_cfg.get("per_device_train_batch_size", 1),
            gradient_accumulation_steps=sft_cfg.get("gradient_accumulation_steps", 1),
            warmup_steps=sft_cfg.get("warmup_steps", 0),
            num_train_epochs=sft_cfg.get("num_epochs", 1),
            learning_rate=sft_cfg.get("learning_rate", 2e-5),
            logging_steps=sft_cfg.get("logging_steps", 10),
            optim=resolve_sft_optimizer(cfg),
            weight_decay=sft_cfg.get("weight_decay", 0.0),
            lr_scheduler_type=sft_cfg.get("lr_scheduler_type", "linear"),
            seed=cfg.get("seed", 42),
            report_to="none",
        ),
    )
    trainer.train()
    os.makedirs(baseline_save_dir, exist_ok=True)
    model.save_pretrained(baseline_save_dir)
    tokenizer.save_pretrained(baseline_save_dir)
    push_model_to_hub_if_configured(model, tokenizer, cfg, "baseline")
    del dataset
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("SFT done.")
    return model


def load_training_dataset(cfg: dict):
    from datasets import load_dataset

    dataset_name = cfg.get("train_dataset", "open-r1/DAPO-Math-17k-Processed")
    dataset_config = cfg.get("train_dataset_config", "en")
    split = cfg.get("train_split", "train")
    if dataset_config:
        dataset = load_dataset(dataset_name, dataset_config, split=split)
    else:
        dataset = load_dataset(dataset_name, split=split)
    dataset = dataset.map(
        lambda x: {
            "prompt": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": x["prompt"]},
            ],
            "answer": x["solution"],
        }
    )

    max_train_samples = cfg.get("max_train_samples")
    if max_train_samples is not None:
        dataset = dataset.select(range(min(int(max_train_samples), len(dataset))))
    return dataset


def setup_wandb(cfg: dict, run_name: str):
    report_to = []
    if cfg.get("wandb_enabled") is False:
        print("wandb_enabled=False, disabling wandb logging for GRPO.")
        return report_to

    if wandb is None:
        print("wandb is not installed. Disabling wandb logging for GRPO.")
        return report_to

    wandb_api_key = (
        cfg.get("wandb_api_key")
        or cfg.get("WANDB_API_KEY")
        or os.environ.get("WANDB_API_KEY")
    )
    if wandb_api_key:
        os.environ["WANDB_API_KEY"] = wandb_api_key

    os.environ["WANDB_PROJECT"] = cfg.get("wandb_project", "RBT")
    os.environ["WANDB_NAME"] = run_name
    if wandb_api_key:
        wandb.login(key=wandb_api_key)
    else:
        wandb.login()
    return ["wandb"]


def build_grpo_config(
    cfg: dict,
    *,
    use_gradient_checkpointing: bool,
    use_kl: bool,
    report_to,
    output_dir: str,
):
    from trl import GRPOConfig

    use_vllm = resolve_vllm_flag(cfg)
    generation_kwargs = {}
    if use_vllm:
        generation_kwargs.update(
            {
                "stop": [solution_end],
                "include_stop_str_in_output": True,
            }
        )

    return GRPOConfig(
        seed=cfg.get("random_state", cfg.get("seed", 42)),
        **get_precision_kwargs(),
        use_vllm=use_vllm,
        vllm_mode=cfg.get("vllm_mode", "colocate"),
        gradient_checkpointing=use_gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        temperature=_cfg_value(cfg, "temperature", 1.0),
        min_p=cfg.get("min_p", cfg.get("vllm", {}).get("min_p", 0.01)),
        top_p=cfg.get("top_p", cfg.get("vllm", {}).get("top_p", 1.0)),
        top_k=cfg.get("top_k", cfg.get("vllm", {}).get("top_k", -1)),
        generation_kwargs=generation_kwargs,
        learning_rate=_cfg_value(cfg, "learning_rate", 2e-6),
        lr_scheduler_type=cfg.get("lr_scheduler", cfg.get("rl", {}).get("lr_scheduler_type", "constant")),
        warmup_ratio=_cfg_value(cfg, "warmup_ratio", 0.1),
        weight_decay=_cfg_value(cfg, "weight_decay", 0.01),
        loss_type=cfg.get("loss_type", "grpo"),
        beta=cfg.get("beta", cfg.get("rl", {}).get("kl_coef", 0.05)) if use_kl else 0.0,
        epsilon_high=float(_cfg_value(cfg, "epsilon", 0.2)),
        optim=resolve_optimizer(cfg),
        logging_steps=_cfg_value(cfg, "logging_steps", 1),
        per_device_train_batch_size=_cfg_value(cfg, "per_device_train_batch_size", 8),
        gradient_accumulation_steps=_cfg_value(cfg, "gradient_accumulation_steps", 1),
        num_generations=resolve_num_rollouts(cfg),
        generation_batch_size=cfg.get("generation_batch_size", _cfg_value(cfg, "per_device_train_batch_size", 8)),
        max_prompt_length=cfg.get("max_prompt_length", 1024),
        max_completion_length=cfg.get("max_completion_length", 1024),
        max_steps=cfg.get("max_steps", cfg.get("rl", {}).get("max_steps", 300)),
        save_steps=cfg.get("save_steps", cfg.get("rl", {}).get("save_steps", 50)),
        report_to=report_to,
        output_dir=output_dir,
    )


def run_training(args, *, use_sage: bool) -> None:
    cfg = normalize_train_config(load_yaml_config(args.config))

    seed = cfg.get("seed", 42)
    model_name = cfg["model_name"]
    max_seq_length = cfg.get("max_seq_len", cfg.get("max_seq_length", 2048))
    use_gradient_checkpointing = bool(cfg.get("gradient_checkpointing", True))
    use_kl = bool(cfg.get("use_kl", True))
    full_ft = resolve_full_ft(cfg)

    set_seed(seed)
    print(f"Training with seed {seed}")

    model, tokenizer = load_model_and_tokenizer(model_name, max_seq_length)
    model = apply_lora_if_needed(model, cfg, full_ft)
    configure_model_for_training(model, use_gradient_checkpointing)

    print(f"Gradient checkpointing: {'enabled' if use_gradient_checkpointing else 'disabled'}")
    if hasattr(model, "print_trainable_parameters"):
        model.print_trainable_parameters()
    else:
        print("Model does not expose print_trainable_parameters(); skipping.")

    model_id = model_name.split("/")[-1]
    output_root = cfg.get("output_root", "outputs/models")
    baseline_save_dir = os.path.join(output_root, model_id, "baseline")
    algorithm_name = build_algorithm_name(cfg, use_sage=use_sage)
    model_save_dir = os.path.join(output_root, model_id, algorithm_name)
    ckpt_dir = os.path.join(model_save_dir, "ckpts")
    os.makedirs(model_save_dir, exist_ok=True)

    model = run_sft_if_needed(model, tokenizer, cfg, baseline_save_dir)

    print(f"Starting {'SAGE' if use_sage else 'GRPO'} training")
    dataset = load_training_dataset(cfg)
    report_to = setup_wandb(cfg, algorithm_name)
    training_args = build_grpo_config(
        cfg,
        use_gradient_checkpointing=use_gradient_checkpointing,
        use_kl=use_kl,
        report_to=report_to,
        output_dir=ckpt_dir,
    )

    if use_sage:
        from .get_trainer import get_trainer

        trainer = get_trainer(
            model=model,
            tokenizer=tokenizer,
            training_args=training_args,
            dataset=dataset,
            reward=math_verify_reward,
            cfg=cfg,
        )
    else:
        from trl import GRPOTrainer

        trainer = GRPOTrainer(
            model=model,
            processing_class=tokenizer,
            reward_funcs=math_verify_reward,
            train_dataset=dataset,
            args=training_args,
        )

    trainer.train()
    model.save_pretrained(model_save_dir)
    tokenizer.save_pretrained(model_save_dir)
    push_model_to_hub_if_configured(model, tokenizer, cfg, algorithm_name)

    del dataset
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
