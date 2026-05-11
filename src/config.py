from collections.abc import Mapping
from copy import deepcopy
from typing import Any

import yaml


def load_yaml_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _section(cfg: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    value = cfg.get(name, {})
    return value if isinstance(value, Mapping) else {}


def _set_missing(cfg: dict, dst_key: str, section: Mapping[str, Any], *src_keys: str) -> None:
    if dst_key in cfg:
        return
    for src_key in src_keys:
        if src_key in section:
            cfg[dst_key] = section[src_key]
            return


def normalize_train_config(raw_cfg: Mapping[str, Any]) -> dict:
    cfg = deepcopy(dict(raw_cfg or {}))

    project = _section(cfg, "project")
    model = _section(cfg, "model")
    data = _section(cfg, "data")
    generation = _section(cfg, "generation")
    rl = _section(cfg, "rl")
    tracking = _section(cfg, "tracking")
    hub = _section(cfg, "hub")

    _set_missing(cfg, "seed", project, "seed")
    _set_missing(cfg, "output_root", project, "output_root")

    _set_missing(cfg, "model_name", model, "name", "model_name")
    _set_missing(cfg, "instruct", model, "instruction_tuned", "instruct")
    _set_missing(cfg, "full_ft", model, "full_finetune", "full_ft")
    _set_missing(cfg, "max_seq_len", model, "max_seq_len", "max_seq_length")
    _set_missing(cfg, "gradient_checkpointing", model, "gradient_checkpointing")

    _set_missing(cfg, "train_dataset", data, "train_dataset", "dataset")
    _set_missing(cfg, "train_dataset_config", data, "train_dataset_config", "dataset_config")
    _set_missing(cfg, "train_split", data, "train_split", "split")
    _set_missing(cfg, "max_train_samples", data, "max_train_samples")

    _set_missing(cfg, "use_vllm", generation, "use_vllm")
    _set_missing(cfg, "vllm_mode", generation, "vllm_mode")
    _set_missing(cfg, "temperature", generation, "temperature")
    _set_missing(cfg, "min_p", generation, "min_p")
    _set_missing(cfg, "top_p", generation, "top_p")
    _set_missing(cfg, "top_k", generation, "top_k")
    _set_missing(cfg, "generation_batch_size", generation, "batch_size", "generation_batch_size")
    _set_missing(cfg, "max_prompt_length", generation, "max_prompt_length")
    _set_missing(cfg, "max_completion_length", generation, "max_completion_length")

    _set_missing(cfg, "num_rollouts", rl, "num_rollouts", "num_generations")
    _set_missing(cfg, "random_state", rl, "trainer_seed", "seed", "random_state")
    _set_missing(cfg, "max_steps", rl, "max_steps")
    _set_missing(cfg, "save_steps", rl, "save_steps")
    _set_missing(cfg, "learning_rate", rl, "learning_rate")
    _set_missing(cfg, "lr_scheduler", rl, "lr_scheduler", "lr_scheduler_type")
    _set_missing(cfg, "warmup_ratio", rl, "warmup_ratio")
    _set_missing(cfg, "weight_decay", rl, "weight_decay")
    _set_missing(cfg, "loss_type", rl, "loss_type")
    _set_missing(cfg, "use_kl", rl, "use_kl")
    _set_missing(cfg, "beta", rl, "beta", "kl_coef")
    _set_missing(cfg, "epsilon", rl, "epsilon", "epsilon_high")
    _set_missing(cfg, "optimizer", rl, "optimizer", "optim")
    _set_missing(cfg, "logging_steps", rl, "logging_steps")
    _set_missing(cfg, "per_device_train_batch_size", rl, "per_device_train_batch_size")
    _set_missing(cfg, "gradient_accumulation_steps", rl, "gradient_accumulation_steps")

    _set_missing(cfg, "wandb_enabled", tracking, "enabled", "wandb_enabled")
    _set_missing(cfg, "wandb_project", tracking, "project", "wandb_project")
    _set_missing(cfg, "wandb_api_key", tracking, "api_key", "wandb_api_key")

    _set_missing(cfg, "hf_upload", hub, "upload", "hf_upload")
    _set_missing(cfg, "hf_user_id", hub, "user_id", "hf_user_id", "username")
    _set_missing(cfg, "hf_token", hub, "token", "hf_token")
    _set_missing(cfg, "hf_private", hub, "private", "hf_private")

    return cfg


def normalize_eval_config(raw_cfg: Mapping[str, Any]) -> dict:
    cfg = deepcopy(dict(raw_cfg or {}))

    model = _section(cfg, "model")
    target = _section(cfg, "target")
    data = _section(cfg, "data")
    generation = _section(cfg, "generation")
    runtime = _section(cfg, "runtime")
    output = _section(cfg, "output")

    _set_missing(cfg, "model_name", model, "name", "model_name")
    _set_missing(cfg, "output_root", model, "output_root")
    _set_missing(cfg, "model_root", model, "model_root")
    _set_missing(cfg, "full_ft", model, "full_finetune", "full_ft")
    _set_missing(cfg, "use_lora", model, "use_lora")
    _set_missing(cfg, "tokenizer_name", model, "tokenizer_name")

    _set_missing(cfg, "include_baseline", target, "include_baseline")
    _set_missing(cfg, "eval_steps", target, "eval_steps")
    _set_missing(cfg, "model_path", target, "model_path")
    _set_missing(cfg, "model_label", target, "model_label", "label")
    _set_missing(cfg, "model_step", target, "model_step", "step")
    _set_missing(cfg, "algorithms", target, "algorithms")

    _set_missing(cfg, "datasets", data, "datasets")
    _set_missing(cfg, "eval_limit", data, "eval_limit", "limit")

    _set_missing(cfg, "k", generation, "k")
    _set_missing(cfg, "generation_batch_size", generation, "batch_size", "generation_batch_size")
    _set_missing(cfg, "temperature", generation, "temperature")
    _set_missing(cfg, "top_p", generation, "top_p")
    _set_missing(cfg, "max_new_tokens", generation, "max_new_tokens", "max_tokens")

    _set_missing(cfg, "seed", runtime, "seed")
    _set_missing(cfg, "seeds", runtime, "seeds")
    _set_missing(cfg, "torch_dtype", runtime, "torch_dtype")
    _set_missing(cfg, "device_map", runtime, "device_map")

    _set_missing(cfg, "json_out_path", output, "json_out_path", "path")

    return cfg
