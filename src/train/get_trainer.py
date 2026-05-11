from typing import Optional

from trl import GRPOTrainer

from .guide_functions import (
    BranchLevelGuideFunction,
    CosineAnnealingScheduler,
    RandomGuideFunction,
    TokenLevelGuideFunction,
)
from .sage_trainer import SAGETrainer


def _as_bool(cfg: dict, key: str, default: bool = False) -> bool:
    return bool(cfg.get(key, default))


def _sage_cfg(cfg: dict) -> dict:
    return cfg.get("sage", {}) or {}


def _rl_cfg(cfg: dict) -> dict:
    return cfg.get("rl", {}) or {}


def guide_name_from_cfg(cfg: dict) -> Optional[str]:
    sage = _sage_cfg(cfg)
    flag_names = {
        "random": _as_bool(cfg, "use_random"),
        "token": _as_bool(cfg, "use_token"),
        "branch": _as_bool(cfg, "use_branch"),
    }

    if "guide" in sage:
        guide_name = sage.get("guide")
        if guide_name in {"", "none", None}:
            return None
        if guide_name not in flag_names:
            raise ValueError("sage.guide must be one of: random, token, branch, none")
        return guide_name

    enabled_flags = [name for name, enabled in flag_names.items() if enabled]
    if len(enabled_flags) > 1:
        raise ValueError(f"Only one SAGE guide can be enabled at a time, got: {enabled_flags}")
    if enabled_flags:
        return enabled_flags[0]
    return None


def _common_q_bounds(cfg: dict):
    sage = _sage_cfg(cfg)
    rl = _rl_cfg(cfg)
    q_min = sage.get("q_min", rl.get("q_min", 0.6))
    q_max = sage.get("q_max", rl.get("q_max", 1.4))
    return float(q_min), float(q_max)


def _scheduler_from_cfg(raw_cfg: dict, max_steps: int, prefix: str) -> CosineAnnealingScheduler:
    return CosineAnnealingScheduler(
        max_steps=max_steps,
        init_eps=raw_cfg[f"{prefix}_ub"],
        lb_eps=raw_cfg[f"{prefix}_lb"],
        init_sig=raw_cfg["sig_ub"],
        lb_sig=raw_cfg["sig_lb"],
        num_cycles=raw_cfg.get("N", raw_cfg.get("num_cycles", 1)),
        decay_rate=raw_cfg.get("decay_rate", 0.9),
    )


def build_guide_function(cfg: dict, max_steps: int):
    guide_name = guide_name_from_cfg(cfg)
    if guide_name is None:
        return None

    if _as_bool(cfg, "use_fkl"):
        raise NotImplementedError("FKLTrainer is not implemented in the TRL-native SAGE path.")

    use_kl = _as_bool(cfg, "use_kl", default=True)
    if not use_kl:
        raise ValueError("SAGE guide functions require use_kl=true because they reshape the KL anchor.")

    sage = _sage_cfg(cfg)
    rl = _rl_cfg(cfg)
    q_min, q_max = _common_q_bounds(cfg)

    if guide_name == "branch":
        branch = sage.get("branch", rl.get("branch", {}))
        return BranchLevelGuideFunction(
            ratio=branch.get("ratio", 0.3),
            threshold=branch.get("threshold", 1.2),
            use_source=branch.get("use_source", "new"),
            q_min=q_min,
            q_max=q_max,
            detach_entropy=branch.get("detach_entropy", True),
        )

    if guide_name == "random":
        random_cfg = sage.get("random", rl.get("random", {}))
        scheduler = _scheduler_from_cfg(random_cfg, max_steps=max_steps, prefix="eps")
        return RandomGuideFunction(scheduler=scheduler, q_min=q_min, q_max=q_max)

    if guide_name == "token":
        token = sage.get("token", rl.get("token", {}))
        scheduler = _scheduler_from_cfg(token, max_steps=max_steps, prefix="alpha")
        return TokenLevelGuideFunction(
            scheduler=scheduler,
            use_source=token.get("use_source", "new"),
            norm=token.get("norm", "minmax"),
            q_min=q_min,
            q_max=q_max,
            detach_surprisal=token.get("detach_surprisal", True),
        )

    raise ValueError(f"Unknown SAGE guide: {guide_name}")


def get_trainer(model, tokenizer, training_args, dataset, reward, cfg):
    base_kwargs = dict(
        model=model,
        processing_class=tokenizer,
        reward_funcs=reward,
        args=training_args,
        train_dataset=dataset,
    )
    guide_function = build_guide_function(cfg, max_steps=getattr(training_args, "max_steps", cfg.get("max_steps", 0)))
    if guide_function is None:
        return GRPOTrainer(**base_kwargs)

    if getattr(training_args, "beta", 0.0) == 0.0:
        raise ValueError("SAGETrainer received beta=0.0; set use_kl=true and beta/rl.kl_coef > 0.")
    print(f"Using SAGETrainer with {guide_function.name} guide.")
    return SAGETrainer(guide_function=guide_function, **base_kwargs)
