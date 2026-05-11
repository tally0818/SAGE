import math
from typing import Optional

import torch


class CosineAnnealingScheduler:
    """Cosine scheduler used by the stochastic SAGE guide variants."""

    def __init__(
        self,
        max_steps: int,
        init_eps: float,
        lb_eps: float,
        init_sig: float,
        lb_sig: float,
        num_cycles: int,
        decay_rate: float = 0.9,
    ):
        if num_cycles < 1:
            raise ValueError("num_cycles must be >= 1")
        self.max_steps = int(max_steps)
        self.init_eps = float(init_eps)
        self.lb_eps = float(lb_eps)
        self.init_sig = float(init_sig)
        self.lb_sig = float(lb_sig)
        self.num_cycles = int(num_cycles)
        self.decay_rate = float(decay_rate)
        self.cycle_length = max(1, self.max_steps // self.num_cycles)
        self.step_count = 0
        self.cycle = 0

    def step(self) -> None:
        self.step_count += 1
        if (self.step_count % self.cycle_length == 0) and (self.cycle < self.num_cycles - 1):
            self.cycle += 1
            self.init_eps *= self.decay_rate
            self.init_sig *= self.decay_rate

    def sample(self):
        t = (self.step_count % self.cycle_length) / self.cycle_length
        anneal = 0.5 * (1.0 + math.cos(math.pi * t))
        eps = self.lb_eps + (self.init_eps - self.lb_eps) * anneal
        sig = self.lb_sig + (self.init_sig - self.lb_sig) * anneal
        return float(eps), float(sig)


class GuideFunction:
    """Base interface for SAGE guide functions."""

    name = "base"

    def sample(
        self,
        *,
        new: torch.Tensor,
        ref: torch.Tensor,
        mask: torch.Tensor,
        entropies: Optional[torch.Tensor] = None,
        ref_entropies: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        raise NotImplementedError

    @staticmethod
    def _apply_mask(q: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if q.shape != mask.shape:
            raise ValueError(f"q shape {tuple(q.shape)} does not match mask shape {tuple(mask.shape)}")
        mask = mask.to(dtype=q.dtype)
        return q * mask + (1.0 - mask)


class RandomGuideFunction(GuideFunction):
    name = "random"

    def __init__(self, scheduler: CosineAnnealingScheduler, q_min: float, q_max: float):
        self.scheduler = scheduler
        self.q_min = float(q_min)
        self.q_max = float(q_max)

    @torch.no_grad()
    def sample(
        self,
        *,
        new: torch.Tensor,
        ref: torch.Tensor,
        mask: torch.Tensor,
        entropies: Optional[torch.Tensor] = None,
        ref_entropies: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        eps, std = self.scheduler.sample()
        self.scheduler.step()

        q = torch.ones_like(ref, dtype=torch.float32)
        perturb = torch.rand_like(q) < eps
        noise = 1.0 + std * torch.randn_like(q)
        q = torch.where(perturb, noise, q).clamp(self.q_min, self.q_max)
        return self._apply_mask(q, mask)


class TokenLevelGuideFunction(GuideFunction):
    name = "token"

    def __init__(
        self,
        scheduler: CosineAnnealingScheduler,
        use_source: str,
        norm: str,
        q_min: float,
        q_max: float,
        detach_surprisal: bool = True,
    ):
        if use_source not in {"new", "ref"}:
            raise ValueError("use_source must be 'new' or 'ref'")
        if norm not in {"rank", "minmax", "zscore"}:
            raise ValueError("norm must be one of: rank, minmax, zscore")
        self.scheduler = scheduler
        self.use_source = use_source
        self.norm = norm
        self.q_min = float(q_min)
        self.q_max = float(q_max)
        self.detach_surprisal = bool(detach_surprisal)

    @torch.no_grad()
    def _normalize(self, surprisal: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        mask = mask.bool()
        weights = torch.zeros_like(surprisal, dtype=torch.float32)
        for row_idx in range(surprisal.size(0)):
            valid = mask[row_idx]
            if not valid.any():
                continue

            vals = surprisal[row_idx, valid].float()
            if self.norm == "rank":
                ranks = torch.argsort(torch.argsort(vals))
                denom = max(1, int(valid.sum().item()) - 1)
                weights[row_idx, valid] = ranks.float() / denom
            elif self.norm == "minmax":
                low = torch.quantile(vals, 0.05)
                high = torch.quantile(vals, 0.95)
                vals = vals.clamp(low, high)
                span = vals.max() - vals.min()
                weights[row_idx, valid] = (vals - vals.min()) / span.clamp_min(1e-6)
            else:
                std = vals.std(unbiased=False).clamp_min(1e-6)
                weights[row_idx, valid] = torch.sigmoid((vals - vals.mean()) / std)
        return weights.clamp(0.0, 1.0)

    @torch.no_grad()
    def sample(
        self,
        *,
        new: torch.Tensor,
        ref: torch.Tensor,
        mask: torch.Tensor,
        entropies: Optional[torch.Tensor] = None,
        ref_entropies: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        alpha, sigma = self.scheduler.sample()
        self.scheduler.step()

        source = new if self.use_source == "new" else ref
        surprisal = -source.float()
        if self.detach_surprisal:
            surprisal = surprisal.detach()

        weights = self._normalize(surprisal, mask)
        mean = 1.0 + alpha * weights
        std = sigma * weights
        q = (mean + std * torch.randn_like(mean)).clamp(self.q_min, self.q_max)
        return self._apply_mask(q, mask)


class BranchLevelGuideFunction(GuideFunction):
    name = "branch"

    def __init__(
        self,
        ratio: float,
        threshold: float,
        use_source: str,
        q_min: float,
        q_max: float,
        detach_entropy: bool = True,
    ):
        if use_source not in {"new", "ref"}:
            raise ValueError("use_source must be 'new' or 'ref'")
        self.ratio = float(ratio)
        self.threshold = float(threshold)
        self.use_source = use_source
        self.q_min = float(q_min)
        self.q_max = float(q_max)
        self.detach_entropy = bool(detach_entropy)

    @torch.no_grad()
    def sample(
        self,
        *,
        new: torch.Tensor,
        ref: torch.Tensor,
        mask: torch.Tensor,
        entropies: Optional[torch.Tensor] = None,
        ref_entropies: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.use_source == "ref":
            if ref_entropies is None:
                raise ValueError(
                    "BranchLevelGuideFunction(use_source='ref') requires ref_entropies, "
                    "but this trainer path supplies policy entropies only. Use use_source='new'."
                )
            entropy = ref_entropies
        else:
            if entropies is None:
                raise ValueError("BranchLevelGuideFunction requires token entropies from the policy model.")
            entropy = entropies

        entropy = entropy.float()
        if self.detach_entropy:
            entropy = entropy.detach()

        q = 1.0 + self.ratio * torch.relu(entropy - self.threshold)
        q = q.clamp(self.q_min, self.q_max)
        return self._apply_mask(q, mask)
