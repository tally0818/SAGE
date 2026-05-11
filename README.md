# SAGE: Shaping Anchors for Guided Exploration in RLVR of LLMs

Official implementation of **SAGE: Shaping Anchors for Guided Exploration in RLVR of LLMs**.

SAGE is a framework for improving exploration in reinforcement learning with verifiable rewards (RLVR). Instead of removing the stabilizing reverse-KL anchor, SAGE reshapes the anchor distribution with a guide function `q(x, y)`, replacing the usual anchor `pi_ref` with `q * pi_ref`. This keeps the stabilizing role of reverse-KL while encouraging controlled empirical support expansion.

## Overview

This repository provides a TRL-native implementation of SAGE:

- SAGE training on top of `trl.GRPOTrainer`
- Plain GRPO baseline training
- LoRA and full fine-tuning modes
- Optional pre-RL SFT for non-instruction-tuned base models
- Math reward verification with `math-verify`
- Evaluation on MATH-500, AIME, AMC23, and Minerva

The default configuration targets `deepseek-ai/DeepSeek-R1-Distill-Qwen-7B`.

## Method

Standard reverse-KL RLVR optimizes against a fixed reference model. This often improves `pass@1`, but can reduce coverage and limit gains in `pass@k` because the policy remains anchored to high-density modes of the reference distribution.

SAGE modifies the KL anchor with a positive guide function:

```text
pi_anchor(y | x) = q(x, y) * pi_ref(y | x)
```

The guide function is treated as fixed during each policy update and can be computed from intrinsic signals available during generation. This implementation includes three guide variants:

| Guide | Signal | Purpose |
| --- | --- | --- |
| `random` | Scheduled random perturbation | Generic exploration baseline |
| `token` | Token surprisal | Encourages uncommon lexical choices |
| `branch` | Token entropy | Encourages exploration at high-uncertainty reasoning branches |

The branch-level guide is the main SAGE variant used by the default config.

## Repository Structure

```text
configs/
  train_config.yaml        # Training config for SAGE and GRPO
  eval_config.yaml         # Evaluation config
scripts/
  train_sage.sh            # Launch SAGE training
  train_grpo.sh            # Launch plain GRPO training
  eval_model.sh            # Launch evaluation
src/
  config.py                # YAML loading and config normalization
  data/loaders.py          # Evaluation dataset loaders
  train/
    common.py              # Shared training pipeline
    guide_functions.py     # SAGE guide functions
    sage_trainer.py        # SAGETrainer implementation
    get_trainer.py         # Trainer factory
    rewards.py             # Math verification rewards
    train_sage.py          # SAGE entry point
    train_grpo.py          # GRPO entry point
  eval/eval_model.py       # pass@1 and pass@k evaluation
```

## Installation

Create a Python 3.10+ environment and install the dependencies:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

For GPU training, use a CUDA-compatible PyTorch installation. `bitsandbytes` and `vllm` are listed as Linux-only dependencies in `requirements.txt`; on non-Linux systems the code falls back when those packages are unavailable.

If you use gated Hugging Face models or want to upload checkpoints, authenticate first:

```bash
huggingface-cli login
```

## Training

### SAGE

Run the default SAGE configuration:

```bash
bash scripts/train_sage.sh
```

The default guide is configured in `configs/train_config.yaml`:

```yaml
sage:
  guide: branch
```

Available values are `branch`, `token`, and `random`.

### Plain GRPO Baseline

Run the GRPO baseline with the same training config:

```bash
bash scripts/train_grpo.sh
```

### SFT Before RL

If `model.instruction_tuned` is set to `false`, the training pipeline first runs a lightweight SFT stage using the dataset configured under `sft`, then starts RLVR from the resulting baseline checkpoint.

```yaml
model:
  instruction_tuned: false

sft:
  dataset_name: unsloth/OpenMathReasoning-mini
  split: cot
```

This is useful for reproducing the two-stage setup used for non-instruction-tuned base models.

## Configuration

The main training settings live in `configs/train_config.yaml`.

Important sections:

- `model`: base model, fine-tuning mode, sequence length, gradient checkpointing
- `lora`: LoRA rank and optional adapter settings
- `data`: RLVR training dataset and split
- `generation`: rollout sampling and vLLM settings
- `rl`: GRPO/DAPO/BNPO-style optimization settings exposed through TRL
- `sage`: guide function type and guide-specific hyperparameters
- `tracking`: Weights & Biases logging
- `hub`: optional Hugging Face Hub upload

By default, model outputs are saved under:

```text
outputs/models/<model_name>/<run_name>/
```

Intermediate checkpoints are saved under:

```text
outputs/models/<model_name>/<run_name>/ckpts/
```

Run names are intentionally simple: `GRPO` for the baseline and `SAGE_<Guide>` for SAGE runs, such as `SAGE_Branch`. Set a top-level `algorithm_name` in the training config to override this directory name.

## Evaluation

Edit `configs/eval_config.yaml`, then run:

```bash
bash scripts/eval_model.sh
```

The evaluator can either auto-discover checkpoints under `outputs/models` or evaluate an explicit checkpoint path:

```yaml
target:
  model_path: outputs/models/DeepSeek-R1-Distill-Qwen-7B/SAGE_Branch
  model_label: SAGE_Branch
  model_step: 0
```

To evaluate discovered checkpoints, leave `model_path: null` and optionally filter algorithms:

```yaml
target:
  include_baseline: false
  eval_steps: false
  algorithms:
    - SAGE_Branch
```

Evaluation results are written to:

```text
outputs/eval/<model_name>/evaluation_results.json
```

The JSON contains per-seed records and aggregate `pass@1` / `pass@k` statistics.

## Datasets

Training defaults to:

- `open-r1/DAPO-Math-17k-Processed`

Evaluation loaders are available for:

- `MATH-500`
- `AIME`
- `AMC23`
- `Minerva`

Some evaluation loaders download data from Hugging Face or public GitHub URLs at runtime.

## Paper Results

The paper reports that branch-level SAGE improves the accuracy-coverage trade-off across mathematical reasoning benchmarks.

Average results for the GRPO family on AIME, AMC23, and MATH-500 with Qwen2.5-Math-7B-Base:

| Method | Avg pass@1 | Avg pass@256 |
| --- | ---: | ---: |
| Base Model | 0.227 | 0.708 |
| GRPO | 0.266 | 0.714 |
| GRPO + Branch | 0.284 | 0.729 |

Algorithm ablation with Qwen2.5-Math-7B-Base:

| Method | Avg pass@1 | Avg pass@256 |
| --- | ---: | ---: |
| Base Model | 0.227 | 0.708 |
| BNPO (no KL) | 0.243 | 0.699 |
| BNPO + Branch | 0.271 | 0.715 |
| DAPO (no KL) | 0.264 | 0.696 |
| DAPO + Branch | 0.267 | 0.712 |

Across BNPO and DAPO, adding the branch-level SAGE guide improves the accuracy-coverage trade-off, recovering or surpassing baseline high-sample coverage while preserving pass@1 gains.

Model ablation with DeepSeek-R1-Distill-Qwen-7B:

| Method | Avg pass@1 | Avg pass@64 |
| --- | ---: | ---: |
| Base Model | 0.429 | 0.778 |
| GRPO | 0.449 | 0.771 |
| GRPO + Branch | 0.452 | 0.774 |

## Citation

If you use this repository, please cite:

```bibtex
@inproceedings{lee2026sage,
  title = {SAGE: Shaping Anchors for Guided Exploration in RLVR of LLMs},
  author = {Lee, Chanuk and Kang, Minki and Hwang, Sung Ju},
  booktitle = {Proceedings of the 43rd International Conference on Machine Learning},
  year = {2026},
  organization = {PMLR}
}
```

## Acknowledgements

This code builds on the Hugging Face ecosystem, including `transformers`, `trl`, `peft`, `datasets`, `huggingface_hub`, and `math-verify`.