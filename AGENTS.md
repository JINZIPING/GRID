# Repository Guidelines

## Project Structure & Module Organization
- `src/` contains the training and inference entry points plus reusable modules (`components/`, `models/`, `modules/`, `utils/`); keep new code in these subpackages to benefit from shared utilities.
- `configs/` stores Hydra configuration trees; create overrides in `configs/experiment/` instead of hard-coding parameters in Python.
- Data artifacts live under `data/`, while intermediate outputs and checkpoints are written to `outputs/`, `logs/`, or `model_local/`; never commit large generated files.
- Place reusable notebooks or diagnostics alongside `notices.txt` only if they are dependency disclosures; all other documentation belongs with the relevant module.

## Build, Test, and Development Commands
- `pip install -r requirements.txt` sets up the full environment (CUDA wheels are pinned; activate a Python 3.10+ virtualenv first).
- `python -m src.train experiment=tiger_train_flat data_dir=...` launches training with Hydra; pass overrides such as `trainer.fast_dev_run=True` during iteration.
- `python -m src.inference experiment=tiger_inference_flat data_dir=... ckpt_path=...` generates recommendation sequences from a trained checkpoint.
- `python -m src.inference experiment=rkmeans_inference_flat embedding_path=...` converts embeddings into semantic IDs after residual quantization.

## Coding Style & Naming Conventions
- Follow PEP 8 with 4-space indentation, type hints, and descriptive docstrings for pipeline entry points.
- Use snake_case for modules, configs, and Hydra parameters; reserve PascalCase for Lightning modules or dataclasses.
- Prefer the project `RankedLogger` for structured logging and keep logging noise at the INFO level.
- Configuration constants belong in YAML under `configs/`; avoid hard-coded paths in Python files.

## Testing Guidelines
- Write pytest test cases under a `tests/` package mirroring the structure of `src/`; integration tests can call Hydra modules with `trainer.fast_dev_run=True` for speed.
- Run `python -m pytest` locally before submitting; use `-k <pattern>` to target slow suites and add new markers where helpful.
- Capture minimal fixture data in `data/` subfolders and clean it up after tests to keep the repository lean.

## Commit & Pull Request Guidelines
- Match the existing history by using concise, present-tense summaries and referencing issues (e.g., `addressing issue #12`).
- Squash or rebase before opening a PR so each change maps cleanly to a problem statement.
- PR descriptions should include the Hydra command used, notable config overrides, and links to relevant logs or metrics.
- Attach screenshots only when UI-facing artefacts change; otherwise share scalar metrics or table snippets in Markdown.

## Configuration & Environment Tips
- Hydra composes configs in the order defined in `train.yaml` and `inference.yaml`; add new defaults there when introducing modules.
- `rootutils` creates the `PROJECT_ROOT` env variable—use it together with `configs/paths/default.yaml` for portable file references.
- Local jobs run through `LocalJobLauncher`; prefer it for multi-stage experiments so cleanup hooks execute reliably.
