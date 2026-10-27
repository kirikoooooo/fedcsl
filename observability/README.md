# `observability` package

Dependency-light runtime for the FedCSL Argo/Prometheus/Grafana pipeline.

## Modules

| Module | Purpose |
|---|---|
| `prepare` | validate/materialize a dataset and write `dataset.json` |
| `stage` | run one phase, stream logs, write `stages/<phase>.json` |
| `evaluate` | validate result logs and write `eval_summary.json` |
| `exporter` | Prometheus file-collector on `:9108` (stdlib only) |
| `progress` | per-round JSON writer called from `FedCSL_All.py` |
| `pipeline` | local executor for prepare → pretrain → finetune → evaluate |
| `common` | atomic JSON, timestamps, path helpers |

## Commands

```bash
# local closed loop (uses the CSL conda env in this repo)
python -m observability.pipeline \
  --dataset Epilepsy-TSTCC --config config/configACF.yml \
  --num-rounds 2 --num-client 3 --alpha 0.1 \
  --ft-ratios 0.1,1.0 --ft-seeds 1

# exporter (separate process)
python -m observability.exporter --root observability/runs --port 9108

# individual stages
python -m observability.prepare --dataset Epilepsy-TSTCC --run-dir <run_dir>
python -m observability.evaluate --run-dir <run_dir> --dataset Epilepsy-TSTCC \
  --pretrain-desc <run_id>_pretrain --finetune-desc <run_id>_finetune \
  --expect-ft --final-ckpt checkpoint/Epilepsy-TSTCC/<run_id>_final_round.pt
```

`stage` is the Argo-facing wrapper:

```bash
python -m observability.stage --run-dir <run_dir> --phase pretrain -- \
  python -u FedCSL_All.py ... --description <run_id>_pretrain \
  --save-final-checkpoint --final-checkpoint-path <ckpt>
```

## Environment variables

| Variable | Used by | Meaning |
|---|---|---|
| `FEDCSL_RUN_DIR` | training hooks, stage | run directory (enables progress JSON) |
| `FEDCSL_PHASE` | training hooks, stage | `pretrain` / `finetune` |
| `FEDCSL_RUNS_DIR` | exporter | root directory (default `observability/runs`) |
| `FEDCSL_EXPORTER_PORT` | exporter | default `9108` |
| `FEDCSL_EXPORTER_MAX_AGE_HOURS` | exporter | default `168` |
| `CUDA_VISIBLE_DEVICES` | local pipeline | default `0` |

## Run directory contract

```
<run_dir>/
├── run.json
├── dataset.json
├── progress.json / progress.jsonl
├── progress_finetune.json
├── stages/<phase>.json
├── logs/<phase>.log
└── eval_summary.json
```

The exporter only reads these files; it never imports torch or the training code.
