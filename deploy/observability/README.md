# FedCSL local observability

Prometheus + Grafana for the FedCSL training/observability pipeline.

## Quick start

```bash
# 1. Start the exporter on the host (reads observability/runs/)
cd <repo-root>
python -m observability.exporter --root observability/runs --port 9108

# 2. Start Prometheus + Grafana
cd deploy/observability
docker compose up -d

# 3. Open
#    Prometheus: http://localhost:9090
#    Grafana:    http://localhost:3000  (admin / fedcsl)
```

The exporter is a plain Python process on the host because it needs access to
the run directory and `nvidia-smi`. Prometheus reaches it through
`host.docker.internal:9108`.

## Images

Docker Hub is not directly reachable on this workstation, so the compose file
uses the DaoCloud mirror by default:

```
docker.m.daocloud.io/prom/prometheus:v2.53.0
docker.m.daocloud.io/grafana/grafana:10.4.5
```

If you pulled the images from another mirror, either tag them to the default
names or override per service:

```bash
PROMETHEUS_IMAGE=quay.io/prometheus/prometheus:v2.53.0 \
GRAFANA_IMAGE=grafana/grafana:10.4.5 \
docker compose up -d
```

## Layout

```
deploy/observability/
├── docker-compose.yml
├── prometheus/
│   ├── prometheus.yml       # local scrape config (host.docker.internal:9108)
│   ├── prometheus-k8s.yml   # in-cluster scrape config
│   └── alerts.yml
└── grafana/
    ├── provisioning/
    │   ├── datasources/prometheus.yml
    │   └── dashboards/dashboards.yml
    └── dashboards/fedcsl-overview.json
```

## Verify

```bash
curl -s localhost:9108/healthz
curl -s localhost:9108/metrics | grep fedcsl_
curl -s localhost:9090/api/v1/targets | jq '.data.activeTargets[] | {job:.labels.job,health}'
curl -s -u admin:fedcsl localhost:3000/api/health
```

Expected after a run:

- `fedcsl_exporter_up 1`
- `fedcsl_run_info{run_id="...",status="succeeded"} 1`
- `fedcsl_round_test_accuracy{phase="pretrain"} ...`
- `fedcsl_finetune_mean_accuracy{ratio="0.1000"} ...`
- `fedcsl_eval_passed{run_id="..."} 1`

## Verified on this workstation (2026-10-27)

- Prometheus targets: `fedcsl-exporter` UP, `prometheus` UP.
- Grafana 10.4.5: datasource health OK; `FedCSL Observability` dashboard
  provisioned with 16 panels; queries `fedcsl_run_info` successfully.
- The training run used for validation:
  `obs-20261027-202829-d518a7` (Epilepsy-TSTCC, 2 rounds, test acc 0.9826;
  fine-tune 10% = 0.9622, 100% = 0.9743).

Docker Hub was unreachable directly; images were pulled through the system
proxy (`127.0.0.1:7892`) and tagged to the compose defaults. If pulls stall
with `unexpected EOF`, enable the proxy before retrying or preload the images.

## Data flow

```
FedCSL_All.py / observability.prepare / observability.evaluate
        │  JSON files
        ▼
observability/runs/<run_id>/
        │  read-only
        ▼
observability.exporter :9108/metrics
        │  scrape every 5s
        ▼
Prometheus :9090  ──►  Grafana :3000
```

## Troubleshooting

| Symptom | Fix |
|---|---|
| Grafana shows "No data" | check Prometheus target `fedcsl-exporter` is UP; check the run is younger than `FEDCSL_EXPORTER_MAX_AGE_HOURS` (168h) |
| `host.docker.internal` unresolved | Docker Desktop: enable "host networking" or run exporter in a container attached to the same network |
| Port 9090/3000 occupied | change the host port in `docker-compose.yml` |
| Old runs pollute metrics | delete stale directories under `observability/runs/` or lower `--max-age-hours` |
