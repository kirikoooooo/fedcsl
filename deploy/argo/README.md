# FedCSL on Argo Workflows

Four-stage DAG: `prepare -> pretrain (GPU) -> finetune (GPU) -> evaluate`.

This directory contains the production-target manifests. They were authored and
YAML-validated on the workstation; a GPU Kubernetes cluster is required to run
them. The same stage commands can be executed locally with
`python -m observability.pipeline` (see `docs/observability_argo_prom_grafana_design.md`).

## Files

| File | Purpose |
|---|---|
| `pvc.yaml` | `fedcsl-work` PVC (datasets, checkpoints, run state) |
| `workflowtemplate.yaml` | Argo WorkflowTemplate with 4 steps |
| `exporter.yaml` | Exporter Deployment + Service (argo namespace) |
| `argo-controller-metrics-service.yaml` | expose Argo controller metrics on :9090 |
| `observability-stack.yaml` | Prometheus + Grafana in `observability` namespace |
| `entrypoint.sh` | links `/opt/fedcsl` code to `/data` PVC paths |
| `../observability/Dockerfile.training` | training/exporter image |

## Prerequisites

- Kubernetes 1.24+ with an NVIDIA device plugin and GPU nodes.
- Argo Workflows installed in the `argo` namespace (v3.6.x tested syntax).
- An artifact repository (MinIO/S3) if you keep `archiveLogs: true`.
- A RWX storage class for the PVC (or a single-node cluster where RWO suffices).
- The image available on every node (push to a registry or preload it).

## 1. Build the image

```bash
cd <repo-root>
docker build -f deploy/observability/Dockerfile.training -t fedcsl-train:observe .
# push to your registry and update workflowtemplate.yaml image param
```

The image contains code only. Datasets, `algoutils/shapelet_weight_All.npy`,
checkpoints and results live on the PVC.

## 2. Prepare the PVC

```bash
kubectl apply -f deploy/argo/pvc.yaml

# copy the required assets into /data (example with a helper pod):
kubectl -n argo run fedcsl-pvc-helper --rm -it --image=busybox \
  --overrides='{"spec":{"volumes":[{"name":"w","persistentVolumeClaim":{"claimName":"fedcsl-work"}}],"containers":[{"name":"c","image":"busybox","command":["sh"],"stdin":true,"tty":true,"volumeMounts":[{"name":"w","mountPath":"/data"}]}]}}'
# inside: mkdir -p /data/Epilepsy /data/algoutils /data/observability/runs
# copy data with a pod, kubectl cp, or your storage provider's tooling.
```

Expected layout:

```
/data/
├── Epilepsy/{train,val,test}.pt              # or Multivariate_ts/LSST/...
├── Multivariate_ts/LSST/LSST_{TRAIN,TEST}.ts # if dataset=LSST
├── UCI HAR Dataset/...                       # only if prepare must build HAR
├── algoutils/shapelet_weight_All.npy
├── checkpoint/
├── result/
├── data/
└── observability/runs/
```

## 3. Install Argo Workflows

```bash
kubectl create namespace argo
kubectl apply -n argo -f https://github.com/argoproj/argo-workflows/releases/download/v3.6.5/install.yaml
```

For `archiveLogs: true`, configure `workflow-controller-configmap` with an
artifactRepository (MinIO/S3). Without one, logs remain on pods only.

## 4. Deploy exporter and observability stack

```bash
kubectl apply -f deploy/argo/exporter.yaml
kubectl apply -f deploy/argo/argo-controller-metrics-service.yaml

# ConfigMaps consumed by observability-stack.yaml
kubectl apply -f deploy/argo/observability-stack.yaml   # creates namespace observability
kubectl -n observability create configmap fedcsl-prometheus-config \
  --from-file=prometheus.yml=deploy/observability/prometheus/prometheus-k8s.yml
kubectl -n observability create configmap fedcsl-prometheus-alerts \
  --from-file=alerts.yml=deploy/observability/prometheus/alerts.yml
kubectl -n observability create configmap fedcsl-grafana-datasource \
  --from-file=prometheus.yml=deploy/observability/grafana/provisioning/datasources/prometheus.yml
kubectl -n observability create configmap fedcsl-grafana-dashboard-provider \
  --from-file=dashboards.yml=deploy/observability/grafana/provisioning/dashboards/dashboards.yml
kubectl -n observability create configmap fedcsl-grafana-dashboard \
  --from-file=fedcsl-overview.json=deploy/observability/grafana/dashboards/fedcsl-overview.json

# restart pods once the ConfigMaps exist
kubectl -n observability rollout restart deploy/fedcsl-prometheus deploy/fedcsl-grafana
```

Access:

```bash
kubectl -n argo port-forward svc/argo-server 2746:2746       # Argo UI
kubectl -n observability port-forward svc/prometheus 9090:9090
kubectl -n observability port-forward svc/fedcsl-grafana 3000:3000
```

## 5. Submit

```bash
kubectl -n argo create -f deploy/argo/workflowtemplate.yaml

argo submit -n argo --from workflowtemplate/fedcsl-pipeline \
  -p dataset=Epilepsy-TSTCC \
  -p numRounds=2 \
  -p numClients=3 \
  -p ftRatios=0.1,1.0

argo watch -n argo @latest
```

Artifacts land on the PVC at:

```
/data/observability/runs/<workflow.uid>/{run.json,dataset.json,progress*.json,stages/*.json,logs/*.log,eval_summary.json}
/data/result/<dataset>/*<workflow.uid>_pretrain*.txt
/data/result/<dataset>/*<workflow.uid>_finetune*.txt
/data/checkpoint/<workflow.uid>/final_round.pt
```

The exporter reads `/data/observability/runs`; Prometheus scrapes it every 5s.

## Notes

- **GPU**: `nvidia.com/gpu: 1` is requested on pretrain/finetune. Do not set
  `CUDA_VISIBLE_DEVICES`; the device plugin handles it. `--server-gpu 0`
  and `--client-gpus 0` are logical IDs inside the pod.
- **Config**: `config/configACF.yml` is baked into the image. For experiments
  with different configs, mount a ConfigMap over
  `/opt/fedcsl/config/configACF.yml` or bake a new image.
- **Micro-batching / OOM**: lower `--num-client` or use a YAML with
  `numEpoch: 1`; check `fedcsl_gpu_memory_used_bytes`.
- **Security**: Argo UI and Grafana are unauthenticated/weakly authenticated in
  this demo. Do not expose them publicly; use an ingress with SSO.
- **Exporter label cardinality**: run ids are unique per workflow. Tune
  `FEDCSL_EXPORTER_MAX_AGE_HOURS` before long-term retention.
