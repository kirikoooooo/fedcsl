# FedCSL 可观测性服务访问信息

> 仅用于本机开发环境。Grafana 的 `fedcsl` 是演示用默认弱口令；共享、局域网或公网访问前必须修改。
> 验证日期：2026-10-27；分支：`feat/observability-argo-prom-grafana`。

## 当前运行状态

| 服务 | 地址 | 凭据 | 状态 |
|---|---|---|---|
| Grafana | http://localhost:3000 | 用户名 `admin` / 密码 `fedcsl` | 运行中 |
| Grafana Dashboard | http://localhost:3000/d/fedcsl-overview | 同上 | 已自动加载 |
| Prometheus | http://localhost:9090 | 无认证 | 运行中 |
| Prometheus Targets | http://localhost:9090/targets | 无 | `fedcsl-exporter` / `prometheus` 均 UP |
| Prometheus Alerts | http://localhost:9090/alerts | 无 | 6 条告警规则 |
| FedCSL Exporter | http://localhost:9108/metrics | 无认证 | 运行中（宿主机进程） |
| Exporter API | http://localhost:9108/api/runs | 无 | 返回 run 状态 JSON |
| Exporter 健康检查 | http://localhost:9108/healthz | 无 | 返回 `ok` |
| Jenkins（既有服务） | http://localhost:8080 | 既有 Jenkins 账号 | 运行中 |
| Argo UI | https://localhost:2746（需 port-forward） | 未部署 | 本机无 K8s 集群 |

同一 WLAN 下可尝试（需 Windows 防火墙放行）：

```
http://192.168.1.10:3000    # Grafana
http://192.168.1.10:9090    # Prometheus
http://192.168.1.10:9108/metrics
```

## 快速验证

```bash
curl -s -u admin:fedcsl http://127.0.0.1:3000/api/health
curl -s -u admin:fedcsl http://127.0.0.1:3000/api/datasources/uid/prometheus/health
curl -s http://127.0.0.1:9090/-/ready
curl -s http://127.0.0.1:9090/api/v1/targets
curl -s http://127.0.0.1:9108/metrics | grep '^fedcsl_'
```

当前验证结果：

- Grafana `10.4.5`，`/api/health` → `database: ok`
- Grafana 数据源健康检查 → `"Successfully queried the Prometheus API."`
- Grafana dashboard `FedCSL Observability` → 16 个 panel
- Prometheus target → `fedcsl-exporter` UP、`prometheus` UP
- 最新 run：`obs-20261027-202829-d518a7`，status `succeeded`，best test acc `0.9826`

## 修改 Grafana 密码

方式一：通过环境变量重启容器（推荐，compose 会持久化）：

```bash
cd deploy/observability
GRAFANA_ADMIN_PASSWORD='替换成强密码' docker compose up -d grafana
```

方式二：Grafana UI → 左下角头像 → Profile → Change password。

修改后请同步更新本文件（或把本文件从版本库排除，改为本地保存）。

## Exporter 重启

Exporter 是宿主机 Python 进程，不随 Docker 自动启动：

```bash
cd <repo-root>
E:/code_tools/miniconda3/envs/CSL/python.exe -m observability.exporter \
  --root observability/runs --port 9108
```

## Argo UI（部署到 GPU 集群后）

```bash
kubectl -n argo port-forward svc/argo-server 2746:2746
# 浏览器打开 https://localhost:2746（自签名证书）
```

Argo Server 的认证方式取决于安装配置；默认 `server` 模式可能需要 service account token：

```bash
kubectl -n argo create token argo-server
```

本机目前没有可用 Kubernetes 集群，因此 Argo UI 未运行。

## 数据与生命周期

- Prometheus 数据：命名卷 `fedcsl-observability_prometheus-data`（默认保留 15 天）
- Grafana 数据：命名卷 `fedcsl-observability_grafana-data`
- Run 产物：`observability/runs/<run_id>/`
- `docker compose down` 保留数据；`docker compose down -v` 会删除卷
- 服务端口绑定在 `0.0.0.0`，请勿直接暴露到公网；Grafana/Prometheus/Exporter 均无 TLS 和强认证

## 相关文档

- 架构与设计：`docs/observability_argo_prom_grafana_design.md`
- 本地部署：`deploy/observability/README.md`
- Argo 集群部署：`deploy/argo/README.md`
- 运行时代码：`observability/README.md`
