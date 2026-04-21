# FedCSL Dashboard

一个独立的轻量 Web 控制台，用于：

1. **配置训练选项**：在线编辑 `config/*.yml`，查看实时训练日志。
2. **监控 GPU**：解析 `nvidia-smi`，展示每张卡的利用率、显存与计算进程。
3. **脚本启动**：扫描 `scripts/` 与根目录下可运行的 `.sh` / `.py`，展示脚本首部注释作为说明，并支持一键启动、查看日志、终止任务。

**完全独立**：不改动任何训练代码，只调用 `nvidia-smi` 与现有脚本。

---

## 安装

```bash
cd dashboard
pip install -r requirements.txt
```

## 启动

```bash
# 在项目根目录执行（推荐）
python -m dashboard.app --host 0.0.0.0 --port 8765

# 或
cd dashboard && python app.py --host 0.0.0.0 --port 8765
```

浏览器打开 <http://localhost:8765>

> 远程服务器用法：`python -m dashboard.app --host 0.0.0.0 --port 8765` 后，
> 在本地执行 `ssh -L 8765:localhost:8765 <user>@<server>`，然后访问 <http://localhost:8765>。

## 目录说明

```
dashboard/
├── app.py               # FastAPI 主程序
├── static/
│   └── index.html       # 单页前端（无需构建）
├── runs.json            # 运行时生成：记录启动过的任务
└── logs/                # 运行时生成：每个任务一个日志文件
```

## API 速览

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| GET | `/api/gpus` | nvidia-smi 状态（卡 / 利用率 / 显存 / 进程） |
| GET | `/api/scripts` | 扫描 `scripts/**/*.sh` 并提取首部注释作为说明 |
| GET | `/api/configs` | 列出 `config/*.yml` |
| GET | `/api/configs/{name}` | 读取某个 YAML 原文 |
| PUT | `/api/configs/{name}` | 保存 YAML（会做 yaml.safe_load 校验） |
| POST | `/api/runs` | 启动脚本：`{script, env:{...}}`，返回 run_id |
| GET | `/api/runs` | 列出历史与在跑任务 |
| GET | `/api/runs/{id}/log?tail=500` | 读取日志末尾 N 行 |
| POST | `/api/runs/{id}/stop` | 终止任务（含子进程组） |

## 安全

- **无鉴权**：默认监听本机 127.0.0.1；如需暴露公网请自行加反向代理 + 鉴权。
- 脚本启动路径强制限制在项目根下，禁止目录穿越。
