# Jenkins CICD 部署 FedCSL Dashboard —— 实施过程记录

> 日期：2026-10-24
> 目标：用 Jenkins 定时检测 GitHub 仓库更新，有更新时自动部署 dashboard（FastAPI 后端 + 静态前端），全程零手工干预。
> 状态：✅ **已上线并完成端到端验证**（构建 #2 实际部署成功；构建 #3 无更新正确跳过）

---

## 1. 架构总览

```
GitHub: kirikoooooo/fedcsl（feat/muti-routine 分支，dashboard/ 所在分支）
        │  git ls-remote（Jenkins 容器，cron H/15 * * * *）
        ▼
┌─ Jenkins LTS 2.580.1（Docker 容器，127.0.0.1:8080，62 个插件）────────┐
│  流水线 "dashboard-deploy"（init.groovy.d 创建，沙箱模式）             │
│   1. 取 feat/muti-routine 最新 commit，与 /state/last_deployed.txt 比对│
│   2. 有变化 → 写 /deploy/request.json                                  │
│   3. 轮询 /deploy/result.json（≤10 分钟）                              │
│   4. 成功 → 更新状态文件；失败 → 构建标红                              │
└───────────────────────────────────────────────────────────────────────┘
        │ 文件交换（bind mount: C:\jenkins\deploy、C:\jenkins\state）
        ▼
本机 watcher（Windows 计划任务 FedCSLDeployWatcher，每分钟，pythonw 无窗口）
   1. 读 request.json → git fetch origin feat/muti-routine（绕过失效代理）
   2. 仅当 dashboard/ 有差异：git checkout FETCH_HEAD -- dashboard/
   3. 按端口 8765 精确重启 dashboard 进程（不动训练等其他进程）
   4. 健康检查 → 写 result.json
```

**关键设计决策**

| 决策 | 理由 |
|---|---|
| Jenkins 跑在 Docker 容器 | 本机无 Java、shell 无管理员权限；官方 LTS 镜像自带 JRE，`--restart unless-stopped` 保证开机自启 |
| 用 cron 触发而非 pollSCM | 任务定义不依赖仓库里的 Jenkinsfile（避免鸡生蛋）；pollSCM 要求执行脚本来自 SCM |
| 任务由 init.groovy.d 创建，不用 JCasC jobs | JCasC 的 `jobs:` 根元素要求 Job DSL 插件且脚本为 Job DSL 语法；原生 API 不依赖额外插件 |
| 部署执行端放本机 watcher | dashboard 必须留在 Windows 本机（要读 nvidia-smi、启动训练脚本），容器内无法管理宿主进程 |
| 只更新 dashboard/ 子目录 | 工作区有大量未提交研究代码（FedCSL_All.py 等），`git checkout FETCH_HEAD -- dashboard/` 精准更新，互不干扰 |
| 部署源分支 feat/muti-routine | dashboard/ 只存在于该分支，main 上还没有（实测确认） |
| 每次 git 操作绕过代理 | git 全局配置了 `http.proxy=127.0.0.1:7892`，该代理未运行；用 `-c http.proxy=` 直连 |
| 计划任务用 pythonw.exe | python.exe 每分钟弹一次控制台窗口；pythonw 无窗口（实测已解决） |

---

## 2. 环境准备

- Docker Desktop 27.0.3
- Python：`E:\code_tools\miniconda3\envs\CSL\python.exe`（dashboard 依赖 fastapi/uvicorn/psutil 已装）
- Jenkins 镜像：`jenkins/jenkins:lts` → 实际版本 2.580.1

### 2.1 坑：Docker Hub 直连超时

国内直连 `registry-1.docker.io` 超时。**用 DaoCloud 镜像源拉取后 retag**：

```bash
docker pull docker.m.daocloud.io/jenkins/jenkins:lts
docker tag docker.m.daocloud.io/jenkins/jenkins:lts jenkins/jenkins:lts
```

> 注意：`docker pull ... | tail` 会掩盖失败（管道退出码是 tail 的 0），必须 `docker images` 确认镜像真实存在。

### 2.2 坑：git 全局代理已失效

本机 git 配置 `http.proxy=http://127.0.0.1:7892`，但代理未运行。脚本发起的 GitHub 访问必须绕过：

```bash
git -c http.proxy= -c https.proxy= fetch origin feat/muti-routine
```

---

## 3. Jenkins 安装

### 3.1 目录结构

```
C:\jenkins\
├── home\                  # jenkins_home（容器 /var/jenkins_home）
│   ├── init.groovy.d\     # 启动脚本：创建/更新 dashboard-deploy 任务
│   └── plugins\           # 62 个插件
├── casc\
│   ├── jenkins.yaml       # JCasC：admin 用户 + 安全策略（不含 jobs）
│   ├── secrets.env        # ADMIN_PASSWORD=<随机生成>
│   ├── uc-orig.json       # 官方 update-center.json 原始副本
│   └── uc-mirror.json     # 下载地址重写为清华镜像的版本
├── deploy\                # Jenkins ↔ watcher 文件交换区
├── state\                 # last_deployed.txt / dashboard.pid / watcher.log
└── deploy_watcher.py      # 本机部署执行端
```

### 3.2 启动容器

```bash
MSYS_NO_PATHCONV=1 docker run -d --name jenkins \
  -p 127.0.0.1:8080:8080 -p 127.0.0.1:50000:50000 \
  -v C:/jenkins/home:/var/jenkins_home \
  -v C:/jenkins/casc:/var/casc \
  -v C:/jenkins/deploy:/deploy \
  -v C:/jenkins/state:/state \
  -e CASC_JENKINS_CONFIG=/var/casc/jenkins.yaml \
  -e "JAVA_OPTS=-Djenkins.install.runSetupWizard=false" \
  --env-file C:/jenkins/casc/secrets.env \
  --restart unless-stopped \
  jenkins/jenkins:lts
```

**安全设置**（应权限审查要求）：
- 端口只绑定 `127.0.0.1`，不暴露局域网
- admin 密码随机生成（`secrets.token_urlsafe(16)`），存于 `secrets.env`，不硬编码在命令/脚本中
- JCasC 关闭注册（`allowsSignup: false`）、关闭匿名读

> ⚠️ **Git Bash 路径转换坑**：MSYS 会把参数中的 `/var/casc/...` 转成 Windows 路径，
> 导致 `CASC_JENKINS_CONFIG` 变成 `E:/user_tools/Git/var/casc/jenkins.yaml` 而启动失败。
> 必须加 `MSYS_NO_PATHCONV=1`（`docker cp` 的目标路径同理）。

### 3.3 插件安装（国内网络方案）

Jenkins 官方更新中心（`updates.jenkins.io` / `archives.jenkins.io`）国内基本不可用。实际方案：

1. 从官方下载 update-center.json（约 3.4MB，33s，可通）：
   ```bash
   curl -sL -o C:/jenkins/casc/uc-orig.json https://updates.jenkins.io/update-center.json
   ```
2. 把下载地址重写为清华镜像（`.hpi` 文件在 TUNA 上完整可用，628KB/0.67s）：
   ```python
   data.replace('https://updates.jenkins.io/download/plugins/',
                'https://mirrors.tuna.tsinghua.edu.cn/jenkins/plugins/')
   ```
3. 拷进容器，用 `file://` + `--jenkins-update-center-download-url` 安装：
   ```bash
   MSYS_NO_PATHCONV=1 docker cp C:/jenkins/casc/uc-mirror.json jenkins:/tmp/uc-mirror.json
   MSYS_NO_PATHCONV=1 docker exec jenkins jenkins-plugin-cli \
     --jenkins-update-center file:///tmp/uc-mirror.json \
     --jenkins-update-center-download-url https://mirrors.tuna.tsinghua.edu.cn/jenkins/plugins/ \
     --plugin-download-directory /var/jenkins_home/plugins \
     --plugins "configuration-as-code workflow-aggregator timestamper pipeline-utility-steps"
   ```

**踩过的两个坑**：
- `jenkins-plugin-cli` 默认装到 `/usr/share/jenkins/ref/plugins`（镜像内置目录），运行时安装必须显式 `--plugin-download-directory /var/jenkins_home/plugins`
- 仅设 `JENKINS_UC` 指向清华的 `update-center.json` 无效（TUNA 无此文件，404），且即便拿到 JSON，依赖解析仍会从 `archives.jenkins.io` 拉取 → 必须用 `--jenkins-update-center-download-url` 覆盖下载基址

结果：**62 个插件**（含 configuration-as-code、workflow-aggregator、pipeline-utility-steps、timestamper 及其依赖）。

---

## 4. 配置方式：JCasC + init.groovy.d

### 4.1 JCasC（`C:\jenkins\casc\jenkins.yaml`）

只负责账号与安全策略：

```yaml
jenkins:
  systemMessage: "FedCSL Dashboard CICD ..."
  numExecutors: 2
  quietPeriod: 3
  securityRealm:
    local:
      allowsSignup: false
      users:
        - id: admin
          password: ${ADMIN_PASSWORD}      # 从 secrets.env 注入
  authorizationStrategy:
    loggedInUsersCanDoAnything:
      allowAnonymousRead: false
  views:
    - all:
        name: "all"

unclassified:        # 注意：顶层键！
  location:
    url: http://localhost:8080/
```

**踩过的两个坑**：
- `unclassified` 必须是**顶层键**，放在 `jenkins:` 下会报 `Invalid configuration elements for type: class jenkins.model.Jenkins : unclassified`，Jenkins 直接启动失败
- JCasC 的 `jobs:` 根元素需要 Job DSL 插件，且其 `script` 是 Job DSL 语法而非 pipeline 语法 → 放弃 JCasC 建任务

### 4.2 init.groovy.d（`C:\jenkins\home\init.groovy.d\01-create-pipeline-job.groovy`）

启动时用 Jenkins 原生 API 创建/更新 `dashboard-deploy` 任务（幂等），并注册 `TimerTrigger('H/15 * * * *')`。**沙箱模式（sandbox=true）**，只调用白名单步骤：

| 步骤 | 来源插件 |
|---|---|
| `sh` / `fileExists` / `readFile` / `writeFile` / `sleep` / `error` | workflow-basic-steps |
| `readJSON` / `writeJSON` | pipeline-utility-steps（单独安装） |

> 踩坑：初版用 `sandbox=false` + `new File(...)`，构建报 `UnapprovedUsageException: script not yet approved for use`。改用沙箱 + utility-steps 后正常。

---

## 5. 流水线逻辑

| 阶段 | 内容 |
|---|---|
| DetectAndDeploy | `git ls-remote` 取 feat/muti-routine 最新 commit → 与 `/state/last_deployed.txt` 比对 |
| 无变化 | `currentBuild.result = 'NOT_BUILT'`，不写请求文件（构建灰色，不算失败） |
| 有变化 | `writeJSON /deploy/request.json`（request_id=BUILD_TAG）→ 每 10s 轮询 `/deploy/result.json`，超时 10 分钟 |
| 结果处理 | `deployed`/`no_changes` → 更新状态文件；`failed` → `error()` 构建标红 |

构建保留：`logRotator(numToKeepStr: '200')`。

---

## 6. 本机 watcher 与计划任务

### 6.1 脚本：`C:\jenkins\deploy_watcher.py`

短生命周期（每分钟被计划任务拉起，无请求立刻退出）。核心逻辑：

```python
fetch = _git(["fetch", "origin", "feat/muti-routine"])          # 绕过失效代理
diff  = _git(["diff", "--quiet", "FETCH_HEAD", "--", "dashboard/"])
if diff.returncode == 0:  return "no_changes"                    # 无变化不重启
co = _git(["checkout", "FETCH_HEAD", "--", "dashboard/"])        # 只更新 dashboard/
_stop_dashboard()        # psutil 按端口 8765 精确定位 pid，terminate + 等端口释放
_start_dashboard()       # 新进程（CREATE_NEW_PROCESS_GROUP，pid 写 state/dashboard.pid）
_wait_healthy(60)        # 轮询 http://127.0.0.1:8765/ 直到 200
```

安全边界：只 kill 监听 8765 的进程；只 checkout `dashboard/` 路径；失败时回写 `failed` 不掩盖。

### 6.2 计划任务

```bash
schtasks /create /tn "FedCSLDeployWatcher" \
  /tr "\"E:\code_tools\miniconda3\envs\CSL\pythonw.exe\" \"C:\jenkins\deploy_watcher.py\"" \
  /sc minute /mo 1 /f
```

用户级任务，无需管理员。**必须用 `pythonw.exe`**（用 `python.exe` 会每分钟闪一次控制台窗口）。

---

## 7. 端到端验证记录

| 测试 | 结果 | 说明 |
|---|---|---|
| watcher 手工测试 #1 | ❌ | `dashboard/` 不在 main 分支 → 部署源分支改为 feat/muti-routine |
| watcher 手工测试 #2 | ✅ | 杀旧进程(30044)→起新进程→健康检查 200，commit `fcf6bd2` |
| Jenkins 构建 #1 | ❌ | `UnapprovedUsageException`（sandbox=false）→ 改沙箱 + utility-steps |
| **Jenkins 构建 #2** | ✅ **SUCCESS** | 完整链路：ls-remote → request.json → watcher 部署（pid 27736）→ result.json → SUCCESS |
| **Jenkins 构建 #3** | ✅ **NOT_BUILT** | 无新提交，正确跳过，不重复部署 |

构建 #2 实际日志：

```
[2026-10-24T15:46:13] Sleeping for 10 sec
[2026-10-24T15:46:26] deploy result: [status:deployed, message:dashboard/ 已更新到 fcf6bd2b7ceb 并重启成功 (pid=27736)]
Finished: SUCCESS
```

watcher 侧：

```
[2026-10-24 23:46:01] 开始部署 request_id=jenkins-dashboard-deploy-2
[2026-10-24 23:46:10] 新 dashboard 进程 pid=27736, 停止旧进程: killed pid=30044
[2026-10-24 23:46:15] 部署结束: status=deployed
```

---

## 8. 使用说明

| 项 | 值 |
|---|---|
| **Jenkins 地址** | **http://localhost:8080**（仅本机） |
| 账号 | `admin` |
| 密码 | `C:\jenkins\casc\secrets.env` 中 `ADMIN_PASSWORD=` 后的值 |
| 流水线任务 | `dashboard-deploy`（每 15 分钟自动跑；也可手动 Build Now） |
| dashboard 地址 | http://localhost:8765 |
| watcher 日志 | `C:\jenkins\state\watcher.log` |
| dashboard 日志 | `C:\jenkins\state\dashboard.log` |
| 已部署 commit 记录 | `C:\jenkins\state\last_deployed.txt` |

**日常运维**

```bash
docker restart jenkins                      # 重启 Jenkins（数据在 jenkins_home 卷中，不丢）
docker logs --tail 100 jenkins              # 查看 Jenkins 日志
schtasks /change /tn FedCSLDeployWatcher /disable   # 暂停自动部署
schtasks /change /tn FedCSLDeployWatcher /enable    # 恢复自动部署
```

**发布更新**：push 到 `feat/muti-routine` 分支且改动包含 `dashboard/` 目录 → 最多 15 分钟（Jenkins 轮询）+ 1 分钟（watcher 周期）后自动部署。

---

## 9. 已知限制与后续可选优化

- **延迟**：最大约 16 分钟（轮询间隔 15 分钟 + watcher 1 分钟）。如需秒级，可加 GitHub webhook + 内网穿透。
- **只部署 dashboard/**：其他代码改动不会触发部署（刻意设计，保护工作区研究代码）。
- **dashboard/requirements.txt 缺失**：README 引用了但仓库中没有；watcher 目前不做依赖安装，换机器时需手动 `pip install fastapi uvicorn psutil`。
- **JCasC 每次启动强制覆盖**：`jenkins.yaml` 会覆盖 UI 上对账号/安全策略的修改；任务定义由 init.groovy.d 覆盖。改配置请改文件而非 UI。
- **仅本机可访问**：如需局域网访问，把 `-p 127.0.0.1:8080:8080` 改为 `-p 8080:8080`，并务必先改强密码。
