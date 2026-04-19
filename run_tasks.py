"""
批量调度任务脚本：

- 从 tasks.json 读取任务列表（由 generate_tasks.py 生成）
- 检测空闲 GPU（无进程占用）后，把 pending 任务一对一分配上去
- 用 subprocess.Popen 直接拉起 python 子进程（不再依赖 shell 重定向）
- 通过 process.poll() 精确判断完成 / 失败，并记录 returncode

参考服务器最后几条手写命令的形式：

  CUDA_VISIBLE_DEVICES=2 nohup python -u FedCSL_All.py -dataset PEMS-SF \
    --config configACF.yml --use-client-selection \
    --client-selection-method uniform --client-selection-ratio 0.7 \
    --min-selection-prob 0.01 --ema-alpha 0 --description "uniform+acf" &
"""

from __future__ import annotations

import os
import json
import time
import shlex
import signal
import subprocess
import datetime
from typing import Dict, List


TASKS_FILE = "tasks.json"
LOG_DIR = "task_logs"
POLL_INTERVAL = 30  # 秒


# 内存中保存子进程对象（不写入 json）
_running_procs: Dict[int, subprocess.Popen] = {}


def check_available_gpus():
    """返回当前没有 compute 进程占用的 GPU 索引列表。"""
    try:
        gpu_list_out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader,nounits"],
            stderr=subprocess.STDOUT,
        ).decode("utf-8").strip()
    except Exception as e:
        print(f"[GPU] 检查失败：{e}")
        return []

    available = []
    for line in gpu_list_out.splitlines():
        line = line.strip()
        if not line:
            continue
        gpu_id = int(line)
        try:
            apps = subprocess.check_output(
                [
                    "nvidia-smi",
                    "--query-compute-apps=pid",
                    f"--id={gpu_id}",
                    "--format=csv,noheader",
                ],
                stderr=subprocess.STDOUT,
            ).decode("utf-8").strip()
        except Exception as e:
            print(f"[GPU{gpu_id}] 查询占用失败：{e}")
            continue
        if not apps:
            available.append(gpu_id)
    return available


def load_tasks():
    if not os.path.exists(TASKS_FILE):
        print(f"未找到 {TASKS_FILE}，请先运行 generate_tasks.py")
        return []
    try:
        with open(TASKS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"读取 {TASKS_FILE} 失败：{e}")
        return []


def save_tasks(tasks):
    try:
        with open(TASKS_FILE, "w", encoding="utf-8") as f:
            json.dump(tasks, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"保存 {TASKS_FILE} 失败：{e}")


def ensure_log_dir():
    os.makedirs(LOG_DIR, exist_ok=True)
    return LOG_DIR


def build_command(task: dict) -> list[str]:
    """
    构建 python 子进程的 argv（直接传 list，不走 shell，避免引号注入与孤儿进程）。
    与服务器手写命令的参数顺序、键名保持一致。
    """
    dataset = task["dataset"]
    alpha = task["dirichlet_alpha"]
    config = task.get("config", "configACF.yml")
    method = task.get("client_selection_method", "omp")
    ratio = task.get("client_selection_ratio", 0.7)
    min_prob = task.get("min_selection_prob", 0.01)
    ema_alpha = task.get("ema_alpha", 0.0)
    description = task.get("description", f"{method}+acf")

    cmd = [
        "python", "-u", "FedCSL_All.py",
        "-dataset", str(dataset),
        "--config", str(config),
        "--dirichlet-alpha", str(alpha),
        "--use-client-selection",
        "--client-selection-method", str(method),
        "--client-selection-ratio", str(ratio),
        "--min-selection-prob", str(min_prob),
        "--ema-alpha", str(ema_alpha),
        "--description", str(description),
    ]

    # batch_size 仅在显式给定时传入，避免与 yaml 配置默认值冲突
    if task.get("batch_size") is not None:
        cmd += ["--batch-size", str(task["batch_size"])]

    return cmd


def run_task(task: dict, gpu_id: int) -> dict:
    log_dir = ensure_log_dir()
    method = task.get("client_selection_method", "omp")
    log_file = os.path.join(
        log_dir,
        f"task_{task['id']}_{task['dataset']}_alpha_{task['dirichlet_alpha']}_{method}.log",
    )

    cmd = build_command(task)
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    pretty_cmd = "CUDA_VISIBLE_DEVICES={} {}".format(
        gpu_id, " ".join(shlex.quote(c) for c in cmd)
    )
    print(f"[GPU{gpu_id}] 启动任务 {task['id']}: {pretty_cmd}")

    try:
        log_fp = open(log_file, "a", buffering=1, encoding="utf-8")
        log_fp.write(f"# 启动时间: {datetime.datetime.now():%Y-%m-%d %H:%M:%S}\n")
        log_fp.write(f"# GPU: {gpu_id}\n")
        log_fp.write(f"# CMD: {pretty_cmd}\n\n")
        log_fp.flush()

        process = subprocess.Popen(
            cmd,
            env=env,
            stdout=log_fp,
            stderr=subprocess.STDOUT,
            start_new_session=True,  # 独立进程组，便于 kill 整组
        )
    except Exception as e:
        task["status"] = "failed"
        task["error"] = str(e)
        print(f"[GPU{gpu_id}] 启动失败：{e}")
        return task

    _running_procs[task["id"]] = process

    task["status"] = "running"
    task["gpu"] = gpu_id
    task["pid"] = process.pid
    task["start_time"] = time.time()
    task["log_file"] = log_file
    task.pop("error", None)
    task.pop("returncode", None)
    return task


def update_running_tasks(tasks: list[dict]) -> bool:
    """
    扫描 running 任务，根据 Popen.poll() 更新状态。
    返回值：是否有任何任务在本次循环中被刷新。
    """
    changed = False
    for task in tasks:
        if task.get("status") != "running":
            continue
        proc = _running_procs.get(task["id"])
        if proc is None:
            # 脚本被中途重启时，内存里没有 Popen 句柄。
            # 尝试用 pid 兜底判断；判不准的就当仍在跑，避免误判。
            if task.get("pid") is not None:
                try:
                    os.kill(task["pid"], 0)
                except OSError:
                    task["status"] = "completed"  # 仅做兜底
                    task["end_time"] = time.time()
                    task["completion"] = 100
                    changed = True
            continue

        rc = proc.poll()
        if rc is None:
            continue  # 仍在跑

        task["end_time"] = time.time()
        task["returncode"] = rc
        if rc == 0:
            task["status"] = "completed"
            task["completion"] = 100
            print(f"[done] 任务 {task['id']} 完成 (rc=0)")
        else:
            task["status"] = "failed"
            print(f"[fail] 任务 {task['id']} 失败 (rc={rc}), 见 {task.get('log_file')}")
        if task.get("log_file") and os.path.exists(task["log_file"]):
            with open(task["log_file"], "a", encoding="utf-8") as f:
                f.write(
                    f"\n# 结束时间: {datetime.datetime.now():%Y-%m-%d %H:%M:%S}, returncode={rc}\n"
                )
        _running_procs.pop(task["id"], None)
        changed = True
    return changed


def schedule_pending(tasks: list[dict]) -> bool:
    """把 pending 任务分配到空闲 GPU。每张卡只跑一个任务。返回是否分配过。"""
    available = check_available_gpus()
    if not available:
        return False

    occupied_gpus = {
        t["gpu"] for t in tasks if t.get("status") == "running" and t.get("gpu") is not None
    }
    free_gpus = [g for g in available if g not in occupied_gpus]
    if not free_gpus:
        return False

    pending = [t for t in tasks if t.get("status") == "pending"]
    if not pending:
        return False

    print(f"[schedule] 空闲 GPU: {free_gpus}, 待跑任务: {len(pending)}")
    scheduled = False
    for gpu_id in free_gpus:
        if not pending:
            break
        task = pending.pop(0)
        run_task(task, gpu_id)
        scheduled = True
    return scheduled


def all_finished(tasks: list[dict]) -> bool:
    return all(t.get("status") in ("completed", "failed") for t in tasks)


def generate_report(tasks: list[dict]) -> None:
    report_file = "task_report.txt"
    with open(report_file, "w", encoding="utf-8") as f:
        f.write("任务执行报告\n")
        f.write("=" * 50 + "\n")
        f.write(f"生成时间: {datetime.datetime.now():%Y-%m-%d %H:%M:%S}\n")
        f.write(f"总任务数: {len(tasks)}\n")
        completed = [t for t in tasks if t.get("status") == "completed"]
        failed = [t for t in tasks if t.get("status") == "failed"]
        f.write(f"成功: {len(completed)}, 失败: {len(failed)}\n\n")
        f.write("-" * 50 + "\n")
        for task in tasks:
            f.write(f"任务ID: {task['id']}\n")
            f.write(f"数据集: {task.get('dataset')}\n")
            f.write(f"alpha: {task.get('dirichlet_alpha')}\n")
            f.write(f"method: {task.get('client_selection_method')}\n")
            f.write(f"status: {task.get('status')}\n")
            if task.get("returncode") is not None:
                f.write(f"returncode: {task['returncode']}\n")
            if task.get("start_time"):
                f.write(
                    f"开始: {datetime.datetime.fromtimestamp(task['start_time']):%Y-%m-%d %H:%M:%S}\n"
                )
            if task.get("end_time") and task.get("start_time"):
                dur = task["end_time"] - task["start_time"]
                f.write(f"耗时: {dur:.1f} s\n")
            if task.get("log_file"):
                f.write(f"日志: {task['log_file']}\n")
            f.write("-" * 50 + "\n")
    print(f"报告已写入 {report_file}")


def shutdown(_signum, _frame):
    print("\n[signal] 收到终止信号，杀掉所有子进程...")
    for tid, proc in list(_running_procs.items()):
        if proc.poll() is None:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            except Exception as e:
                print(f"  - 任务 {tid} kill 失败：{e}")
    raise SystemExit(130)


def main():
    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    tasks = load_tasks()
    if not tasks:
        print("没有任务可执行")
        return

    running_at_start = [t for t in tasks if t.get("status") == "running"]
    if running_at_start:
        print(
            f"提示：tasks.json 中有 {len(running_at_start)} 个 running 状态的旧任务，"
            "脚本本次启动无法接管这些进程，会按 pid 兜底轮询。"
        )

    while True:
        update_running_tasks(tasks)
        schedule_pending(tasks)
        save_tasks(tasks)

        if all_finished(tasks):
            print("[done] 全部任务结束")
            generate_report(tasks)
            return

        time.sleep(POLL_INTERVAL)


if __name__ == "__main__":
    main()
