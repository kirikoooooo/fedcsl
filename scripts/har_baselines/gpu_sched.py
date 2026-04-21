#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""基于 ``nvidia-smi`` 的简易 GPU 调度器，供 har_baselines shell 脚本调用。

提供两条子命令（均以 reserve-file 作为本会话的预留账本）:

  wait     阻塞轮询，直到满足调度条件后挑一张 GPU 返回
           (对应 "监听到剩余>2个卡时挂上一个训练" 的语义)。
           -- 条件: free_gpus(排除 exclude / reserved) 数量 >= min-free。
           -- 返回: 选中的 GPU id 写到 stdout（一行），并追加到 reserve-file。

  release  释放指定的 GPU 占用（从 reserve-file 中删除）。

设计约束:
  * "空闲" = ``nvidia-smi --query-compute-apps`` 返回空（与 run_tasks.py 保持一致）。
  * ``--exclude`` 永远排除（典型用法: ``--exclude 2`` 禁用 2 号卡）。
  * ``--reserve-file`` 存放 "本调度器会话" 刚刚挑过的 GPU id，避免
    进程起步阶段 nvidia-smi 还未体现占用导致重复派发。
  * ``--min-free`` 是 "启动前必须的空闲数" 门槛:
      - min-free=2  → 剩余>=2 时才挂任务；启动 1 张后仍保留 ≥1 张空闲（默认）。
      - min-free=3  → 剩余>=3 时才挂任务；启动 1 张后仍保留 ≥2 张空闲。
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from typing import Iterable, List, Set


def _nvidia_smi_ok() -> bool:
    return subprocess.run(
        ["nvidia-smi", "-L"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    ).returncode == 0


def list_all_gpus() -> List[int]:
    """所有 GPU id，按 nvidia-smi 顺序。"""
    out = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader,nounits"],
        stderr=subprocess.STDOUT,
    ).decode("utf-8")
    ids: List[int] = []
    for line in out.splitlines():
        line = line.strip()
        if line.isdigit():
            ids.append(int(line))
    return ids


def gpu_is_idle(gpu_id: int) -> bool:
    """GPU 没有 compute 进程占用即视为空闲（与 run_tasks.py 保持一致）。"""
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-compute-apps=pid",
                f"--id={gpu_id}",
                "--format=csv,noheader",
            ],
            stderr=subprocess.STDOUT,
        ).decode("utf-8").strip()
    except subprocess.CalledProcessError:
        return False
    return not out


def list_free_gpus(exclude: Iterable[int], reserved: Iterable[int]) -> List[int]:
    banned = set(exclude) | set(reserved)
    return [g for g in list_all_gpus() if g not in banned and gpu_is_idle(g)]


# ---------------------------------------------------------------------------
# reserve-file (本调度器会话的 "刚挑过" 账本)
# ---------------------------------------------------------------------------
def _read_reserved(path: str) -> Set[int]:
    if not path or not os.path.isfile(path):
        return set()
    out: Set[int] = set()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line.isdigit():
                out.add(int(line))
    return out


def _write_reserved(path: str, reserved: Set[int]) -> None:
    if not path:
        return
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = f"{path}.tmp.{os.getpid()}"
    with open(tmp, "w", encoding="utf-8") as f:
        for g in sorted(reserved):
            f.write(f"{g}\n")
    os.replace(tmp, path)


def _reserve(path: str, gpu_id: int) -> None:
    reserved = _read_reserved(path)
    reserved.add(gpu_id)
    _write_reserved(path, reserved)


def _release(path: str, gpu_id: int) -> None:
    reserved = _read_reserved(path)
    reserved.discard(gpu_id)
    _write_reserved(path, reserved)


# ---------------------------------------------------------------------------
# 子命令实现
# ---------------------------------------------------------------------------
def cmd_wait(args: argparse.Namespace) -> int:
    exclude = set(args.exclude or [])
    if not _nvidia_smi_ok():
        print("[gpu_sched] nvidia-smi 不可用，回退到 CPU (返回 -1)", file=sys.stderr)
        print("-1")
        return 0

    poll = max(1, int(args.poll_interval))
    min_free = max(1, int(args.min_free))
    started = time.time()
    last_msg = 0.0

    while True:
        reserved = _read_reserved(args.reserve_file) if args.reserve_file else set()
        free = list_free_gpus(exclude=exclude, reserved=reserved)

        if len(free) >= min_free:
            gpu_id = free[0]
            if args.reserve_file:
                _reserve(args.reserve_file, gpu_id)
            print(gpu_id)
            return 0

        # 打印等待信息，避免长时间静默
        now = time.time()
        if now - last_msg >= max(30, poll * 2):
            elapsed = int(now - started)
            print(
                f"[gpu_sched] 等待空闲 GPU: free={sorted(free)} reserved={sorted(reserved)} "
                f"需要>={min_free}, exclude={sorted(exclude)}, 已等待 {elapsed}s",
                file=sys.stderr,
            )
            last_msg = now

        if args.timeout and (now - started) >= args.timeout:
            print(
                f"[gpu_sched] 等待超时 ({args.timeout}s)，没有足够空闲 GPU",
                file=sys.stderr,
            )
            return 2

        time.sleep(poll)


def cmd_release(args: argparse.Namespace) -> int:
    if args.reserve_file and args.gpu is not None and args.gpu >= 0:
        _release(args.reserve_file, int(args.gpu))
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    exclude = set(args.exclude or [])
    reserved = _read_reserved(args.reserve_file) if args.reserve_file else set()
    all_ids = list_all_gpus()
    free = list_free_gpus(exclude=exclude, reserved=reserved)
    print(f"all      : {all_ids}")
    print(f"exclude  : {sorted(exclude)}")
    print(f"reserved : {sorted(reserved)}")
    print(f"free     : {free}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="基于 nvidia-smi 的 GPU 调度器（for har_baselines）"
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_wait = sub.add_parser("wait", help="阻塞轮询直到有足够空闲 GPU")
    p_wait.add_argument("--min-free", type=int, default=2,
                        help="启动前必须的空闲 GPU 数量 (默认 2: 剩余>=2 才挂, 启动后仍保留 >=1 张)")
    p_wait.add_argument("--exclude", type=int, nargs="*", default=[2],
                        help="永远不使用的 GPU id 列表 (默认 [2])")
    p_wait.add_argument("--reserve-file", type=str, default="",
                        help="本会话已派发 GPU 的账本文件路径")
    p_wait.add_argument("--poll-interval", type=int, default=20,
                        help="轮询间隔 (秒)")
    p_wait.add_argument("--timeout", type=int, default=0,
                        help="最大等待时间 (秒)，0 表示无限")

    p_rel = sub.add_parser("release", help="从 reserve-file 中移除一张 GPU")
    p_rel.add_argument("--gpu", type=int, required=True)
    p_rel.add_argument("--reserve-file", type=str, required=True)

    p_sta = sub.add_parser("status", help="打印当前 GPU 占用/预留状态")
    p_sta.add_argument("--exclude", type=int, nargs="*", default=[2])
    p_sta.add_argument("--reserve-file", type=str, default="")

    args = parser.parse_args()
    if args.cmd == "wait":
        return cmd_wait(args)
    if args.cmd == "release":
        return cmd_release(args)
    if args.cmd == "status":
        return cmd_status(args)
    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
