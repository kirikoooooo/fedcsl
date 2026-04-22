#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""基于 ``nvidia-smi`` 的简易 GPU 调度器，供 har_baselines shell 脚本调用。

提供三条子命令（均以 reserve-file 作为本会话的预留账本）::

  wait     阻塞轮询，直到满足调度条件后挑一张 GPU 返回
  release  释放指定的 GPU 占用（从 reserve-file 中删除）
  status   打印调度器当前视图（all / exclude / reserved / free）

判定一张 GPU 是否 **可用** 有两种模式，可并存或二选一：

  1. **idle** 模式：GPU 没有任何 compute 进程（``nvidia-smi --query-compute-apps``）
     —— 老版本 har_baseline 的严格模式。
  2. **memory** 模式：GPU 的 **显存空闲比** ≥ ``--mem-free-ratio`` 时即视为可用
     —— 默认模式（阈值 0.7），允许**多个任务共享同一张卡**，
     只要别人没把显存吃到 30% 以上就可以继续插入新任务。

两种模式通过 ``--strategy`` 控制；默认 ``mem``。

为避免 ``nvidia-smi`` 反映更新迟滞导致的瞬时重复派发，依然保留 reserve-file，
不过在 mem 模式下它仅作为 "本轮挑过、还没来得及吃显存" 的短期提示
—— 新版本允许挑中已在 reserve-file 里的卡（当它显存余额仍满足阈值时）。
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from typing import Iterable, List, Set, Tuple


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
    """GPU 没有 compute 进程占用即视为空闲。"""
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


def gpu_mem_info(gpu_id: int) -> Tuple[int, int]:
    """返回 (memory_used_mb, memory_total_mb)；失败时返回 (0, 0)。"""
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=memory.used,memory.total",
                f"--id={gpu_id}",
                "--format=csv,noheader,nounits",
            ],
            stderr=subprocess.STDOUT,
        ).decode("utf-8").strip()
    except subprocess.CalledProcessError:
        return 0, 0
    if not out:
        return 0, 0
    parts = [p.strip() for p in out.split(",")]
    if len(parts) < 2:
        return 0, 0
    try:
        used = int(float(parts[0]))
        total = int(float(parts[1]))
    except ValueError:
        return 0, 0
    return used, total


def gpu_mem_free_ratio(gpu_id: int) -> float:
    used, total = gpu_mem_info(gpu_id)
    if total <= 0:
        return 0.0
    return max(0.0, 1.0 - used / total)


# ---------------------------------------------------------------------------
# 可用卡选取：支持 idle / mem 两种策略
# ---------------------------------------------------------------------------
def list_available_gpus(
    *,
    strategy: str,
    exclude: Iterable[int],
    reserved: Iterable[int],
    mem_free_ratio: float,
) -> List[Tuple[int, float]]:
    """返回 [(gpu_id, mem_free_ratio)]，按 mem_free_ratio 降序（空闲最多的优先）。

    - ``strategy='idle'``：严格空闲模式；reserved 中的 GPU 也被跳过（兼容旧行为）。
    - ``strategy='mem'``：显存阈值模式；**仅** 排除 ``exclude``，不再跳过 reserved，
      也不再强制 "没有进程"，只要 ``mem_free_ratio ≥ threshold`` 就可用。
    """
    banned = set(exclude)
    result: List[Tuple[int, float]] = []
    for g in list_all_gpus():
        if g in banned:
            continue
        free_r = gpu_mem_free_ratio(g)
        if strategy == "idle":
            if g in set(reserved):
                continue
            if not gpu_is_idle(g):
                continue
            # idle 还加一重阈值：避免 idle 但 bug 导致显存没释放
            if free_r < mem_free_ratio:
                continue
            result.append((g, free_r))
        else:  # mem
            if free_r < mem_free_ratio:
                continue
            result.append((g, free_r))
    result.sort(key=lambda x: x[1], reverse=True)
    return result


def list_free_gpus(exclude: Iterable[int], reserved: Iterable[int]) -> List[int]:
    """向后兼容：返回严格空闲的 GPU id 列表（供 status 使用）。"""
    banned = set(exclude) | set(reserved)
    return [g for g in list_all_gpus() if g not in banned and gpu_is_idle(g)]


# ---------------------------------------------------------------------------
# reserve-file
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
    started = time.time()
    last_msg = 0.0
    strategy = args.strategy
    mem_ratio = float(args.mem_free_ratio)

    while True:
        reserved = _read_reserved(args.reserve_file) if args.reserve_file else set()
        avail = list_available_gpus(
            strategy=strategy, exclude=exclude, reserved=reserved, mem_free_ratio=mem_ratio
        )

        # idle 模式沿用 min-free 语义；mem 模式只要有 1 张满足阈值就挑
        min_free = max(1, int(args.min_free)) if strategy == "idle" else 1

        if len(avail) >= min_free:
            gpu_id = avail[0][0]
            if args.reserve_file:
                _reserve(args.reserve_file, gpu_id)
            print(gpu_id)
            return 0

        now = time.time()
        if now - last_msg >= max(30, poll * 2):
            elapsed = int(now - started)
            print(
                f"[gpu_sched] 等待可用 GPU: strategy={strategy} "
                f"mem_free>={mem_ratio:.2f} avail={[(g,round(r,2)) for g,r in avail]} "
                f"reserved={sorted(reserved)} exclude={sorted(exclude)} 已等待 {elapsed}s",
                file=sys.stderr,
            )
            last_msg = now

        if args.timeout and (now - started) >= args.timeout:
            print(f"[gpu_sched] 等待超时 ({args.timeout}s)", file=sys.stderr)
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
    free_idle = list_free_gpus(exclude=exclude, reserved=reserved)
    avail_mem = list_available_gpus(
        strategy="mem", exclude=exclude, reserved=reserved, mem_free_ratio=float(args.mem_free_ratio)
    )
    print(f"all                 : {all_ids}")
    print(f"exclude             : {sorted(exclude)}")
    print(f"reserved            : {sorted(reserved)}")
    print(f"free (idle strict)  : {free_idle}")
    print(f"avail (mem≥{args.mem_free_ratio:.2f}) : {[(g, round(r,2)) for g,r in avail_mem]}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="基于 nvidia-smi 的 GPU 调度器（har_baselines / dashboard 共用）"
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_wait = sub.add_parser("wait", help="阻塞轮询直到有可用 GPU")
    p_wait.add_argument(
        "--strategy", choices=("idle", "mem"), default="mem",
        help="idle=严格空闲 (旧行为); mem=显存阈值 (默认)",
    )
    p_wait.add_argument(
        "--mem-free-ratio", type=float, default=0.7,
        help="mem 模式下的显存空闲比阈值 (默认 0.7; 即已用 <= 30%%)",
    )
    p_wait.add_argument(
        "--min-free", type=int, default=1,
        help="idle 模式下启动前必须的空闲 GPU 数量 (默认 1)",
    )
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
    p_sta.add_argument("--mem-free-ratio", type=float, default=0.7)

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
