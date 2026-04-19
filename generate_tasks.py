"""
生成 tasks.json，供 run_tasks.py 调度执行。

注意:
- FedCSL_All.py 没有 --num-epochs 参数, epoch 由 yaml 决定。
  这里的 num_epochs 字段 **不会** 被透传, 仅用于人工记录。
- 服务器最近的命令模板（参考 `服务器` 文件最后几行）：

    CUDA_VISIBLE_DEVICES=2 nohup python -u FedCSL_All.py -dataset PEMS-SF \
      --config configACF.yml --use-client-selection \
      --client-selection-method uniform --client-selection-ratio 0.7 \
      --min-selection-prob 0.01 --ema-alpha 0 --description "uniform+acf" &
"""

import json


# ---- 实验设计：按需修改 ----------------------------------------------------
DATASETS = ["PEMS-SF"]                      # 可填多个数据集
ALPHA_VALUES = [0.1, 0.2, 0.3, 0.4, 0.5,
                0.6, 0.7, 0.8, 0.9, 1.0,
                1.5, 2.0, 3.0, 5.0, 10.0]
CLIENT_SELECTION_METHODS = ["uniform", "fedcs", "omp", "oort"]

CONFIG = "configACF.yml"
CLIENT_SELECTION_RATIO = 0.7
MIN_SELECTION_PROB = 0.01
BATCH_SIZE = None  # None 表示用 yaml 里的默认值; 想覆盖再填具体数字
# ---------------------------------------------------------------------------


def make_description(method: str) -> str:
    return f"{method}+acf"


def ema_alpha_for(method: str) -> float:
    # 与服务器命令保持一致: 仅 omp 启用 0.4 的 EMA, 其他取 0
    return 0.4 if method == "omp" else 0.0


def generate_tasks() -> list[dict]:
    tasks: list[dict] = []
    task_id = 1

    for dataset in DATASETS:
        for alpha in ALPHA_VALUES:
            for method in CLIENT_SELECTION_METHODS:
                tasks.append({
                    "id": task_id,
                    "dataset": dataset,
                    "dirichlet_alpha": alpha,
                    "batch_size": BATCH_SIZE,
                    "config": CONFIG,
                    "use_client_selection": True,
                    "client_selection_method": method,
                    "client_selection_ratio": CLIENT_SELECTION_RATIO,
                    "min_selection_prob": MIN_SELECTION_PROB,
                    "ema_alpha": ema_alpha_for(method),
                    "description": make_description(method),
                    "status": "pending",
                    "gpu": None,
                    "pid": None,
                    "start_time": None,
                    "end_time": None,
                    "completion": 0,
                })
                task_id += 1
    return tasks


def main() -> None:
    tasks = generate_tasks()
    with open("tasks.json", "w", encoding="utf-8") as f:
        json.dump(tasks, f, indent=2, ensure_ascii=False)
    print(f"生成了 {len(tasks)} 个任务, 已保存到 tasks.json")


if __name__ == "__main__":
    main()
