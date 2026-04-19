import json
import os

# 生成任务列表
def generate_tasks():
    # 定义不同的dirichlet alpha值
    alpha_values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0]
    
    # 定义batch size值为16
    batch_sizes = [16]
    
    # 定义数据集
    dataset = "PEMS-SF"
    
    # 定义epoch数
    num_epochs = 300
    
    # 定义客户端选择方法
    client_selection_methods = ["uniform", "fedcs", "omp", "oort"]
    
    # 生成任务列表
    tasks = []
    task_id = 1
    
    for alpha in alpha_values:
        for batch_size in batch_sizes:
            for method in client_selection_methods:
                # 根据不同的客户端选择方法设置不同的ema-alpha
                ema_alpha = 0.4 if method == "omp" else 0
                
                task = {
                    "id": task_id,
                    "dataset": dataset,
                    "dirichlet_alpha": alpha,
                    "batch_size": batch_size,
                    "num_epochs": num_epochs,
                    "config": "configACF.yml",
                    "use_client_selection": True,
                    "client_selection_method": method,
                    "client_selection_ratio": 0.7,
                    "min_selection_prob": 0.01,
                    "ema_alpha": ema_alpha,
                    "description": f"{method}+acf",
                    "status": "pending",
                    "gpu": None,
                    "pid": None,
                    "start_time": None,
                    "end_time": None,
                    "completion": 0
                }
                tasks.append(task)
                task_id += 1
    
    # 保存任务列表到文件
    with open("tasks.json", "w") as f:
        json.dump(tasks, f, indent=2)
    
    print(f"生成了 {len(tasks)} 个任务")
    print("任务列表已保存到 tasks.json 文件")

if __name__ == "__main__":
    generate_tasks()
