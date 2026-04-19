import os
import json
import time
import subprocess
import re
import signal
import datetime

# 检查可用GPU
def check_available_gpus():
    try:
        output = subprocess.check_output(['nvidia-smi', '--query-gpu=index,utilization.gpu', '--format=csv,noheader,nounits'])
        output = output.decode('utf-8').strip()
        lines = output.split('\n')
        
        available_gpus = []
        for line in lines:
            if line:
                gpu_id, utilization = line.split(',')
                gpu_id = int(gpu_id.strip())
                utilization = int(utilization.strip())
                
                # 检查GPU是否有进程在运行
                process_output = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', f'--id={gpu_id}', '--format=csv,noheader'])
                process_output = process_output.decode('utf-8').strip()
                
                if not process_output:  # 没有进程在运行
                    available_gpus.append(gpu_id)
        
        return available_gpus
    except Exception as e:
        print(f"检查GPU时出错: {e}")
        return []

# 读取任务列表
def load_tasks():
    try:
        with open("tasks.json", "r") as f:
            tasks = json.load(f)
        return tasks
    except Exception as e:
        print(f"读取任务列表时出错: {e}")
        return []

# 保存任务列表
def save_tasks(tasks):
    try:
        with open("tasks.json", "w") as f:
            json.dump(tasks, f, indent=2)
    except Exception as e:
        print(f"保存任务列表时出错: {e}")

# 检查任务是否完成
def check_task_completion(task):
    # 这里可以根据实际情况检查任务是否完成
    # 例如检查日志文件或进程是否存在
    if task.get("pid"):
        try:
            os.kill(task["pid"], 0)
            return False  # 进程仍在运行
        except:
            return True  # 进程已结束
    return False

# 创建日志目录
def create_log_directory():
    log_dir = "task_logs"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    return log_dir

# 执行任务
def run_task(task, gpu_id):
    dataset = task["dataset"]
    alpha = task["dirichlet_alpha"]
    batch_size = task.get("batch_size", 8)  # 默认batch size为8
    num_epochs = task["num_epochs"]
    config = task.get("config", "configACF.yml")
    use_client_selection = task.get("use_client_selection", True)
    client_selection_method = task.get("client_selection_method", "omp")
    client_selection_ratio = task.get("client_selection_ratio", 0.7)
    min_selection_prob = task.get("min_selection_prob", 0.01)
    ema_alpha = task.get("ema_alpha", 0.4)
    description = task.get("description", f"{client_selection_method}+acf")
    
    # 创建日志目录
    log_dir = create_log_directory()
    
    # 生成日志文件名
    log_file = os.path.join(log_dir, f"task_{task['id']}_alpha_{alpha}_batch_{batch_size}_{client_selection_method}.log")
    
    # 构建命令
    command = f"CUDA_VISIBLE_DEVICES={gpu_id} python -u FedCSL_All.py -dataset {dataset} --config {config} --dirichlet-alpha {alpha} --batch-size {batch_size} --use-client-selection --client-selection-method {client_selection_method} --client-selection-ratio {client_selection_ratio} --min-selection-prob {min_selection_prob} --ema-alpha {ema_alpha} --description \"{description}\" > {log_file} 2>&1"
    
    # 执行命令
    print(f"在GPU {gpu_id} 上执行任务: {command}")
    
    # 启动进程
    try:
        process = subprocess.Popen(command, shell=True)
        
        # 更新任务状态
        task["status"] = "running"
        task["gpu"] = gpu_id
        task["pid"] = process.pid
        task["start_time"] = time.time()
        task["log_file"] = log_file
    except Exception as e:
        # 记录错误
        task["status"] = "failed"
        task["error"] = str(e)
        print(f"执行任务时出错: {e}")
        return task
    
    # 记录任务开始
    with open(log_file, "a") as f:
        f.write(f"任务开始时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"GPU ID: {gpu_id}\n")
        f.write(f"Dirichlet Alpha: {alpha}\n")
        f.write(f"Batch Size: {batch_size}\n")
        f.write(f"数据集: {dataset}\n")
        f.write(f"Epochs: {num_epochs}\n")
        f.write("\n")
    
    return task

# 主函数
def main():
    # 加载任务列表
    tasks = load_tasks()
    
    if not tasks:
        print("没有任务可执行")
        return
    
    # 检查是否有未完成的任务
    running_tasks = [task for task in tasks if task["status"] == "running"]
    if running_tasks:
        print(f"发现 {len(running_tasks)} 个正在运行的任务，开始监控")
    
    while True:
        # 检查可用GPU
        available_gpus = check_available_gpus()
        print(f"可用GPU: {available_gpus}")
        
        # 检查运行中的任务
        for task in tasks:
            if task["status"] == "running":
                if check_task_completion(task):
                    # 任务已完成
                    task["status"] = "completed"
                    task["end_time"] = time.time()
                    task["completion"] = 100
                    print(f"任务 {task['id']} 已完成")
                    
                    # 记录任务完成
                    if "log_file" in task and os.path.exists(task["log_file"]):
                        with open(task["log_file"], "a") as f:
                            f.write(f"\n任务完成时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        # 保存任务状态
        save_tasks(tasks)
        
        # 分配任务到可用GPU
        for gpu_id in available_gpus:
            # 检查该GPU是否已分配任务
            gpu_assigned = False
            for task in tasks:
                if task["status"] == "running" and task["gpu"] == gpu_id:
                    gpu_assigned = True
                    break
            
            if not gpu_assigned:
                # 寻找待执行的任务
                for task in tasks:
                    if task["status"] == "pending":
                        # 分配任务到GPU
                        run_task(task, gpu_id)
                        save_tasks(tasks)
                        break
        
        # 检查是否所有任务都已完成
        all_completed = True
        for task in tasks:
            if task["status"] != "completed":
                all_completed = False
                break
        
        if all_completed:
            print("所有任务已完成")
            # 生成任务完成报告
            generate_report(tasks)
            break
        
        # 等待一段时间后再次检查
        time.sleep(60)

# 生成任务完成报告
def generate_report(tasks):
    report_file = "task_report.txt"
    with open(report_file, "w") as f:
        f.write("任务执行报告\n")
        f.write("=" * 50 + "\n")
        f.write(f"生成时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"总任务数: {len(tasks)}\n")
        
        completed_tasks = [task for task in tasks if task["status"] == "completed"]
        f.write(f"已完成任务数: {len(completed_tasks)}\n")
        f.write("\n")
        
        f.write("任务详情:\n")
        f.write("-" * 50 + "\n")
        
        for task in tasks:
            f.write(f"任务ID: {task['id']}\n")
            f.write(f"Dirichlet Alpha: {task['dirichlet_alpha']}\n")
            if 'batch_size' in task:
                f.write(f"Batch Size: {task['batch_size']}\n")
            f.write(f"状态: {task['status']}\n")
            if task.get("start_time"):
                start_time = datetime.datetime.fromtimestamp(task["start_time"]).strftime('%Y-%m-%d %H:%M:%S')
                f.write(f"开始时间: {start_time}\n")
            if task.get("end_time"):
                end_time = datetime.datetime.fromtimestamp(task["end_time"]).strftime('%Y-%m-%d %H:%M:%S')
                f.write(f"结束时间: {end_time}\n")
                # 计算执行时间
                execution_time = task["end_time"] - task["start_time"]
                f.write(f"执行时间: {execution_time:.2f} 秒\n")
            if task.get("log_file"):
                f.write(f"日志文件: {task['log_file']}\n")
            f.write("-" * 50 + "\n")
    
    print(f"任务执行报告已生成: {report_file}")

if __name__ == "__main__":
    main()
