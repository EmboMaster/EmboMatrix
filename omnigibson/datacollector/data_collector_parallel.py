#======================= 控制脚本 control_script.py =======================
import os
import threading
import subprocess
import argparse
import select
import time
from datetime import datetime

def get_available_gpus():
    result = subprocess.check_output(
        "nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits", 
        shell=True
    ).decode().strip().split('\n')
    
    available_gpus = []
    for line in result:
        gpu_id, mem_used = map(int, line.split(', '))
        if mem_used < 20000:  # 显存使用小于4GB视为可用
            available_gpus.append(gpu_id)
    return available_gpus

def distribute_tasks(base_dir, gpu_list):
    """将全部任务平铺分配到各GPU"""
    all_tasks = []
    scenes = os.listdir(base_dir)
    scenes.sort()
    
    # 收集所有场景的所有任务
    for scene_name in scenes:
        if 'garden' in scene_name:
            continue
        scene_path = os.path.join(base_dir, scene_name)
        json_files = [f for f in os.listdir(scene_path) if f.endswith('.json')]
        for json_file in json_files:
            all_tasks.append({
                "scene_path": scene_path,
                "scene_name": scene_name,
                "task_file": json_file
            })
    
    gpu_list = [0]
    # 轮询分配任务到GPU
    tasks_per_gpu = {gpu: [] for gpu in gpu_list}
    for idx, task in enumerate(all_tasks):
        assigned_gpu = gpu_list[idx % len(gpu_list)]
        tasks_per_gpu[assigned_gpu].append(task)
    return tasks_per_gpu
def run_gpu_tasks(gpu_id, tasks, experiment_path):
    """顺序执行指定GPU上的所有任务（添加超时控制）"""
    for task in tasks:
        scene_path = task["scene_path"]
        task_file = task["task_file"]
        scene_name = task["scene_name"]
        
        # 构建日志路径
        log_dir = f"./data_collector_logs/{scene_name}/{task_file.replace('.json', '')}/"
        os.makedirs(log_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        log_file = f"{log_dir}{timestamp}.log"
        
        # 构建命令
        # cmd = [
        #     "python", "omnigibson/datacollector/data_collector.py",
        #     "--scene_path", scene_path,
        #     "--task_file", task_file,
        #     "--gpu_id", str(gpu_id)
        # ]
        cmd = [
            "python", "omnigibson/datacollector/data_collector.py",
            "--scene_path", scene_path,
            "--task_file", task_file,
            "--gpu_id", str(0)
        ]

        # 执行任务并记录日志
        with open(log_file, "w") as f:
            p = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1  # 行缓冲模式
            )
            
            start_time = time.time()
            timeout_seconds = 15 * 60  # 30分钟超时
            timed_out = False
            
            while True:
                # 检查超时
                if time.time() - start_time > timeout_seconds:
                    timed_out = True
                    f.write("\n[SYSTEM] 任务执行超时，强制终止...\n")
                    p.terminate()  # 先尝试正常终止
                    try:
                        p.wait(timeout=5)  # 等待5秒
                    except subprocess.TimeoutExpired:
                        p.kill()  # 强制杀死进程
                    break
                
                # 非阻塞读取输出
                ready, _, _ = select.select([p.stdout], [], [], 1)
                if ready:
                    line = p.stdout.readline()
                    if line:
                        if "warning" not in line.lower():
                            f.write(line)
                            f.flush()
                    else:  # EOF
                        break  # 进程正常退出
                
                # 检查进程状态
                if p.poll() is not None:
                    break  # 进程已结束
                
            # 记录最终状态
            if timed_out:
                status_msg = f"[GPU {gpu_id}] {task_file} 执行超时"
            elif p.returncode == 0:
                status_msg = f"[GPU {gpu_id}] {task_file} 执行成功"
            else:
                status_msg = f"[GPU {gpu_id}] {task_file} 异常退出（代码 {p.returncode}）"
            
            f.write(f"\n{status_msg}\n")
            print(status_msg)
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment_path", type=str, required=True)
    args = parser.parse_args()
    
    # 获取可用GPU并分配任务
    gpus = get_available_gpus()
    if not gpus:
        raise RuntimeError("没有可用的GPU")
    
    tasks_per_gpu = distribute_tasks(args.experiment_path, gpus)
    
    # 为每个GPU启动独立线程
    threads = []
    for gpu_id, tasks in tasks_per_gpu.items():
        if not tasks:
            continue
        thread = threading.Thread(
            target=run_gpu_tasks,
            args=(gpu_id, tasks, args.experiment_path)
        )
        thread.start()
        threads.append(thread)
    
    # 等待所有线程完成
    for thread in threads:
        thread.join()

if __name__ == "__main__":
    main()