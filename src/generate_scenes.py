import subprocess
import multiprocessing
import os
import traceback
from datetime import datetime
import argparse
from src.utils.config_loader import config
FOLDER_PATH = config['scene_generation']['output_dir']
LOG_FILE = FOLDER_PATH + "/error_log.txt"
ALREADY_FILE = FOLDER_PATH + "/already_log.txt"
logs_dir = os.path.join(FOLDER_PATH,"logs")
os.makedirs(FOLDER_PATH, exist_ok=True)

if not os.path.exists(LOG_FILE):
    with open(LOG_FILE, "w") as log_file:
        log_file.write("")

if not os.path.exists(ALREADY_FILE):
    with open(ALREADY_FILE, "w") as log_file:
        log_file.write("")

# 解析命令行参数
def parse_args():
    parser = argparse.ArgumentParser(description="Run tasks on a split of the data")
    parser.add_argument('--split_id', type=int, choices=[0, 1, 2, 3, 4, 5, 6], default=4,
                        help="Which split of the data to process (0 to 5)")
    return parser.parse_args()

# 记录错误日志
def log_error(message):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(LOG_FILE, "a") as log_file:
        log_file.write(f"[{timestamp}] {message}\n")

def log_run(message):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(ALREADY_FILE, "a") as log_file:
        log_file.write(f"[{timestamp}] {message}\n")

# 读取txt文件内容并提取Room信息
def read_room_info(file_path):
    room_info = []
    scene_name = None
    room_id = None
    bddl_files = []

    try:
        with open(file_path, 'r') as file:
            lines = file.readlines()
            for line in lines:
                line = line.strip()
                if line.startswith("Room:"):
                    scene_name = line.split(":")[-1].strip()
                elif line.startswith("Room id:"):
                    room_id = int(line.split(":")[-1].strip())
                elif "problem0.bddl" in line:
                    bddl_files.append(line)

                if scene_name and room_id is not None and bddl_files:
                    room_info.append((scene_name, room_id, bddl_files))
                    bddl_files = []
    except Exception as e:
        log_error(f"Error reading room info from {file_path}: {e}")
        log_error(traceback.format_exc())

    return room_info

# 分割任务数据
def split_tasks(room_info, split_id, num_splits=6):
    """将任务按 split_id 分成 num_splits 份，返回对应 split_id 的任务"""
    total_tasks = len(room_info)
    split_size = total_tasks // num_splits
    remainder = total_tasks % num_splits

    # 计算每份的起始和结束索引
    start_idx = split_id * split_size + min(split_id, remainder)
    end_idx = start_idx + split_size + (1 if split_id < remainder else 0)

    return room_info[start_idx:end_idx]

# 执行任务的函数
def execute_task(bddl_file, gpu_id, scene_name):
    log_run(f"RUN: {bddl_file}_{scene_name}")

    retrial_times = 2

    for i in range(retrial_times):

        command = (
            f'export PYTHONPATH="bddl:$PYTHONPATH" && '
            f'python src/scene_generation_pipeline.py '
            f'--bddl_directory {bddl_file} '
            f'--gpu_id {str(gpu_id)} '
            f'--scene_name {scene_name} '
            f'--save_final_path {FOLDER_PATH}'
        )

        try:
            print(f"Starting: {bddl_file} on GPU {gpu_id}")
            subprocess.run(command, timeout=2400, check=True, shell=True) 
            print(f"Finished: {bddl_file} successfully.")
            log_run(f"Success: {bddl_file}_{scene_name}")
            return
        except subprocess.TimeoutExpired:
            message = f"Timeout reached for {bddl_file} on GPU {gpu_id}. Skipping."
            print(message)
            log_error(message)
            log_run(f"Fail: {message}")
            continue
        except Exception as e:
            message = f"Error in {bddl_file} on GPU {gpu_id}: {e}"
            print(message)
            log_error(message)
            log_error(traceback.format_exc())
            log_run(f"Fail: {message}")
            return

def check_bddl_file_in_logs(bddl_file, logs_dir=logs_dir):
    if not os.path.exists(logs_dir):
        return False

    for filename in os.listdir(logs_dir):
        if filename.endswith(".txt"):
            file_path = os.path.join(logs_dir, filename)
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()
                    if bddl_file in content and "No space left on device" not in content and "System sludge is not a valid system name" not in content and "Disk quota exceeded" not in content:
                        return True
            except Exception as e:
                print(f"Error reading {file_path}: {e}")
    return False

# 分配GPU并行执行任务，跳过指定的room_id
def assign_gpus_and_execute(room_info, available_gpus, skip_room_ids=None):
    if skip_room_ids is None:
        skip_room_ids = []
    
    gpu_usage = {gpu_id: 0 for gpu_id in available_gpus}
    tasks_to_execute = []

    for scene_name, room_id, bddl_files in room_info:
        if room_id in skip_room_ids:
            continue

        for bddl_file in bddl_files:
            bddl_file = bddl_file.strip("'")
            scene_final_path = FOLDER_PATH + f"/{scene_name}/{os.path.basename(os.path.dirname(bddl_file))}.json" 
            gpu_id = min(gpu_usage, key=gpu_usage.get)
            gpu_usage[gpu_id] += 1

            tasks_to_execute.append((bddl_file, gpu_id, scene_name))
    
    with multiprocessing.Pool(processes=len(available_gpus)) as pool:
        results = []

        def on_task_complete(_):
            gpu_usage[gpu_id] -= 1

        def on_task_fail(e):
            gpu_usage[gpu_id] -= 1
            log_error(f"Task failed: {e}")

        for bddl_file, gpu_id, scene_name in tasks_to_execute:
            result = pool.apply_async(
                execute_task, 
                (bddl_file, gpu_id, scene_name), 
                callback=on_task_complete,
                error_callback=on_task_fail
            )
            results.append(result)

        for result in results:
            try:
                result.get()
            except Exception as e:
                message = f"Error in task: {e}"
                print(message)
                log_error(message)
                log_error(traceback.format_exc())

if __name__ == "__main__":
    args = parse_args()
    split_id = args.split_id
    available_gpus = config['scene_generation']['available_gpus']
    preprocessed_task_list_path = config['scene_generation']['preprocessed_task_list_path']
    room_info = read_room_info(preprocessed_task_list_path)
    skip_room_ids = []

    if int(split_id) == 0:
        assign_gpus_and_execute(room_info, available_gpus, skip_room_ids)
    else:
        # 分割任务数据
        print(f"Machine {split_id}!")
        split_room_info = split_tasks(room_info, split_id-1, num_splits=6)
        print(f"Processing split {split_id}/5 with {len(split_room_info)} tasks")
        assign_gpus_and_execute(split_room_info, available_gpus, skip_room_ids)

    
    