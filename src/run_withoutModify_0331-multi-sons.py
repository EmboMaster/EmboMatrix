import subprocess
import multiprocessing
import os
import traceback
from datetime import datetime
import argparse

FOLDER_PATH = "omnigibson/shengyin/results0511/"
LOG_FILE = "omnigibson/shengyin/results0511/error_log.txt"
ALREADY_FILE = "omnigibson/shengyin/results0511/already_log.txt"

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
    parser.add_argument('--split_id', type=int, required=True,
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
# def split_tasks(room_info, split_id, num_splits=41):
#     """将任务按 split_id 分割，split_id=0 取前1/6，split_id=1到20 各取后5/6的1/20"""
#     total_tasks = len(room_info)
#     first_split_size = total_tasks // 6  # 前1/6
#     remaining_tasks = total_tasks - first_split_size  # 后5/6
#     sub_split_size = remaining_tasks // 40  # 后5/6分成20份
#     remainder = remaining_tasks % 40

#     if split_id == 0:
#         # split_id=0 取前1/6
#         return room_info[:first_split_size]
#     else:
#         # split_id=1到20 取后5/6的对应1/20
#         start_idx = first_split_size + (split_id - 1) * sub_split_size + min(split_id - 1, remainder)
#         end_idx = start_idx + sub_split_size + (1 if split_id - 1 < remainder else 0)
#         return room_info[start_idx:end_idx]

def split_tasks(room_info, split_id, num_splits=40):
    """将任务均分成 num_splits 份（默认 40 份），根据 split_id 返回对应份的任务"""
    total_tasks = len(room_info)
    split_size = total_tasks // num_splits  # 每份的基本大小
    remainder = total_tasks % num_splits  # 余数，用于分配到前 remainder 份

    # 计算当前 split_id 的起始和结束索引
    start_idx = split_id * split_size + min(split_id, remainder)
    end_idx = start_idx + split_size + (1 if split_id < remainder else 0)

    return room_info[start_idx:end_idx]


# 执行任务的函数
def execute_task(bddl_file, gpu_id, scene_name):
    log_run(f"RUN: {bddl_file}_{scene_name}")

    retrial_times = 2

    for i in range(retrial_times):
        command = [
            'python',
            'omnigibson/shengyin/ddl_pipeline.py',
            '--bddl_directory', bddl_file,
            '--gpu_id', str(gpu_id),
            '--scene_name', scene_name,
            '--save_final_path', "omnigibson/shengyin/results0511",
        ]
        try:
            print(f"Starting: {bddl_file} on GPU {gpu_id}")
            subprocess.run(command, timeout=1500, check=True)  # 10分钟超时
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

def check_bddl_file_in_logs(bddl_file, logs_dir="omnigibson/shengyin/results0511/logs"):
    if not os.path.exists(logs_dir):
        return False

    for filename in os.listdir(logs_dir):
        if filename.endswith(".txt"):
            file_path = os.path.join(logs_dir, filename)
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()
                    if bddl_file in content and "No space left on device" not in content:
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
            scene_final_path = f"omnigibson/shengyin/results0511/{scene_name}/{os.path.basename(os.path.dirname(bddl_file))}.json"
            
            if os.path.exists(scene_final_path):
                print(f"Result file already exists: {scene_final_path}")
                continue

            if not os.path.exists(scene_final_path):
                if check_bddl_file_in_logs(bddl_file):
                    print(f"Result file already failed: {scene_final_path}")
                    continue

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

    # if int(split_id) == 0 or int(split_id) == 1:
    #     available_gpus = [0, 1, 2, 3, 4, 5, 6, 7]
    # elif int(split_id) == 42 or int(split_id) == 50 or int(split_id) == 51 or int(split_id) == 52:
    #     available_gpus = [0, 1, 2, 3]
    # elif int(split_id) == 43 or int(split_id) == 44 or int(split_id) == 45 or int(split_id) == 47 or int(split_id) == 48 or int(split_id) == 49:
    #     available_gpus = [0, 1]
    # else:
    #     available_gpus = [0]

    room_info = read_room_info('omnigibson/shengyin/withoutModify_bddlfile_0511.txt')
    skip_room_ids = []

    # if int(split_id) == 0 or int(split_id) == 52:
    #     assign_gpus_and_execute(room_info, available_gpus, skip_room_ids)
    # else:
    #     # 分割任务数据
    #     print(f"Machine {split_id}!")
    #     if int(split_id) == 42:
    #         split_room_info = split_tasks(room_info, 30, num_splits=6)
    #         split_room_info.extend(split_tasks(room_info, 31, num_splits=6))
    #         split_room_info.extend(split_tasks(room_info, 32, num_splits=6))
    #         split_room_info.extend(split_tasks(room_info, 33, num_splits=6))
    #     elif int(split_id) == 43:
    #         split_room_info = split_tasks(room_info, 6, num_splits=6)
    #         split_room_info.extend(split_tasks(room_info, 7, num_splits=6))
    #         split_room_info.extend(split_tasks(room_info, 9, num_splits=6))
    #         split_room_info.extend(split_tasks(room_info, 10, num_splits=6))
    #     elif int(split_id) == 44:
    #         split_room_info = split_tasks(room_info, 11, num_splits=6)
    #         split_room_info.extend(split_tasks(room_info, 12, num_splits=6))
    #         split_room_info.extend(split_tasks(room_info, 13, num_splits=6))
    #     elif int(split_id) == 45:
    #         split_room_info = split_tasks(room_info, 14, num_splits=6)
    #         split_room_info.extend(split_tasks(room_info, 15, num_splits=6))
    #         split_room_info.extend(split_tasks(room_info, 16, num_splits=6))
    #     elif int(split_id) == 47:
    #         split_room_info = split_tasks(room_info, 17, num_splits=6)
    #         split_room_info.extend(split_tasks(room_info, 18, num_splits=6))
    #     elif int(split_id) == 48:
    #         split_room_info = split_tasks(room_info, 20, num_splits=6)
    #         split_room_info.extend(split_tasks(room_info, 21, num_splits=6))
    #     elif int(split_id) == 49:
    #         split_room_info = split_tasks(room_info, 23, num_splits=6)
    #         split_room_info.extend(split_tasks(room_info, 24, num_splits=6))
    #     elif int(split_id) == 51:
    #         split_room_info = split_tasks(room_info, 17, num_splits=6)
    #         split_room_info.extend(split_tasks(room_info, 18, num_splits=6))
    #         split_room_info.extend(split_tasks(room_info, 19, num_splits=6))
    #     else:
    #         if int(split_id) == 46 or int(split_id) == 50:
    #             split_id = 1
    #         split_room_info = split_tasks(room_info, split_id-1, num_splits=6)
    #     print(f"Processing split {split_id}/41 with {len(split_room_info)} tasks")
    #     assign_gpus_and_execute(split_room_info, available_gpus, skip_room_ids)

    if int(split_id) in [0, 1, 2]:
        available_gpus = [0, 1]  # 2 张 GPU
    elif int(split_id) == 3:
        available_gpus = [0, 1, 2, 3]  # 4 张 GPU
    elif int(split_id) == 28:
        available_gpus = [0, 1, 2, 3, 4, 5, 6, 7]  # 4 张 GPU
    else:
        available_gpus = [0]  # 1 张 GPU

    skip_room_ids = []

    # 任务分割
    if int(split_id) in [0, 1, 2]:
        # 分配 2 份数据
        base_idx = int(split_id) * 2  # 第 0, 2, 4 份
        split_room_info = split_tasks(room_info, base_idx, num_splits=40)
        split_room_info.extend(split_tasks(room_info, base_idx + 1, num_splits=40))
    elif int(split_id) == 3:
        # 分配 4 份数据
        split_room_info = split_tasks(room_info, 6, num_splits=40)
        split_room_info.extend(split_tasks(room_info, 7, num_splits=40))
        split_room_info.extend(split_tasks(room_info, 8, num_splits=40))
        split_room_info.extend(split_tasks(room_info, 9, num_splits=40))
    elif int(split_id) >= 4 and int(split_id) <= 27:
        # 分配 1 份数据
        task_idx = int(split_id) + 6  # 从第 10 份到第 33 份
        split_room_info = split_tasks(room_info, task_idx, num_splits=40)
    elif int(split_id) == 28:
        # 分配剩余所有任务（第 34 份到第 39 份）
        split_room_info = []
        for task_idx in range(34, 40):
            split_room_info.extend(split_tasks(room_info, task_idx, num_splits=40))
    else:
        # split_id 超出范围
        print(f"Error: split_id {split_id} is out of range (0-28)")

    print(f"Machine {split_id}: Processing {len(split_room_info)} tasks with {len(available_gpus)} GPUs")
    assign_gpus_and_execute(split_room_info, available_gpus, skip_room_ids)

    
    