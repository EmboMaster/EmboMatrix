import subprocess
import multiprocessing
import os
import traceback
from datetime import datetime

FOLDER_PATH = "omnigibson/shengyin/results0506-layoutgpt/"

LOG_FILE = "omnigibson/shengyin/results0506-layoutgpt/error_log.txt"

already_file = "omnigibson/shengyin/results0506-layoutgpt/already_log.txt"

os.makedirs(FOLDER_PATH, exist_ok=True)

if not os.path.exists(LOG_FILE):
    with open(LOG_FILE, "w") as log_file:
        log_file.write("")

if not os.path.exists(already_file):
    with open(already_file, "w") as log_file:
        log_file.write("")

# 记录错误日志
def log_error(message):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(LOG_FILE, "a") as log_file:
        log_file.write(f"[{timestamp}] {message}\n")

def log_run(message):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(already_file, "a") as log_file:
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
            '--save_final_path', "omnigibson/shengyin/results0506-layoutgpt",
            '--using_layoutgpt', True
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

def check_bddl_file_in_logs(bddl_file, logs_dir="omnigibson/shengyin/results0506-layoutgpt/logs"):
    # 遍历logs_dir下所有.txt文件

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
    
    gpu_usage = {gpu_id: 0 for gpu_id in available_gpus}  # 记录每个 GPU 的使用数
    tasks_to_execute = []

    for scene_name, room_id, bddl_files in room_info:
        if room_id in skip_room_ids:
            continue

        for bddl_file in bddl_files:
            # 选择当前占用最少的 GPU
            
            bddl_file = bddl_file.strip("'")
            scene_final_path = f"omnigibson/shengyin/results0506-layoutgpt/{scene_name}/{os.path.basename(os.path.dirname(bddl_file))}.json"
            
            if os.path.exists(scene_final_path):
                print(f"Result file already exists: {scene_final_path}")
                continue  # 结果文件已存在，跳过

            if not os.path.exists(scene_final_path):
                if check_bddl_file_in_logs(bddl_file):
                    print(f"Result file already failed: {scene_final_path}")
                    continue

            gpu_id = min(gpu_usage, key=gpu_usage.get)
            gpu_usage[gpu_id] += 1  # 增加该 GPU 的使用数

            tasks_to_execute.append((bddl_file, gpu_id, scene_name))
    
    with multiprocessing.Pool(processes=len(available_gpus)) as pool:
        results = []

        def on_task_complete(_):
            gpu_usage[gpu_id] -= 1  # 任务完成后减少 GPU 负载

        def on_task_fail(e):
            gpu_usage[gpu_id] -= 1  # 失败也要减少 GPU 负载
            log_error(f"Task failed: {e}")

        for bddl_file, gpu_id, scene_name in tasks_to_execute:
            # result = pool.apply_async(execute_task, (bddl_file, gpu_id, scene_name), callback=lambda _: gpu_usage.__setitem__(gpu_id, gpu_usage[gpu_id] - 1))
            result = pool.apply_async(
                execute_task, 
                (bddl_file, gpu_id, scene_name), 
                callback=on_task_complete,
                error_callback=on_task_fail
            )
            results.append(result)

        # 等待所有任务完成
        for result in results:
            try:
                result.get()  # 确保异常不会影响其他任务
            except Exception as e:
                message = f"Error in task: {e}"
                print(message)
                log_error(message)
                log_error(traceback.format_exc())


if __name__ == "__main__":
    available_gpus = [0, 1, 2, 3, 4, 5, 6, 7]
    room_info = read_room_info('omnigibson/shengyin/withoutModify_bddlfile_0506-layoutgpt.txt')

    skip_room_ids = []  # 需要跳过的 room_id
    assign_gpus_and_execute(room_info, available_gpus, skip_room_ids)
