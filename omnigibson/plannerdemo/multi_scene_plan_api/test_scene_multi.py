from env_reward_fastapi import start_server 
import os
import multiprocessing
def get_tasks_to_execute(base_dir):
    tasks_to_execute = []
    for root, dirs, files in os.walk(base_dir):
        for dir_name in dirs:
            scene_path = os.path.join(root, dir_name)
            for file in os.listdir(scene_path):
                if file.endswith('.bddl'):
                    task_name = os.path.splitext(file)[0]  # 去掉.bddl后缀
                    tasks_to_execute.append((dir_name, task_name))
    
    return tasks_to_execute

def execute_task(scene,task,cuda_device_index,port):
    try:
        success = start_server(scene, task, cuda_device_index, port)
        log_message = f"===\nScene: {scene}, Task: {task}, CUDA Device Index: {cuda_device_index}, Success: {success}\n"
        log_to_file(log_message)
        return True
    except Exception as e:
        error_message = str(e)
        log_message = f"===\nScene: {scene}, Task: {task}, CUDA Device Index: {cuda_device_index}, Error: {error_message}\n"
        log_to_file(log_message)
        return False
    
def log_to_file(message):
    with open('/data/zxlei/embodied/planner_bench/omnigibson/result0304/log.txt', 'a') as log_file:
        log_file.write(message + '\n')

def parallel_execution(tasks_to_execute, available_gpus, port):
    # 使用 multiprocessing.Pool 并行执行任务
    with multiprocessing.Pool(processes=4) as pool:
        results = pool.starmap(execute_task, [(scene, task, available_gpus[i % len(available_gpus)], port) for i, (scene, task) in enumerate(tasks_to_execute)])
if __name__ == "__main__":
    # 设置可用的GPU（最多四个）
    available_gpus = [5,6,7]
    base_dir = '/data/zxlei/embodied/planner_bench/omnigibson/result0304'  
    port = 5000  

    # 获取任务列表
    tasks_to_execute = get_tasks_to_execute(base_dir)

    # 开始并行执行任务
    parallel_execution(tasks_to_execute, available_gpus, port)