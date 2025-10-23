import multiprocessing
import os
import GPUtil
import time
import random
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import requests
from multiprocessing import Manager
import logging
from datetime import datetime
from multiprocessing import Manager
import subprocess

# Create a Manager for shared state.
manager = Manager()
# Shared dictionary: key=(scene, task) -> value=list of ports hosting environments.
env_port_map_dict = manager.dict()

# 全局失败计数器，记录每个端口连续失败的次数
failure_counts = {}

# Define the list of environments to create:
selected_scenes_tasks = [
    ("Pomaria_0_int", "recycling_office_papers", 16),
    # ("restaurant_urban", "clean_up_after_a_dinner_party", 1),
    # ("restaurant_brunch", "set_a_dinner_table", 1),
    # ("house_single_floor", "delivering_groceries_to_doorstep", 2)
]

# 定义日志文件路径和命名格式
log_directory = "/data/zxlei/embodied/planner_bench/models/planserverlog"

log_filename = f"{log_directory}/{datetime.now().strftime('%Y-%m-%d/%H-%M')}.log"

if not os.path.exists(os.path.dirname(log_filename)):
    os.makedirs(os.path.dirname(log_filename))
# 配置日志
logging.basicConfig(filename=log_filename, level=logging.INFO, format='%(asctime)s - %(message)s')

# This function will launch an environment server in a new process.
def launch_server(scene, task, gpu_id, port):
    # Set the CUDA device for this process.
    # os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    print(f"Starting FastAPI server for scene '{scene}', task '{task}' on port {port} with GPU {gpu_id}")
    # Import and start the environment FastAPI server.
    # Note: env_reward_fastapi.py should be in the same directory.
    import env_reward_fastapi
    env_reward_fastapi.start_server(scene, task, gpu_id, port)

# 新增函数：重新初始化指定端口的服务
def reinitialize_service(port, scene, task):
    logging.info(f"Reinitializing service on port {port} for scene '{scene}', task '{task}'")
    key = (scene, task)

    # 查找并 Kill 掉对应端口的服务
    try:
        # 获取所有端口上的进程
        result = subprocess.run(["lsof", "-t", f"-i:{port}"], capture_output=True, text=True)
        if result.stdout:
            pid = result.stdout.strip()
            subprocess.run(["kill", "-9", pid])
            logging.info(f"Service on port {port} killed (PID: {pid})")
        else:
            logging.info(f"No service running on port {port} to kill.")
    except Exception as e:
        logging.error(f"Error killing service on port {port}: {str(e)}")

    time.sleep(3)

    # 移除有问题的端口，避免重复选择
    if key in env_port_map_dict and port in env_port_map_dict[key]:
        env_port_map_dict[key].remove(port)
    
    # 选择可用 GPU（与初始启动时类似）
    gpus = GPUtil.getGPUs()
    if not gpus:
        logging.error("No GPUs available for reinitialization")
        return
    selected_gpu = max(gpus, key=lambda gpu: gpu.memoryFree)
    gpu_id = selected_gpu.id

    # 启动新的服务进程
    p = multiprocessing.Process(target=launch_server, args=(scene, task, gpu_id, port))
    p.start()

    # 将该端口重新加入共享字典
    if key not in env_port_map_dict:
        env_port_map_dict[key] = manager.list()
    env_port_map_dict[key].append(port)

# This function creates the environments sequentially.
def create_environments():
    port_counter = 4000
    processes = []
    for scene, task, count in selected_scenes_tasks:
        for i in range(count):
            port = port_counter
            port_counter += 1
            # Choose a GPU with the lowest load.
            gpus = GPUtil.getGPUs()
            if not gpus:
                raise Exception("No GPUs available")
            selected_gpu = max(gpus, key=lambda gpu: gpu.memoryFree)
            gpu_id = selected_gpu.id

            # Launch the environment server as a separate process.
            p = multiprocessing.Process(target=launch_server, args=(scene, task, gpu_id, port))
            p.start()
            processes.append(p)
            
            # Update the shared env_port_map_dict.
            key = (scene, task)
            if key not in env_port_map_dict:
                env_port_map_dict[key] = manager.list()
            env_port_map_dict[key].append(port)
            
            time.sleep(30)  # Delay to ensure each process starts properly.
    
    for p in processes:
        p.join()

# Gateway server to receive client requests and forward to the right environment.
gateway_app = FastAPI()

# Define the client request model.
class ClientRequest(BaseModel):
    llmplan: str
    scene: str
    task: str

@gateway_app.post("/process_request")
def process_request(request: ClientRequest):
    key = (request.scene, request.task)
    if key not in env_port_map_dict or len(env_port_map_dict[key]) == 0:
        raise HTTPException(status_code=400, detail="No environment available for this scene and task")
    
    # Randomly choose one port among those available.
    port = random.choice(list(env_port_map_dict[key]))
    url = f"http://127.0.0.1:{port}/get_reward"
    payload = {"llm_plans": request.llmplan}
    
    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()
        # 调用成功后，重置该端口的失败计数
        failure_counts[port] = 0
        return response.json()
    except requests.RequestException as e:
        # 请求失败时，更新该端口的连续失败计数
        failure_counts[port] = failure_counts.get(port, 0) + 1
        if failure_counts[port] >= 100:
            # 达到连续10次失败，触发重新初始化该端口的服务
            reinitialize_service(port, request.scene, request.task)
            failure_counts[port] = 0  # 重新初始化后重置计数
        raise HTTPException(status_code=500, detail=f"Error forwarding request to environment server on port {port}: {str(e)}")

if __name__ == "__main__":
    # Start the environment creation in a separate process so the gateway can run concurrently.
    env_creation_process = multiprocessing.Process(target=create_environments)
    env_creation_process.start()
    
    # Start the gateway server on port 8000.
    uvicorn.run(gateway_app, host="0.0.0.0", port=8003)
    
    env_creation_process.join()
