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
# selected_scenes_tasks = [
#     ("Pomaria_0_int", "recycling_office_papers", 4),
#     ("Ihlen_0_int", "bring_laptop_to_desk", 4),
#     ("Beechwood_0_int", "take_laundry_basket", 4),
#     ("Beechwood_1_int", "bring_remote_control_to_coffee_table", 4),
#     ("restaurant_urban", "clean_up_after_a_dinner_party", 1),
#     ("restaurant_brunch", "set_a_dinner_table", 1),
#     ("house_single_floor", "delivering_groceries_to_doorstep", 2)
# ]

# restaurant_diner/heat_cookie_stove_shelf_bar
# /GPFS/rhome/yuanzhuoding/planner-bench/planner_bench/omnigibson/shengyin/results0421/office_vendor_machine/pickup_fries_cook_table.bddl
selected_scenes_tasks = [
    ("office_vendor_machine", "pickup_fries_cook_table", 1),
    # ("Rs_garden", "bring_the_paperback_book_from_", 1),
    # ("Merom_1_int", "bring_comic_book_sunglasses_an", 1),
    # ("Benevolence_0_int", "place_the_paperback_book_from_", 1),
    # ("Pomaria_0_garden", "fetch_the_security_camera_from", 1),
    # ("Benevolence_1_int", "bring_shopping_basket_from_boo", 1),
    # ("Rs_int", "place_the_bath_towel_on_the_st", 1),
    # ("Pomaria_1_int", "place_the_soccer_ball_from_the", 1),
    # ("Ihlen_1_int", "bring_the_wicker_basket_and_dinner_napkin_to_the_breakfast_table", 2),

]



# 定义日志文件路径和命名格式
log_directory = "models/planserverlog"

log_filename = f"{log_directory}/{datetime.now().strftime('%Y-%m-%d/%H-%M')}.log"

if not os.path.exists(os.path.dirname(log_filename)):
    os.makedirs(os.path.dirname(log_filename))
# 配置日志
logging.basicConfig(filename=log_filename, level=logging.INFO, format='%(asctime)s - %(message)s')

# 定义日志文件路径和命名格式
# log_directory = "/data/zxlei/embodied/planner_bench/models/planserverlog"
log_directory = "models/planserverlog"

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
    # Import and start the environment FastAPI server.q
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
    port_counter = 6000
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
            
            time.sleep(60)  # Delay to ensure each process starts properly.
    
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
    uvicorn.run(gateway_app, host="0.0.0.0", port=8000)
    
    env_creation_process.join()
