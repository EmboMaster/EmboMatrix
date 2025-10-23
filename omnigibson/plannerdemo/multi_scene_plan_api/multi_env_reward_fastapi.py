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

# Create a Manager for shared state.
manager = Manager()
# Shared dictionary: key=(scene, task) -> value=list of ports hosting environments.
env_port_map_dict = manager.dict()

# Define the list of environments to create:
selected_scenes_tasks = [
    ("Pomaria_0_int", "recycling_office_papers", 2),
    ("restaurant_urban", "clean_up_after_a_dinner_party", 1),
    # ("restaurant_brunch", "set_a_dinner_table", 1),
    # ("house_single_floor", "delivering_groceries_to_doorstep", 2)
]

# This function will launch an environment server in a new process.
def launch_server(scene, task, gpu_id, port):
    # Set the CUDA device for this process.
    # os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    print(f"Starting FastAPI server for scene '{scene}', task '{task}' on port {port} with GPU {gpu_id}")
    # Import and start the environment FastAPI server.
    # Note: env_reward_fastapi.py should be in the same directory.
    import env_reward_fastapi
    env_reward_fastapi.start_server(scene, task, gpu_id, port)

# This function creates the environments sequentially.
def create_environments():
    port_counter = 5000
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
            # If key does not exist, initialize with a managed list.
            if key not in env_port_map_dict:
                env_port_map_dict[key] = manager.list()
            env_port_map_dict[key].append(port)
            
            time.sleep(30)  # Delay to ensure each process starts properly.
    
    # Optionally, wait for all environment processes to complete.
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
    # The environment expects a payload with "llm_plans" as key.
    payload = {"llm_plans": request.llmplan}
    
    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()
    except requests.RequestException as e:
        raise HTTPException(status_code=500, detail=f"Error forwarding request to environment server on port {port}: {str(e)}")
    
    return response.json()

if __name__ == "__main__":
    # Start the environment creation in a separate process so the gateway can run concurrently.
    env_creation_process = multiprocessing.Process(target=create_environments)
    env_creation_process.start()
    
    # Start the gateway server on port 8000.
    uvicorn.run(gateway_app, host="0.0.0.0", port=8000)
    
    env_creation_process.join()
