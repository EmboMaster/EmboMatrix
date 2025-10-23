import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
import yaml
import sys
sys.path.append("/GPFS/rhome/yuanzhuoding/planner-bench/planner_bench")
import omnigibson as og
from omnigibson.plannerdemo.planner_api import planner  # 假设你的 planner 类在 planner.py 文件中
import json
import asyncio

# 初始化 FastAPI 应用
app = FastAPI()

# 定义数据模型，用于请求数据
class LLMPlans(BaseModel):
    llm_plans: str

import os

def find_scene_task_json_file(root_path, scene, task):
    """
    Find a JSON file under root_path or its subdirectories with the pattern {scene}/{task}.json.

    Args:
        root_path (str): The root directory to start the search.
        scene (str): The scene name (e.g., 'restaurant_diner').
        task (str): The task name (e.g., 'heat_cookie_stove_shelf_bar').

    Returns:
        str or None: The full path to the matching file, or None if no match is found.
    """
    # Construct the target file name and scene directory
    target_filename = f"{task}.json"
    target_scene_dir = scene
    
    # Walk through the directory tree
    for dirpath, _, filenames in os.walk(root_path):
        # Check if the current directory name matches the scene
        if os.path.basename(dirpath) == target_scene_dir and target_filename in filenames:
            # Ensure the file is directly under the scene directory
            full_path = os.path.join(dirpath, target_filename)
            return full_path
    
    return None


# 加载配置文件并初始化环境和规划器
def initialize_env_and_planner(scene, task, cuda_device_index: int):
    # 设置CUDA设备
    # torch.cuda.set_device(cuda_device_index)
    # 读取配置文件并初始化环境
    og.macros.gm.USE_GPU_DYNAMICS=True
    og.macros.gm.ENABLE_FLATCACHE=False
    og.macros.gm.GPU_ID = cuda_device_index

    with open("omnigibson/configs/fetch_discrete_behavior_planner.yaml", "r") as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)

    # scene_file = "/data/zxlei/embodied/planner_bench/omnigibson/data/og_dataset/scenes/Pomaria_0_int/json/Pomaria_0_int_task_recycling_office_papers_0_0_template.json"
    
    #scene_file = f"/data/zxlei/embodied/planner_bench/omnigibson/data/og_dataset/scenes_with_newfetch/{scene}_task_{task}_0_0_template.json"

    #activity_name = scene_file.strip('.json').split('task_')[-1].split('_0')[0]
    scene_file = f"omnigibson/data/og_dataset/scenes_with_newfetch/{scene}_{task}.json" 
    try:
        f = open(scene_file, "r")
    except:
        #scene_file = f"omnigibson/result0304/{scene}/{task}.json" 
        scene_file = find_scene_task_json_file("/GPFS/rhome/yuanzhuoding/planner-bench/planner_bench/omnigibson/shengyin", scene, task)
        try:
            f = open(scene_file, "r")
        except:
            # scene_file = f"/data/zxlei/embodied/planner_bench/omnigibson/data/og_dataset/scenes_with_newfetch/0323/avail_task_scene/json_newrobots/{scene}_{task}_0_0_template.json"
            scene_file = f"omnigibson/data/og_dataset/scenes_with_newfetch/0323/avail_task_scene/json_newrobots/{scene}_{task}_0_0_template.json"
    activity_name = task
    cfg['scene'].update({
        "scene_file": scene_file,
        "not_load_object_categories": ["door", "blanket", "carpet", "bath_rug", "mat", "place_mat", "yoga_mat"],
        "waypoint_resolution": 0.1,
        "trav_map_resolution": 0.05,
    })

    cfg['task'].update({
        "activity_name": activity_name,
    })

    cfg['env'].update({
        "action_frequency": 120,
        "rendering_frequency": 120,
        "flatten_action_space": True,
        "flatten_obs_space": True,
    })

    # 初始化 OmniGibson 环境

    env = og.Environment(configs=cfg)
    env.reset()

    # 初始化规划器
    planner_pipeline = planner(env, cfg)
    return planner_pipeline

# 创建一个用于处理请求的 FastAPI 端点
@app.post("/get_reward")
async def get_reward(llm_plans: LLMPlans):
    try:
        # 从请求中获取 llm_plans
        # print(llm_plans.llm_plans)
        # llm_plans_data = json.loads(llm_plans.llm_plans)
        llm_plans_data = llm_plans.llm_plans
        # 调用 get_reward_dict 函数
        try:
            reward_dict = planner_pipeline.get_reward_dict(llm_plans_data)
        except Exception as e:
            print(e)
            raise HTTPException(status_code=500, detail=str(e))

        # 返回 reward_dict
        return reward_dict
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def start_server(scene, task, cuda_device_index: int, port: int):
    global planner_pipeline

    # 初始化环境和规划器
    planner_pipeline = initialize_env_and_planner(scene, task, cuda_device_index)
    
    import asyncio
    import nest_asyncio

    # 如果 OmniKit 弄了猴子补丁，恢复 asyncio 的原始 run 方法
    if hasattr(asyncio, "_orig_run"):
        asyncio.run = asyncio._orig_run

    # 设置默认事件循环策略并应用 nest_asyncio 补丁
    asyncio.set_event_loop_policy(asyncio.DefaultEventLoopPolicy())
    nest_asyncio.apply()

    # 显式创建一个新的事件循环，并设置为当前循环
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    # 启动 FastAPI 服务器
    uvicorn.run(app, host="127.0.0.1", port=port)