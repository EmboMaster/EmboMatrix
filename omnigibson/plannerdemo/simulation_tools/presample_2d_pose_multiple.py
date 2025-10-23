#======================= 实验脚本 experiment_script.py =======================
import json
import os
import time
import torch
import argparse
import yaml
import omnigibson as og
import re
import math
from omnigibson.action_primitives.starter_semantic_action_primitives import StarterSemanticActionPrimitives
from tqdm import tqdm

def extract_task(s):
    # Use regular expression to find the content between 'task_' and '_0'
    match = re.search(r'task_(.*?)_0', s)
    if match:
        return match.group(1)
    return None

def run_experiment(scene_file, activity_name, task_description, gpu_id):
    # bddl_file
    bddl_file = f"bddl/bddl/activity_definitions/{activity_name}/problem0.bddl"
    
    # 加载配置文件
    with open("omnigibson/configs/fetch_discrete_behavior_planner.yaml", "r") as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)

    # 动态配置参数
    cfg['scene'].update({
        "scene_file": scene_file,
        "not_load_object_categories": ["door", "blanket","carpet","bath_rug","mat","place_mat","yoga_mat"],
        "waypoint_resolution": 0.1,
        "trav_map_resolution": 0.05,
    })
    
    scene_name = scene_file.split('/')[-1].split('_task')[0]

    cfg['task'].update({
        "activity_name": activity_name,
        "problem_filename": bddl_file
    })

    cfg['env'].update({
        "action_frequency": 120,
        "rendering_frequency": 120,
        "flatten_action_space": True,
        "flatten_obs_space": True,
    })

    cfg['planner'].update({
        "task_description": task_description
    })

    # GPU配置
    # og.macros.gm.GPU_ID = 0

    
    env = og.Environment(configs=cfg)
    env.reset()

    cache_dict = {}
    index_name_dict = {}
    for k, v in env.scene._scene_info_meta_inst_to_name.items():
        index_name_dict[v] = k
        
    print(cache_dict)
    
    _primitive_controller = StarterSemanticActionPrimitives(env, enable_head_tracking=False)
    
    for object in tqdm(env.scene.objects):
        print(object.name)
        if 'floor' not in object.name and 'window' not in object.name and 'wall' not in object.name and 'ceilings' not in object.name:
            cache_dict[object.name] = object
            print(object.name)
            sampled_pose_2d = None
            for _ in range(2):
                try:
                    sampled_pose_2d = _primitive_controller._sample_pose_near_object(object, pose_on_obj=None, distance_lo=0.1, distance_hi=0.7, yaw_lo=-math.pi, yaw_hi=math.pi) 
                    print(f"{object.name}: {sampled_pose_2d}")
                    break
                except:
                    pass
            if sampled_pose_2d is None:
                print(f"Failed to sample pose for {object.name}")
                cache_dict[object.name] = sampled_pose_2d
                continue
            else:
                cache_dict[object.name] = sampled_pose_2d
        
    # json.dump(cache_dict, open(f"omnigibson/plannerdemo/simulation_tools/posecache/{scene_name}_{activity_name}.json", 'w'))
    converted_cache_dict = convert_to_serializable(cache_dict)
    json.dump(converted_cache_dict, open(f"omnigibson/plannerdemo/simulation_tools/posecache/{scene_name}_{activity_name}.json", 'w'))
    # 初始化环境
    try:
        og.sim.stop()
    except:
        pass
            
    
def presample_from_scene_task_txt(scene_task_txt_path, gpu_id):
    # 读取scene_task.txt文件
    with open(scene_task_txt_path, 'r') as f:
        lines = f.readlines()
    
    # 遍历每行，提取场景和任务信息
    for line in lines:
        # 假设格式为 ("scene", "task", count)
        match = re.match(r'\("(.+?)", "(.+?)", (\d+)\)', line.strip())
        if match:
            scene_name = match.group(1)
            task_name = match.group(2)
            
            # 拼接 scene 文件路径
            scene_file = f'/data/zxlei/embodied/planner_bench/omnigibson/data/og_dataset/scenes_with_newfetch/0323/avail_task_scene/json_newrobots/{scene_name}_{task_name}_0_0_template.json'
            task_description = task_name.replace('_', ' ')
            
            # 调用 run_experiment 函数执行预采样
            try:
                run_experiment(scene_file, task_name, task_description, gpu_id)
            except:
                try:
                    og.sim.stop()
                except:
                    pass

def convert_to_serializable(obj):
    if isinstance(obj, torch.Tensor):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]
    else:
        return obj

if __name__ == "__main__":
    # 指定 scene_task.txt 路径
    scene_task_txt_path = '/data/zxlei/embodied/planner_bench/omnigibson/data/og_dataset/scenes_with_newfetch/0323/avail_task_scene/scene_task_stats.txt'
    gpu_id = 0  # 设置 GPU ID，或者根据需要修改

    # 执行预采样
    presample_from_scene_task_txt(scene_task_txt_path, gpu_id)
