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

def run_experiment():
    parser = argparse.ArgumentParser()
    # parser.add_argument("--scene_path", type=str, required=True)
    # parser.add_argument("--task_file", type=str, required=True) 
    parser.add_argument("--gpu_id", type=int, required=False, default=0)
    parser.add_argument("--scene_file", type=str, required=False, default='/data/zxlei/embodied/planner_bench/omnigibson/data/og_dataset/scenes/Pomaria_0_int/json/Pomaria_0_int_task_recycling_office_papers_0_0_template.json')
    parser.add_argument("--mode", type=str, required=False, default='train')
    # parser.add_argument(
    #     "--num_transformer_block",
    #     type=int,
    #     default=1,
    #     help="GPU ID for the network",
    # )
    args = parser.parse_args()



    # 加载配置文件
    scene_file = "/data/zxlei/embodied/planner_bench/omnigibson/result0304/Beechwood_0_garden/bring_blanket_to_armchair.json"
    task_description = extract_task(scene_file.split('/')[-1]).replace('_', ' ')
    activity_name = scene_file.strip('.json').split('task_')[-1]
    # find {_number} pattern and split this str 
    activity_name = activity_name.split('_0')[0]
    
    # bddl_file = scene_file.replace('.json', '.bddl')
    bddl_file = f"bddl/bddl/activity_definitions/{activity_name}/problem0.bddl"
    # bddl_file = f"bddl/bddl/bddl_tocheck0303/{activity_name}.bddl" 
    with open("omnigibson/configs/fetch_discrete_behavior_planner.yaml", "r") as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)

    # 动态配置参数
    cfg['scene'].update({
        # "scene_model": os.path.basename(args.scene_path),
        # "scene_instance": args.task_file,
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
    og.macros.gm.GPU_ID = 0
    device = f"cuda:{args.gpu_id+4}"

    # 初始化环境
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
            # while sampled_pose_2d is None:
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
        
    json.dump(cache_dict, open(f"omnigibson/plannerdemo/simulation_tools/posecache/{scene_name}_{activity_name}.json", 'w'))
    
            
    
        
        


if __name__ == "__main__":
    run_experiment()