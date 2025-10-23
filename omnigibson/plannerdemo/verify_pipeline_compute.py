import argparse
import json
import os
import concurrent.futures
import itertools
from presample_plus_llm_prompt import run_simulation
from src.utils.config_loader import config
output_dir = config['verification']['llmplan_path']
posecache_dir = config['verification']['posecache_path']
tasks_dir = config['task_generation']['output_dir']
feasible_file_path = config['verification']['feasible_file_path']
other_num = 0

def run_task(scene_file, gpu_id):
    scene_name = scene_file.split('/')[-2]
    activity_name = scene_file.split('/')[-1].replace('.json', '')
    if scene_name == activity_name:
        scene_name = scene_file.split('/')[-3]
    task_description = activity_name.replace('_', ' ')
    llmplan_output_dir = os.path.join(output_dir, scene_name,"llmplans")
    os.makedirs(output_dir, exist_ok=True)
    run_simulation(scene_file, activity_name, task_description, output_dir, gpu_id)

def check_if_success(scene_file):
    scene_name = scene_file.split('/')[-2]
    activity_name = scene_file.split('/')[-1].replace('.json', '')
    if scene_name == activity_name:
        scene_name = scene_file.split('/')[-3]
    
    # Full path with complete activity_name
    posecache_file_full = os.path.join(posecache_dir, f"{scene_name}_{activity_name}.json")
    plan_prompt_file_full = os.path.join(output_dir, scene_name, "llmplans", f"{activity_name}_plan_prompt.json")
    posecache_file_short = os.path.join(posecache_dir, f"{scene_name}_{activity_name[:30]}.json")
    plan_prompt_file_short = os.path.join(output_dir, scene_name, "llmplans", f"{activity_name[:30]}_plan_prompt.json") 


    posecache_exists, plan_prompt_exists = False, False
    
    # If either full path is missing or empty, check shortened paths
    if not posecache_exists:
        posecache_exists = os.path.exists(posecache_file_short)
        
    if not plan_prompt_exists:
        plan_prompt_exists = os.path.exists(plan_prompt_file_short)
    origin_path_list = [os.path.join(tasks_dir, scene_name, activity_name),]

    for origin_path in origin_path_list:
        if os.path.exists(origin_path):
            break

    if posecache_exists and plan_prompt_exists:

        new_bddl_path = f'bddl/bddl/activity_definitions/{activity_name[:30]}'

        if not os.path.exists(new_bddl_path):
            os.makedirs(new_bddl_path)
        
        import shutil
        shutil.copytree(origin_path, new_bddl_path, dirs_exist_ok=True)
        print(f"已成功将 '{origin_path}' 中的所有内容复制到 '{new_bddl_path}'。")
    
    return posecache_exists and plan_prompt_exists

def get_feasible_scene(scene_path):
    json_path = feasible_file_path
    if not os.path.exists(json_path) or os.path.getsize(json_path) == 0:
        print(f"Error: JSON file {json_path} is empty or missing!")
        scene_dict = {}
    else:
        with open(json_path, "r") as f:
            try:
                scene_dict = json.load(f)
            except json.JSONDecodeError as e:
                print(f"Error parsing JSON: {e}")
                scene_dict = {}
    
    for root, dirs, files in os.walk(scene_path):
        for file in files:
            if file.endswith(".json") and "sample_pose" not in file and "plan_prompt" not in file:
                scene_file = os.path.join(root, file)
                scene_dict[scene_file] = {"feasible": False}
    
    # print(scene_dict)
    with open(json_path, "w") as f:
        json.dump(scene_dict, f, indent=4)

def update_feasible_scene(feasible_scene_path):
    with open(feasible_scene_path, "r") as f:
        file_to_update = json.load(f)
    for k, v in file_to_update.items():
        v['feasible'] = check_if_success(k)
    with open(feasible_scene_path, 'w') as f:
        json.dump(file_to_update, f, indent=4)

def caculate_total_num(feasible_scene_path):
    feasible_num = 0
    filtered_feasible_scene = {}
    with open(feasible_scene_path, "r") as f:
        file_to_update = json.load(f)
    for k, v in file_to_update.items():
        if v.get('feasible', False):
            feasible_num += 1
            filtered_feasible_scene[k] = {"feasible": True}
    with open(feasible_scene_path.replace('.json', '_filtered.json'), 'w') as f:
        json.dump(filtered_feasible_scene, f, indent=4)
    print(f"{feasible_num} scenes are feasible")

def main():
    available_gpus = [0, 1]
    with open(feasible_file_path, "r") as f:
        scene_to_check = json.load(f)
    tasks = []
    for scene_file, info in scene_to_check.items():
        if not info.get("feasible", False):
            tasks.append(scene_file)
    
    max_workers = len(available_gpus)
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_scene = {}
        gpu_cycle = itertools.cycle(available_gpus)
        for scene_file in tasks:
            gpu_id = next(gpu_cycle)
            print(f"Processing {scene_file} on GPU {gpu_id}")
            future = executor.submit(run_task, scene_file, gpu_id)
            future_to_scene[future] = (scene_file, gpu_id)
        for future in concurrent.futures.as_completed(future_to_scene):
            scene_file, assigned_gpu = future_to_scene[future]
            try:
                success = check_if_success(scene_file)
                scene_to_check[scene_file]["feasible"] = success
                print(f"Processing {scene_file} on GPU {assigned_gpu} finished, success: {success}")
            except Exception as e:
                print(f"Processing {scene_file} on GPU {assigned_gpu} failed: {e}")
                scene_to_check[scene_file]["feasible"] = False
    
    with open(feasible_file_path, "w") as f:
        json.dump(scene_to_check, f, indent=4)

if __name__ == "__main__":
    get_feasible_scene('omnigibson/shengyin/results-eai-0821')
    update_feasible_scene("omnigibson/feasible_scene/feasible_scene-eai-0821.json")
    caculate_total_num("omnigibson/feasible_scene/feasible_scene-eai-0821.json")