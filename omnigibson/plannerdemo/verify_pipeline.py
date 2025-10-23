import argparse
import json
import os
import datetime
import sys
import logging
import traceback
from presample_plus_llm_prompt_eai import run_simulation
from src.utils.config_loader import config
output_dir = config['verification']['llmplan_path']
posecache_dir = config['verification']['posecache_path']
tasks_dir = config['task_generation']['output_dir']
scene_dir = config['scene_generation']['output_dir']
feasible_file_path = config['verification']['feasible_file_path']
logs_path = config['verification']['logs_path']
class PrintToFile:
    def __init__(self, folder_path="src/logs_new", scene_name=None, task_name=None, gpu_id=0, step=None):
        if step:
            folder_path = os.path.join(folder_path, step)
        os.makedirs(folder_path, exist_ok=True)
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        if scene_name and task_name:
            bddl_name = task_name[:min(30, len(task_name))]
            self.file_path = os.path.join(folder_path, f"gpu{gpu_id}_{timestamp}_{scene_name}_{bddl_name}.txt")
        else:
            self.file_path = os.path.join(folder_path, f"log_{timestamp}.txt")
        self.original_stdout = sys.stdout

    def write(self, message):
        if message and message[0] == "{" and message[-1] == "}":
            try:
                tmp_message = eval(message)
            except:
                tmp_message = message
            if isinstance(tmp_message, dict):
                with open(self.file_path, 'a', encoding='utf-8') as file:
                    timestamp = datetime.datetime.now().strftime("[%Y-%m-%d %H:%M:%S] ")
                    file.write(timestamp + "\n")
                    json.dump(tmp_message, file, indent=4, ensure_ascii=False)
            else:
                with open(self.file_path, 'a', encoding='utf-8') as file:
                    timestamp = datetime.datetime.now().strftime("[%Y-%m-%d %H:%M:%S] ")
                    file.write(timestamp + message + "\n")
        else:
            with open(self.file_path, 'a', encoding='utf-8') as file:
                timestamp = datetime.datetime.now().strftime("[%Y-%m-%d %H:%M:%S] ")
                file.write(timestamp + message + "\n")
        self.original_stdout.write(str(message) + "\n")

    def flush(self):
        self.original_stdout.flush()

def redirect_print_to_file(log_path_rule="src/logs_new", scene_name=None, bddl_directory=None, gpu_id=0, step=None):
    sys.stdout = PrintToFile(folder_path=log_path_rule, scene_name=scene_name, task_name=bddl_directory, gpu_id=gpu_id, step=step)

log_file = 'omnigibson/plannerdemo/logs/error/problem0.log'
logging.basicConfig(
    filename=log_file,
    level=logging.ERROR,
    format='[%(asctime)s] [%(levelname)s] %(message)s'
)

def log_uncaught_exceptions(exc_type, exc_value, exc_traceback):
    if issubclass(exc_type, KeyboardInterrupt):
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return
    error_message = "".join(traceback.format_exception(exc_type, exc_value, exc_traceback))
    print(f"Uncaught exception: {error_message}")
    logging.error("Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback))
    logging.shutdown()

sys.excepthook = log_uncaught_exceptions

def run_task(scene_file, gpu_id,regenerate_flag=False):
    scene_name = scene_file.split('/')[-2]
    activity_name = scene_file.split('/')[-1].replace('.json', '')
    if scene_name == activity_name:
        scene_name = scene_file.split('/')[-3]
    task_description = activity_name.replace('_', ' ')
    llmplan_output_dir = os.path.join(output_dir, scene_name,"llmplans")
    os.makedirs(llmplan_output_dir, exist_ok=True)
    # breakpoint()
    run_simulation(scene_file, activity_name, task_description, llmplan_output_dir, gpu_id,regenerate_flag)

def check_if_success(scene_file):
    # breakpoint()
    scene_name = scene_file.split('/')[-2]
    activity_name = scene_file.split('/')[-1].replace('.json', '')
    if scene_name == activity_name:
        scene_name = scene_file.split('/')[-3]
    
    # Full path with complete activity_name
    posecache_file_full = os.path.join(posecache_dir, f"{scene_name}_{activity_name}.json")
    plan_prompt_file_full = os.path.join(output_dir, scene_name, "llmplans", f"{activity_name}_plan_prompt.json")
    posecache_file_short = os.path.join(posecache_dir, f"{scene_name}_{activity_name[:30]}.json")
    plan_prompt_file_short = os.path.join(output_dir, scene_name, "llmplans", f"{activity_name[:30]}_plan_prompt.json") 


    # Check full paths first
    posecache_exists = os.path.exists(posecache_file_full) and os.path.getsize(posecache_file_full) > 0
    plan_prompt_exists = os.path.exists(plan_prompt_file_full) and os.path.getsize(plan_prompt_file_full) > 0
    
    # If either full path is missing or empty, check shortened paths
    if not posecache_exists:
        posecache_exists = os.path.exists(posecache_file_short) and os.path.getsize(posecache_file_short) > 0
        
    if not plan_prompt_exists:
        plan_prompt_exists = os.path.exists(plan_prompt_file_short) and os.path.getsize(plan_prompt_file_short) > 0
    
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
def split_into_portions(data, num_portions=10):
    """将数据分成num_portions份"""
    keys = list(data.keys())
    portion_size = max(1, len(keys) // num_portions)
    portions = [keys[i:i + portion_size] for i in range(0, len(keys), portion_size)]
    # 如果份数不足10份，补齐空列表
    while len(portions) < num_portions:
        portions.append([])
    return portions

def main():
    parser = argparse.ArgumentParser(description="Run simulation and check feasibility")
    parser.add_argument("--scene_file", type=str, default="omnigibson/shengyin/results-eai/hotel_suite_small/Backpack_Bedroom0.json", help="Path to the scene JSON file")
    parser.add_argument("--gpu_id", type=int, default=6, help="The GPU ID to use")
    parser.add_argument("--mode", type=int, default=0, help="Mode: 0 for single scene, >0 for processing portion of feasible_scene_divide_05080350.json")
    parser.add_argument("--feasible_scene_file", type=str, default=feasible_file_path, help="Path to the feasible scene JSON file")
    parser.add_argument("--regenerate_flag", type=bool, default=False, help="Path to the feasible scene JSON file")
    args = parser.parse_args()
    mode = args.mode
    gpu_id = args.gpu_id if mode == 0 else 0  # mode > 0 时 gpu_id 固定为 0
    feasible_scene_file = args.feasible_scene_file

    # 原有逻辑：处理单个 scene_file
    regenerate_flag = args.regenerate_flag
    scene_file = args.scene_file
    scene_name = scene_file.split('/')[-2]
    activity_name = scene_file.split('/')[-1].replace('.json', '')
    if scene_name == activity_name:
        scene_name = scene_file.split('/')[-3]

    redirect_print_to_file(scene_name=scene_name, bddl_directory=activity_name, gpu_id=gpu_id, log_path_rule=logs_path)

    with open(feasible_scene_file, "r") as f:
        scene_to_check = json.load(f)


    info = scene_to_check.get(scene_file, {"feasible": False})

    if (regenerate_flag and info.get("feasible", False)) or (not info.get("feasible", False)):
        print(f"Processing {scene_file} on GPU {gpu_id}")
        run_task(scene_file, gpu_id, regenerate_flag)
        success = check_if_success(scene_file)
        scene_to_check[scene_file]["feasible"] = success
        print(f"Processing {scene_file} on GPU {gpu_id} finished, success: {success}")
    else:
        print(f"Simulation for {scene_file} already feasible")

    with open(feasible_scene_file, "w") as f:
        json.dump(scene_to_check, f, indent=4)

def run_pipeline():
    try:
        main()
    except Exception as e:
        logging.error("Caught exception during pipeline execution", exc_info=True)
        print(f"[ERROR] {e}")
        print(traceback.format_exc())
        logging.error(traceback.format_exc())

if __name__ == "__main__":
    run_pipeline()