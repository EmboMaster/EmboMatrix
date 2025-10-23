import os
import json
import re
import yaml
import sys
import torch
import ast
import argparse
import logging
import traceback
import math
from pathlib import Path
from tqdm import tqdm
import omnigibson as og
from planner_api import planner
from omnigibson.action_primitives.starter_semantic_action_primitives import StarterSemanticActionPrimitives
from omnigibson.object_states import Inside, OnTop
from omnigibson import object_states
import hashlib
import csv
from config.config_loader import config
log_file = "collect_log.txt"
logging.basicConfig(filename=log_file, level=logging.ERROR, format='%(asctime)s - %(levelname)s - %(message)s')
output_dir = config['verification']['llmplan_path']
posecache_dir = config['verification']['posecache_path']
tasks_dir = config['task_generation']['output_dir']
feasible_file_path = config['verification']['feasible_file_path']
available_gpus = config['scene_generation']['available_gpus']
def extract_task(s):
    match = re.search(r'task_(.*?)_0', s)
    if match:
        return match.group(1)
    return None

def ensure_json_serializable(obj, path="root"):
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    if isinstance(obj, torch.Tensor):
        return obj.tolist()
    if isinstance(obj, (list, tuple)):
        return [ensure_json_serializable(item, path=f"{path}[{idx}]") for idx, item in enumerate(obj)]
    if isinstance(obj, dict):
        new_dict = {}
        for k, v in obj.items():
            if not isinstance(k, str):
                raise TypeError(f"Key '{k}' at '{path}' is not a string, not JSON-serializable.")
            new_dict[k] = ensure_json_serializable(v, path=f"{path}['{k}']")
        return new_dict
    raise TypeError(f"Object of type {type(obj)} at '{path}' is not JSON serializable.")

def safe_dump_to_json(cache_dict, scene_name, activity_name, if_replace=False, check=False):

    filename = f"omnigibson/plannerdemo/simulation_tools/posecache/{scene_name}_{activity_name}.json"

    cache_dict_serializable = ensure_json_serializable(cache_dict)
    parent_dir = os.path.dirname(filename)
    os.makedirs(parent_dir, exist_ok=True)
    short_plan_file = f"omnigibson/plannerdemo/simulation_tools/posecache/{scene_name}_{activity_name[:min(30,len(activity_name))]}.json"

    if check:
        if os.path.exists(short_plan_file):
            return short_plan_file
        else:
            return False

    with open(short_plan_file, 'w') as f:
        json.dump(cache_dict_serializable, f, indent=4)
    print(f"Plan saved successfully to {short_plan_file}")
    return short_plan_file

def get_object_pose(file_path, object_name):
    path_parts = file_path.split('/')
    scene_name = path_parts[-2]
    activity_name = path_parts[-1].replace('.json', '')
    sample_pose_dir = os.path.join(os.path.dirname(os.path.dirname(file_path)), "sample_pose", scene_name, activity_name)
    sample_pose_file = os.path.join(sample_pose_dir, "sample_pose.txt")
    if not os.path.isfile(sample_pose_file):
        print(f"Error: sample_pose file not found at {sample_pose_file}")
        return None
    try:
        data = {}
        with open(sample_pose_file, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                if ': ' in line:
                    key, value = line.split(': ', 1)
                    try:
                        data[key] = ast.literal_eval(value)
                    except (ValueError, SyntaxError) as e:
                        print(f"Error: Failed to parse value for '{key}' in {sample_pose_file} - {str(e)}")
                        continue
        if object_name in data and data[object_name] is not None:
            return data[object_name]
        else:
            print(f"Object '{object_name}' not found or value is None in {sample_pose_file}")
            return None
    except FileNotFoundError:
        print(f"Error: File {sample_pose_file} not found")
        return None
    except Exception as e:
        print(f"Error: Unable to read {sample_pose_file} - {str(e)}")
        return None

def get_task_object_names(json_path):
    if not os.path.isfile(json_path):
        raise FileNotFoundError(f"JSON file not found at: {json_path}")
    with open(json_path, 'r') as f:
        data = json.load(f)
    try:
        inst_to_name = data["metadata"]["task"]["inst_to_name"]
    except KeyError as e:
        raise KeyError(f"Invalid JSON structure, missing key: {e}")
    exclude_keys = ["floors_", "robot"]
    object_names = []
    for key, value in inst_to_name.items():
        if not any(tmp_key in value for tmp_key in exclude_keys):
            object_names.append(value)
    return object_names

def read_bddl(file_path):
    try:
        with open(file_path, 'r') as file:
            content = file.read()
        return content
    except FileNotFoundError:
        raise FileNotFoundError(f"文件未找到: {file_path}")
    except Exception as e:
        raise RuntimeError(f"读取文件时发生错误: {e}")

def replace_objects_in_bddl(bddl_content, json_path):
    try:
        with open(json_path, 'r') as json_file:
            json_data = json.load(json_file)
            inst_to_name = json_data["metadata"]["task"]["inst_to_name"]
    except FileNotFoundError:
        raise FileNotFoundError(f"JSON 文件未找到: {json_path}")
    except KeyError:
        raise KeyError("JSON 文件中缺少 'metadata.task.inst_to_name' 键")
    except Exception as e:
        raise RuntimeError(f"读取 JSON 文件时发生错误: {e}")
    for key, value in inst_to_name.items():
        bddl_content = bddl_content.replace(key, value)
    return bddl_content

def process_goal_section(bddl_content):
    goal_start = bddl_content.find("(:goal")
    if goal_start == -1:
        raise ValueError("未找到 goal 部分")
    goal_content = bddl_content[goal_start:]
    lines = goal_content.splitlines()
    actions = [line.strip() for line in lines if line.strip().startswith("(") and line.strip().endswith(")")]
    processed_goals = [action.replace("?", "").strip("()").split() for action in actions]
    return processed_goals

def triples_to_tree(triples, objects):
    """
    Convert a list of triples into a nested tree dictionary.
    Starts with root categories (e.g., floors, ceiling, wall) and iteratively adds connected objects.

    Args:
        triples (list of tuples): List of triples (obj1, relation, obj2).

    Returns:
        dict: A nested tree structure representing the triples.
    """

    tree = {}

    obj1_list = [triple[0] for triple in triples]
    obj2_list = [triple[2] for triple in triples]
    need_to_add = set()
    for obj in obj1_list:
        if obj not in obj2_list:
            tree[obj] = {}
            need_to_add.add(obj)

    def add_to_tree(root, current_tree, triples, need_to_add):
        """
        Add triples to the tree under the given root object.

        Args:
            root (str): The root object to which the relationships should be added.
            current_tree (dict): The current tree structure.
            triples (list of tuples): List of triples (obj1, relation, obj2).

        Returns:
            None: The current_tree is modified in place.
        """
        def find_subtree(tree, target):
            """Recursively find the subtree where the target root exists."""
            if target in tree:
                return tree[target]
            for key, value in tree.items():
                if isinstance(value, dict):
                    result = find_subtree(value, target)
                    if result is not None:
                        return result
            return None

        # Locate the subtree where the root exists
        subtree = find_subtree(current_tree, root)

        if subtree is None:
            raise ValueError(f"Root '{root}' not found in the current tree.")

        # Add triples to the located subtree
        for obj1, relation, obj2 in triples:
            if obj1 == root:
                if relation not in subtree:
                    subtree[relation] = {}
                if obj2 not in subtree[relation]:
                    subtree[relation][obj2] = {}
                    need_to_add.add(obj2)

        need_to_add.remove(root)

    while len(list(need_to_add)) != 0:
        for root in list(need_to_add):
            add_to_tree(root, tree, triples, need_to_add)

    for obj in objects:
        flag = False
        for triple in triples:
            if obj == triple[0] or obj == triple[2]:
                flag = True
                break
        if not flag:
            tree[obj] = {}

    return tree
def tree_to_list(tree):

    result = []
    
    def traverse(node):
        for parent, relations in node.items():
            for relation, children in relations.items():
                for child, subtree in children.items():
                    # 递归处理子节点
                    traverse({child: subtree})
                    # 添加当前三元组：父节点 -> 关系 -> 子节点
                    result.append([parent, relation, child])
    
    traverse(tree)
    return result

def get_sample_sequence(goals):
    '''
        input: initial goals
        output: goals in the correct sampling order
    '''
    related_goals = [goal for goal in goals if goal[0] in ('inside', 'ontop')]
    unrelated_goals = [goal for goal in goals if goal[0] not in ('inside', 'ontop')]
    related_triples = [(goal[1], goal[0], goal[2]) for goal in related_goals]
    objects = [goal[1] for goal in related_goals] + [goal[2] for goal in related_goals]
    tree = triples_to_tree(related_triples, objects)
    lis = tree_to_list(tree)
    final_goals = [[goal[1], goal[0], goal[2]] for goal in lis] + unrelated_goals
    return final_goals

def get_heat_source_data(category_prefix, csv_file_path='omnigibson/plannerdemo/data/heatSource.csv'):
    """
    Retrieves the row data for a given category prefix from the heatSource.csv file.

    Args:
        category_prefix (str): The prefix of the category to search for, e.g., "microwave".
        csv_file_path (str): The path to the heatSource.csv file.

    Returns:
        dict: A dictionary with keys as column names and values as the corresponding row values.
    """
    # Open and read the CSV file
    with open(csv_file_path, mode='r') as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            # Match the category prefix with the synset column
            if row['synset'].startswith(category_prefix):
                # Return the row as a dictionary
                return {
                    'synset': row['synset'],
                    'requires_toggled_on': int(row['requires_toggled_on']),
                    'requires_closed': int(row['requires_closed']),
                    'requires_inside': int(row['requires_inside']),
                    'temperature': float(row['temperature']),
                    'heating_rate': float(row['heating_rate'])
                }

    # Return None if no match is found
    return None

def get_cold_source_data(category_prefix, csv_file_path='omnigibson/plannerdemo/data/coldSource.csv'):
    """
    Retrieves the row data for a given category from the coldSource.csv file.

    Args:
        category (str): The category to search for, e.g., "microwave.n.02".
        csv_file_path (str): The path to the coldSource.csv file.

    Returns:
        dict: A dictionary with keys as column names and values as the corresponding row values.
    """
    # Open and read the CSV file
    with open(csv_file_path, mode='r') as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            # Match the category with the synset column
            if row['synset'].startswith(category_prefix):
                # Return the row as a dictionary
                return {
                    'synset': row['synset'],
                    'requires_toggled_on': int(row['requires_toggled_on']),
                    'requires_closed': int(row['requires_closed']),
                    'requires_inside': int(row['requires_inside']),
                    'temperature': float(row['temperature']),
                    'heating_rate': float(row['heating_rate'])
                }

    # Return None if no match is found
    return None   

def run_simulation(scene_file, activity_name, task_description, output_dir, gpu_id):
    scene_name = scene_file.split('/')[-2]
    if scene_name == activity_name:
        scene_name = scene_file.split('/')[-3]
    bddl_file = scene_file.replace('.json', '.bddl')

    try:
        # Load configuration
        with open("omnigibson/configs/fetch_discrete_behavior_planner.yaml", "r") as f:
            cfg = yaml.load(f, Loader=yaml.FullLoader)
        cfg['scene'].update({
            "scene_file": scene_file,
            "not_load_object_categories": ["door", "blanket", "carpet", "bath_rug", "mat", "place_mat", "yoga_mat"],
            "waypoint_resolution": 0.1,
            "trav_map_resolution": 0.05,
        })
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

        # Initialize environment
        og.macros.gm.GPU_ID = gpu_id
        og.macros.gm.USE_GPU_DYNAMICS = True
        og.macros.gm.ENABLE_FLATCACHE = False
        og.macros.gm.ENABLE_OBJECT_STATES = True
        og.macros.gm.ENABLE_TRANSITION_RULES = False
        env = og.Environment(configs=cfg)
        env.reset()

        # Step 1: Sample poses and verify task goals
        cache_dict = {}
        saved_path = safe_dump_to_json(cache_dict, scene_name, activity_name, check = True)
        if not saved_path:

            cache_dict = {"nextto": {}, "ontop": {}, "inside": {}}
            bddl_content = read_bddl(bddl_file)
            bddl_content = replace_objects_in_bddl(bddl_content, scene_file)
            goals = process_goal_section(bddl_content)
            task_objects = get_task_object_names(scene_file)
            real_task_objects = []

            for object_name in task_objects:
                query_pose = get_object_pose(scene_file, object_name)
                if query_pose is not None:
                    print(f"Found pose for {object_name}: {query_pose}")
                    cache_dict["nextto"][object_name] = query_pose
                else:
                    real_task_objects.append(object_name)

            _primitive_controller = StarterSemanticActionPrimitives(env, enable_head_tracking=False)
            for object_name in real_task_objects:
                object = env.scene.object_registry("name", object_name)
                sampled_pose_2d = None
                print("Try to sample pose on other object")
                for obj in env.scene.objects:
                    if (object.states[Inside].get_value(obj) and obj.name != object.name) or (object.states[OnTop].get_value(obj) and "floors_" not in obj.name):

                        if cache_dict["nextto"].get(obj.name) is None:
                            if get_object_pose(scene_file, obj.name) is None:
                                query_pose = None
                            else:
                                query_pose = get_object_pose(scene_file, obj.name)
                        else:
                            query_pose = cache_dict["nextto"].get(obj.name)
                        
                        if query_pose is not None:
                            print(f"Found pose for {obj.name}: {query_pose}")
                            sampled_pose_2d = query_pose
                            break
                        else:
                            for _ in range(2):
                                try:
                                    sampled_pose_2d = _primitive_controller._sample_pose_near_object(obj, pose_on_obj=None, distance_lo=0.1, distance_hi=1.5, yaw_lo=-math.pi, yaw_hi=math.pi)
                                    print(f"{object.name}: {sampled_pose_2d}")
                                    break
                                except:
                                    pass
                            if sampled_pose_2d is not None:
                                break
                if sampled_pose_2d is not None:
                    query_pose = sampled_pose_2d
                    print(f"Found pose for {object.name}: {query_pose}")
                    cache_dict["nextto"][object.name] = query_pose
                    continue

                if 'floors' not in object.name and 'window' not in object.name and 'wall' not in object.name and 'ceilings' not in object.name:
                    cache_dict["nextto"][object.name] = []
                    sampled_pose_2d = None
                    for _ in range(2):
                        try:
                            sampled_pose_2d = _primitive_controller._sample_pose_near_object(object, pose_on_obj=None, distance_lo=0.1, distance_hi=1.5, yaw_lo=-math.pi, yaw_hi=math.pi)
                            print(f"{object.name}: {sampled_pose_2d}")
                            break
                        except:
                            pass
                    if sampled_pose_2d is None:
                        print(f"Failed to sample pose for {object.name}! Can not complete the task!")
                        sys.exit(1)
                    else:
                        cache_dict["nextto"][object.name] = sampled_pose_2d
            goals = get_sample_sequence(goals)
            for goal in goals:
                if goal[0] == "ontop":
                    object1 = env.scene.object_registry("name", goal[1])
                    object2 = env.scene.object_registry("name", goal[2])
                    object1.states[object_states.OnTop].set_value(object2, True)
                    if object1.states[object_states.OnTop].get_value(object2):
                        query_pose = object1.get_position_orientation()
                        if query_pose is not None:
                            print(f"Found pose for {object1.name}: {query_pose}")
                            cache_dict["ontop"].setdefault(object1.name, {})[object2.name] = query_pose
                            continue
                    else:
                        print(f"Failed to sample pose for {object1.name} ontop! Can not complete the task!")
                        sys.exit(1)
                elif goal[0] == "inside":
                    object1 = env.scene.object_registry("name", goal[1])
                    object2 = env.scene.object_registry("name", goal[2])
                    object1.states[object_states.Inside].set_value(object2, True)
                    if object1.states[object_states.Inside].get_value(object2):
                        query_pose = object1.get_position_orientation()
                        if query_pose is not None:
                            print(f"Found pose for {object1.name}: {query_pose}")
                            cache_dict["inside"].setdefault(object1.name, {})[object2.name] = query_pose
                            continue
                    else:
                        print(f"Failed to sample pose for {object1.name} inside! Can not complete the task!")
                        sys.exit(1)

                #处理cook/frozen/hot的情况
                #如果goal是cook或者hot，说明需要加热
                elif goal[0] == "cooked" or goal[0] == "hot":
                    object1 = env.scene.object_registry("name", goal[1])
                    #循环所有的物体列表
                    for object_name in task_objects:
                        #获得物体的类别，之后与热源进行匹配
                        object2_category = env.scene.object_registry("name", object_name).category
                        print(f'object2_category = {object2_category}')
                        heat_source_info = get_heat_source_data(object2_category)
                        if heat_source_info is None:
                            #如果不是加热源就跳过这个物体
                            continue
                        #如果是加热源，就按照需求采样ontop和inside
                        object2 = env.scene.object_registry("name", object_name)
                        #首先采样inside的位置   
                        if heat_source_info["requires_inside"] == 1:
                            object1.states[object_states.Inside].set_value(object2, True)
                            if object1.states[object_states.Inside].get_value(object2):
                                #如果成功放置，记录被放置物体的位置
                                query_pose = object1.get_position_orientation()
                                #保存到缓存
                            else:
                                query_pose = None
                            if query_pose is not None:
                                print(f"Found pose for {object1.name}: {query_pose}")
                                cache_dict["inside"].setdefault(object1.name, {})[object2.name] = query_pose
                                continue
                            else:
                                print(f"Failed to sample pose for {object1.name} inside! Can not complete the task !")
                                sys.exit(1)
                        #如果没有要求inside，就采样ontop的位置
                        else:
                            object1.states[object_states.OnTop].set_value(object2, True)
                            if object1.states[object_states.OnTop].get_value(object2):
                                #如果成功放置，记录被放置物体的位置
                                query_pose = object1.get_position_orientation()
                                #保存到缓存
                            else:
                                query_pose = None
                            if query_pose is not None:
                                print(f"Found pose for {object1.name}: {query_pose}")
                                cache_dict["ontop"].setdefault(object1.name, {})[object2.name] = query_pose
                                continue
                            else:
                                print(f"Failed to sample pose for {object1.name} ontop! Can not complete the task !")
                                sys.exit(1)
                            
                elif goal[0] == "frozen":
                    object1 = env.scene.object_registry("name", goal[1])
                    #循环所有的物体列表
                    for object_name in task_objects:
                        #获得物体的类别，之后与冷源进行匹配
                        object2_category = env.scene.object_registry("name", object_name).category
                        cold_source_info = get_cold_source_data(object2_category)
                        if cold_source_info is None:
                            #如果不是冷源就跳过这个物体
                            continue
                        #如果是冷源，就按照需求采样inside
                        object2 = env.scene.object_registry("name", object_name)
                        #首先采样inside的位置   
                        if cold_source_info["requires_inside"] == 1:
                            object1.states[object_states.Inside].set_value(object2, True)
                            if object1.states[object_states.Inside].get_value(object2):
                                #如果成功放置，记录被放置物体的位置
                                query_pose = object1.get_position_orientation()
                                #保存到缓存
                            else:
                                query_pose = None
                            if query_pose is not None:
                                print(f"Found pose for {object1.name}: {query_pose}")
                                cache_dict["inside"].setdefault(object1.name, {})[object2.name] = query_pose
                                continue
                            else:
                                print(f"Failed to sample pose for {object1.name} inside! Can not complete the task !")
                                sys.exit(1)
                #如果goal是open或者close，说明需要打开或者关闭

            saved_path = safe_dump_to_json(cache_dict, scene_name, activity_name)
            print("成功写入 JSON 文件!")

        # Step 2: Generate and save LLM plan prompts
        sample_2d_cache = json.load(open(saved_path, 'r'))
        planner_pipeline = planner(env, cfg)
        object_dict = {}
        for index, key in enumerate(sample_2d_cache["nextto"].keys()):
            object_dict.update({index: key})

        plan_prompt = planner_pipeline.get_plan_prompt("test", task_description, object_dict)
        os.makedirs(output_dir, exist_ok=True)
        plan_file = os.path.join(output_dir, f"{activity_name}_plan_prompt.json")
        # try:
        #     with open(plan_file, 'w') as f:
        #         json.dump({"plan_prompt": plan_prompt}, f, indent=4)
        #     print(f"Saved plan prompt for {activity_name} to {plan_file}")
        # except:
            # dir_path, long_filename = os.path.split(plan_file)
            # short_filename = f"plan_{hashlib.md5(long_filename.encode()).hexdigest()[:8]}.json"
            # short_plan_file = os.path.join(dir_path, short_filename)
        short_plan_file = os.path.join(output_dir, f"{activity_name[:min(30,len(activity_name))]}_plan_prompt.json")
        with open(short_plan_file, 'w') as f:
            json.dump({"plan_prompt": plan_prompt}, f, indent=4)
        print(f"Plan saved successfully to {short_plan_file}")

    except Exception as e:
        logging.error(f"Error processing {scene_file}: {str(e)}")
        logging.error(f"Traceback: {traceback.format_exc()}")
        print(f"Error processing {scene_file}: {str(e)}")
    

def parse_args():
    parser = argparse.ArgumentParser(description="Run simulation with pose sampling and plan prompt generation")
    parser.add_argument("scene_file", type=str, help="Path to the scene JSON file")
    parser.add_argument("activity_name", type=str, help="The name of the activity")
    parser.add_argument("task_description", type=str, help="A brief description of the task")
    parser.add_argument("output_dir", type=str, help="The directory where the plan prompts will be saved")
    parser.add_argument("gpu_id", type=int, choices=available_gpus, help="The GPU ID to use")
    return parser.parse_args()

if __name__ == "__main__":
    scene_file = "/GPFS@rhome/yuanzhuoding/planner-bench/planner_bench/omnigibson/shengyin/results0424/hall_train_station/CoolerBurritoTask.json"
    scene = scene_file.split('/')[-2]
    task_description = scene_file.split('/')[-1].replace('.json', '').replace('_', ' ')
    activity_name = scene_file.split('/')[-1].replace('.json', '')
    output_dir = f"/GPFS@rhome/yuanzhuoding/planner-bench/planner_bench/omnigibson/data/og_dataset/scenes_with_newfetch/{scene}/llmplans"
    gpu_id = 6
    run_simulation(scene_file, activity_name, task_description, output_dir, gpu_id)