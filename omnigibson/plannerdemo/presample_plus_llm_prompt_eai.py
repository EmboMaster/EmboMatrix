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
from planner_api_eai import planner
from omnigibson.action_primitives.starter_semantic_action_primitives import StarterSemanticActionPrimitives
from omnigibson.object_states import Inside, OnTop
from omnigibson import object_states
import hashlib
import csv
from config.config_loader import config
# Configure logging
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

    filename = os.path.join(posecache_dir, f"{scene_name}_{activity_name[:30]}.json")

    cache_dict_serializable = ensure_json_serializable(cache_dict)

    parent_dir = os.path.dirname(filename)
    os.makedirs(parent_dir, exist_ok=True)
    # try:
    #     with open(filename, 'w') as f:
    #         json.dump(cache_dict_serializable, f, indent=4)
    #     print(f"Plan saved successfully to {filename}")
    #     return filename
    # except:
    #     # dir_path, long_filename = os.path.split(filename)
    #     # short_filename = f"plan_{hashlib.md5(long_filename.encode()).hexdigest()[:8]}.json"
    #     # short_plan_file = os.path.join(dir_path, short_filename) 
    short_plan_file = os.path.join(posecache_dir, f"{scene_name}_{activity_name[:30]}.json")

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

# def process_goal_section(bddl_content):
#     goal_start = bddl_content.find("(:goal")
#     if goal_start == -1:
#         raise ValueError("未找到 goal 部分")
#     goal_content = bddl_content[goal_start:]
#     lines = goal_content.splitlines()
#     actions = [line.strip() for line in lines if line.strip().startswith("(") and line.strip().endswith(")")]
#     processed_goals = [action.replace("?", "").strip("()").split() for action in actions]
#     return processed_goals

def _parse_objects(bddl_content: str) -> dict[str, list[str]]:
    """
    一个辅助函数，用于解析 :objects 部分，并返回一个类型到对象实例列表的映射。
    """
    objects_map = {}
    try:
        objects_section = re.search(r'\(:objects(.*?)\)', bddl_content, re.DOTALL).group(1)
    except AttributeError:
        raise ValueError("在 BDDL 内容中未找到 :objects 部分")

    lines = objects_section.strip().splitlines()
    for line in lines:
        line = line.strip()
        if not line or line.startswith(';'):
            continue
        
        parts = line.split('-')
        if len(parts) != 2:
            continue

        object_names = parts[0].strip().split()
        object_type = parts[1].strip()

        if object_type not in objects_map:
            objects_map[object_type] = []
        objects_map[object_type].extend(object_names)
        
    return objects_map

def process_goal_section(bddl_content: str) -> list[list[str]]:
    """
    处理 BDDL 内容中的 :goal 部分。

    - 如果 goal 中包含 '(forall ...)'，它会根据 :objects 部分扩展目标，并包含其他独立目标。
    - 如果 goal 中不包含 '(forall ...)'，它会像以前一样解析简单的目标。
    """
    goal_start_index = bddl_content.find("(:goal")
    if goal_start_index == -1:
        raise ValueError("未找到 goal 部分")

    open_parens = 0
    goal_content = ""
    for i in range(goal_start_index, len(bddl_content)):
        char = bddl_content[i]
        if char == '(':
            open_parens += 1
        elif char == ')':
            open_parens -= 1
        
        if open_parens == 0:
            goal_content = bddl_content[goal_start_index : i+1]
            break
    
    if "(forall" in goal_content:
        # --- `forall` 逻辑 (已更新) ---
        objects_map = _parse_objects(bddl_content)
        match = re.search(r'\(forall\s+\(\s*\?(\S+)\s+-\s+([\w\._-]+)\)', goal_content)
        if not match:
            raise ValueError("在 goal 中找到 'forall' 但无法解析其变量和类型")

        variable_name_no_q = match.group(1)
        variable_name = "?" + variable_name_no_q
        variable_type = match.group(2)
        
        if variable_type not in objects_map:
            raise ValueError(f"类型 '{variable_type}' 在 :objects 部分中未定义")
        concrete_objects = objects_map[variable_type]

        template_strings = re.findall(r'\(([^()]+)\)', goal_content)
        
        final_goals = []
        for template_str in template_strings:
            template_str = template_str.strip()
            
            # 过滤掉 forall 自身的变量定义行
            if variable_name_no_q in template_str and '-' in template_str:
                continue
            
            # *** 这是关键的修改 ***
            # 检查模板是否包含 forall 变量以进行扩展
            if variable_name in template_str:
                parts = template_str.split()
                # 对于每一个具体对象，生成一个新的 goal
                for obj in concrete_objects:
                    # 将模板中的变量替换为具体的对象名
                    new_goal = [obj if item == variable_name else item for item in parts]
                    # 移除所有项中可能存在的 '?'
                    final_goals.append([item.replace("?", "") for item in new_goal])
            else:
                # 如果不包含 forall 变量，则这是一个独立的 goal
                parts = template_str.split()
                # 同样移除所有项中可能存在的 '?'
                final_goals.append([item.replace("?", "") for item in parts])
        
        return final_goals

    else:
        # --- 无 `forall` 的原始逻辑 ---
        lines = goal_content.splitlines()
        actions = [line.strip() for line in lines if line.strip().startswith("(") and line.strip().endswith(")")]
        filtered_actions = [
            act for act in actions 
            if not act.lower().strip("()").strip() in ['and', 'or', 'not']
        ]
        processed_goals = [action.replace("?", "").strip("()").split() for action in filtered_actions]
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
    goals = [goal[1:] if goal and goal[0] == "not" else goal for goal in goals]
    related_goals = [goal for goal in goals if goal[0] in ('inside', 'ontop')]
    unrelated_goals = [goal for goal in goals if goal[0] not in ('inside', 'ontop')]
    related_triples = [(goal[1], goal[0], goal[2]) for goal in related_goals]
    objects = [goal[1] for goal in related_goals] + [goal[2] for goal in related_goals]
    tree = triples_to_tree(related_triples, objects)
    lis = tree_to_list(tree)
    final_goals = [[goal[1], goal[0], goal[2]] for goal in lis] + unrelated_goals

    return [[item.strip('()?') for item in goal] for goal in final_goals]

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


import re
from typing import Dict, List

def _parse_objects_inside(bddl_content: str) -> Dict[str, List[str]]:
    """
    辅助函数：解析 :objects 部分，返回一个从类型到其实例列表的映射。
    例如：{'rag.n.01': ['rag.n.01_1', 'rag.n.01_2']}
    """
    objects_map = {}
    try:
        # 使用正则表达式安全地隔离出 (:objects ...) 内部的内容
        objects_section = re.search(r'\(:objects(.*?)\)', bddl_content, re.DOTALL).group(1)
    except AttributeError:
        # 如果找不到 :objects 部分，可以返回空字典或抛出错误
        return {}

    # 逐行处理 objects 部分的内容
    for line in objects_section.strip().splitlines():
        line = line.strip()
        if not line or line.startswith(';'):
            continue
        
        parts = line.split('-')
        if len(parts) != 2:
            continue

        object_names = parts[0].strip().split()
        object_type = parts[1].strip()

        if object_type not in objects_map:
            objects_map[object_type] = []
        objects_map[object_type].extend(object_names)
        
    return objects_map

def find_inside_objects_in_goal(bddl_content: str) -> List[str]:
    """
    在 BDDL 内容的 (:goal) 部分中查找 (inside ?A ?B) 谓词，并返回所有 A 对象的列表。
    
    - 如果 A 是一个具体的对象实例，直接添加。
    - 如果 A 是一个 forall 变量（类型），则查找并添加该类型的所有实例。
    """
    # 1. 解析 :objects 部分，为处理 forall 情况做准备
    objects_map = _parse_objects_inside(bddl_content)

    # 2. 准确定位并提取 (:goal ...) 的完整内容
    goal_start_index = bddl_content.find("(:goal")
    if goal_start_index == -1:
        return []  # 如果没有 goal 部分，返回空列表

    # 通过匹配括号来找到 goal 部分的结尾
    open_parens = 0
    goal_content = ""
    sub_str = bddl_content[goal_start_index:]
    for i, char in enumerate(sub_str):
        if char == '(':
            open_parens += 1
        elif char == ')':
            open_parens -= 1
        
        if open_parens == 0:
            goal_content = sub_str[:i+1]
            break
    
    # 3. 使用正则表达式查找所有 `(inside ...)` 谓词的第一个参数 A
    #    这个表达式会匹配 '(inside' 后面的第一个非空字符串
    inside_matches = re.findall(r'\(\s*inside\s+([^\s)]+)', goal_content)
    
    result_list = []
    
    # 4. 遍历所有找到的参数 A，并根据情况处理
    for match in inside_matches:
        # 移除参数前缀 '?'
        clean_match = match.lstrip('?')
        
        # 关键逻辑：判断这个参数是一个类型（来自 forall）还是一个实例
        # 我们通过检查它是否存在于 objects_map 的键（即类型名）中来判断
        if clean_match in objects_map:
            # 如果是类型名，说明它来自 forall，我们将该类型的所有实例都加入列表
            result_list.extend(objects_map[clean_match])
        else:
            # 如果不是类型名，说明它是一个具体的对象实例，直接加入列表
            result_list.append(clean_match)
            
    # 使用 set 去除可能存在的重复项，并排序使结果稳定
    return sorted(list(set(result_list)))

import json

def scale_objects_in_json(json_path: str, instances: list, scale_factor: float = 0.3):
    # 读取 JSON 文件
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 获取 instance -> object name 映射
    inst_to_name = data["metadata"]["task"]["inst_to_name"]

    # 遍历要修改的 instance
    for inst in instances:
        if inst not in inst_to_name:
            print(f"⚠️ Warning: {inst} not found in inst_to_name, skipped.")
            continue

        obj_name = inst_to_name[inst]

        if obj_name not in data["objects_info"]["init_info"].keys():
            print(f"⚠️ Warning: {obj_name} not found in objects_info, skipped.")
            continue

        # 修改 scale
        scale = data["objects_info"]["init_info"][obj_name]["args"].get("scale", None)
        if scale:
            new_scale = [round(v * scale_factor, 6) for v in scale]  # 保留6位小数
            data["objects_info"]["init_info"][obj_name]["args"]["scale"] = new_scale
            print(f"✅ Scaled {obj_name}: {scale} -> {new_scale}")
        else:
            print(f"⚠️ Warning: {obj_name} has no scale info, skipped.")

    # 保存回原文件
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

    print(f"🎉 Done! Modified JSON saved to {json_path}")




def run_simulation(scene_file, activity_name, task_description, output_dir, gpu_id,regenerate_flag=False):
    scene_name = scene_file.split('/')[-2]
    if scene_name == activity_name:
        scene_name = scene_file.split('/')[-3]
    bddl_file = scene_file.replace('.json', '.bddl')

    try:

        bddl_content = read_bddl(bddl_file)
        inside_instance = find_inside_objects_in_goal(bddl_content)

        scale_objects_in_json(scene_file, inside_instance, 0.3)

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
        # breakpoint()
        # Step 1: Sample poses and verify task goals
        cache_dict = {}
        saved_path = safe_dump_to_json(cache_dict, scene_name, activity_name, check = True)
        if (regenerate_flag) or (not saved_path):
            flag = False
            json_path = os.path.join(posecache_dir, f"{scene_name}_{activity_name[:30]}.json")
            if os.path.exists(json_path):
                try:
                    data = json.load(open(json_path, 'r'))
                    flag = True
                except Exception as e:
                    print(f"尝试加载 {json_path} 失败: {e}")
            
            cache_dict = {"nearby": {}, "ontop": {}, "inside": {},"nextto": {},"under": {}}
            if flag and regenerate_flag:
                cache_dict["nearby"] = data["nearby"]
            bddl_content = read_bddl(bddl_file)
            bddl_content = replace_objects_in_bddl(bddl_content, scene_file)
            goals = process_goal_section(bddl_content)
            task_objects = get_task_object_names(scene_file)
            real_task_objects = []
            if not flag:
                for object_name in task_objects:
                    query_pose = get_object_pose(scene_file, object_name)
                    if query_pose is not None:
                        print(f"Found pose for {object_name}: {query_pose}")
                        cache_dict["nearby"][object_name] = query_pose
                    else:
                        real_task_objects.append(object_name)

                _primitive_controller = StarterSemanticActionPrimitives(env, enable_head_tracking=False)
                for object_name in real_task_objects:
                    object = env.scene.object_registry("name", object_name)
                    sampled_pose_2d = None
                    print("Try to sample pose on other object")
                    for obj in env.scene.objects:
                        if (object.states[Inside].get_value(obj) and obj.name != object.name) or (object.states[OnTop].get_value(obj) and "floors_" not in obj.name):

                            if cache_dict["nearby"].get(obj.name) is None:
                                if get_object_pose(scene_file, obj.name) is None:
                                    query_pose = None
                                else:
                                    query_pose = get_object_pose(scene_file, obj.name)
                            else:
                                query_pose = cache_dict["nearby"].get(obj.name)
                            
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
                        cache_dict["nearby"][object.name] = query_pose
                        continue

                    if 'floors' not in object.name and 'window' not in object.name and 'wall' not in object.name and 'ceilings' not in object.name:
                        cache_dict["nearby"][object.name] = []
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
                            cache_dict["nearby"][object.name] = sampled_pose_2d

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
                        if regenerate_flag:
                            try:
                                os.remove(json_path)
                                print(f"文件 '{json_path}' 已成功删除。")
                            except FileNotFoundError:
                                print(f"错误：文件 '{json_path}' 未找到。")
                            except OSError as e:
                                print(f"删除文件时出错: {e}")
                        sys.exit(1)
                elif goal[0] == "inside":
                    # import pdb;pdb.set_trace()
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
                        if regenerate_flag:
                            try:
                                os.remove(json_path)
                                print(f"文件 '{json_path}' 已成功删除。")
                            except FileNotFoundError:
                                print(f"错误：文件 '{json_path}' 未找到。")
                            except OSError as e:
                                print(f"删除文件时出错: {e}")
                        sys.exit(1)
                elif goal[0] == "nextto":
                    object1 = env.scene.object_registry("name", goal[1])
                    object2 = env.scene.object_registry("name", goal[2])
                    try:
                        nextto_flag = _set_next_to_magic(object1, object2)
                    except:
                        print("Nextto Error!")
                        sys.exit(1)
                    if nextto_flag and object1.states[object_states.NextTo].get_value(object2):
                        query_pose = object1.get_position_orientation()
                        if query_pose is not None:
                            print(f"Found pose for {object1.name}: {query_pose}")
                            cache_dict["nextto"].setdefault(object1.name, {})[object2.name] = query_pose
                            continue
                    else:
                        print(f"Failed to sample pose for {object1.name} inside! Can not complete the task!")
                        if regenerate_flag:
                            try:
                                os.remove(json_path)
                                print(f"文件 '{json_path}' 已成功删除。")
                            except FileNotFoundError:
                                print(f"错误：文件 '{json_path}' 未找到。")
                            except OSError as e:
                                print(f"删除文件时出错: {e}")
                        sys.exit(1)
                elif goal[0] == "under":
                    object1 = env.scene.object_registry("name", goal[1])
                    object2 = env.scene.object_registry("name", goal[2])
                    object1.states[object_states.Under].set_value(object2, True)
                    if object1.states[object_states.Under].get_value(object2):
                        query_pose = object1.get_position_orientation()
                        if query_pose is not None:
                            print(f"Found pose for {object1.name}: {query_pose}")
                            cache_dict["under"].setdefault(object1.name, {})[object2.name] = query_pose
                            continue
                    else:
                        print(f"Failed to sample pose for {object1.name} inside! Can not complete the task!")
                        if regenerate_flag:
                            try:
                                os.remove(json_path)
                                print(f"文件 '{json_path}' 已成功删除。")
                            except FileNotFoundError:
                                print(f"错误：文件 '{json_path}' 未找到。")
                            except OSError as e:
                                print(f"删除文件时出错: {e}")
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
                        #heat_source_info = get_heat_source_data(object2_category)
                        if "pan" not in object2_category:
                            #如果不是加热源就跳过这个物体
                            continue
                        #如果是加热源，就按照需求采样ontop和inside
                        object2 = env.scene.object_registry("name", object_name)
                        #首先采样inside的位置   
                        #if heat_source_info["requires_inside"] == 1:
                        #    object1.states[object_states.Inside].set_value(object2, True)
                        #    if object1.states[object_states.Inside].get_value(object2):
                                #如果成功放置，记录被放置物体的位置
                        #        query_pose = object1.get_position_orientation()
                                #保存到缓存
                        #    else:
                        #        query_pose = None
                        #    if query_pose is not None:
                        #        print(f"Found pose for {object1.name}: {query_pose}")
                        #        cache_dict["inside"].setdefault(object1.name, {})[object2.name] = query_pose
                        #        continue
                        #    else:
                        #        print(f"Failed to sample pose for {object1.name} inside! Can not complete the task !")
                        #        sys.exit(1)
                        #如果没有要求inside，就采样ontop的位置
                        #else:
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
                            if regenerate_flag:
                                try:
                                    os.remove(json_path)
                                    print(f"文件 '{json_path}' 已成功删除。")
                                except FileNotFoundError:
                                    print(f"错误：文件 '{json_path}' 未找到。")
                                except OSError as e:
                                    print(f"删除文件时出错: {e}")
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
                                if regenerate_flag:
                                    try:
                                        os.remove(json_path)
                                        print(f"文件 '{json_path}' 已成功删除。")
                                    except FileNotFoundError:
                                        print(f"错误：文件 '{json_path}' 未找到。")
                                    except OSError as e:
                                        print(f"删除文件时出错: {e}")
                                sys.exit(1)
                #如果goal是open或者close，说明需要打开或者关闭
                elif goal[0] == "soaked":
                    object1 = env.scene.object_registry("name", goal[1])
                    #循环所有的物体列表
                    for object_name in task_objects:
                        #获得物体的类别，之后与热源进行匹配
                        object2_category = env.scene.object_registry("name", object_name).category
                        print(f'object2_category = {object2_category}')
                        if "sink" not in object2_category and "teapot" not in object2_category:
                            continue
                        object2 = env.scene.object_registry("name", object_name)
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
                            if regenerate_flag:
                                try:
                                    os.remove(json_path)
                                    print(f"文件 '{json_path}' 已成功删除。")
                                except FileNotFoundError:
                                    print(f"错误：文件 '{json_path}' 未找到。")
                                except OSError as e:
                                    print(f"删除文件时出错: {e}")
                            sys.exit(1)
                elif goal[0] == "dusty" or goal[0] == "stained":
                    
                    object1 = env.scene.object_registry("name", goal[1])
                    #循环所有的物体列表
                    for object_name in task_objects:
                        #获得物体的类别，之后与热源进行匹配
                        object2_category = env.scene.object_registry("name", object_name).category
                        print(f'object2_category = {object2_category}')
                        if "sink" not in object2_category and "dishwasher" not in object2_category:
                            continue
                        object2 = env.scene.object_registry("name", object_name)
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
                            print(f"Failed to sample pose for {object1.name} inside!")
                            if regenerate_flag:
                                try:
                                    os.remove(json_path)
                                    print(f"文件 '{json_path}' 已成功删除。")
                                except FileNotFoundError:
                                    print(f"错误：文件 '{json_path}' 未找到。")
                                except OSError as e:
                                    print(f"删除文件时出错: {e}")
                            sys.exit(1)
                    if goal[0] == "stained":
                        object1 = None
                        for object_name in task_objects:
                            object1_category = env.scene.object_registry("name", object_name).category
                            for cleaner_name in ["brush","piece_of_cloth","rag","towel"]:
                                if cleaner_name in object1_category:
                                    object1 = env.scene.object_registry("name", object_name)
                                    break
                            if object1 != None:
                                break
                        if object1 != None:
                            for object_name in task_objects:
                                object2_category = env.scene.object_registry("name", object_name).category
                                print(f'object2_category = {object2_category}')
                                if "sink" not in object2_category and "teapot" not in object2_category:
                                    continue
                                object2 = env.scene.object_registry("name", object_name)
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
                                    if regenerate_flag:
                                        try:
                                            os.remove(json_path)
                                            print(f"文件 '{json_path}' 已成功删除。")
                                        except FileNotFoundError:
                                            print(f"错误：文件 '{json_path}' 未找到。")
                                        except OSError as e:
                                            print(f"删除文件时出错: {e}")
                                    sys.exit(1)


            saved_path = safe_dump_to_json(cache_dict, scene_name, activity_name)
            print("成功写入 JSON 文件!")

        # Step 2: Generate and save LLM plan prompts
        sample_2d_cache = json.load(open(saved_path, 'r'))
        planner_pipeline = planner(env, cfg,full_task_name=activity_name)
        object_dict = {}
        for index, key in enumerate(sample_2d_cache["nearby"].keys()):
            object_dict.update({index: key})

        plan_prompt = planner_pipeline.get_plan_prompt_eai()
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
    
def _set_next_to_magic(obj1,obj2):
    from omnigibson.object_states import Inside, OnTop, AABB, NextTo
    obj1_aabb = obj1.states[AABB].get_value()
    obj2_aabb = obj2.states[AABB].get_value()

    for i in [1,0]:
        for weight in [-1,1]:
            target_center = obj2.aabb_center
            target_center[i] += weight*(0.5 * obj1_aabb[i] + 
                                        0.5 * obj2_aabb[i])
            target_pos = tar_pos_for_new_aabb_center(obj1,target_center)
            obj1.set_position(target_pos)
            if obj1.states[NextTo].get_value(obj2):
                return True
    return False

def tar_pos_for_new_aabb_center(obj1,new_center):
    cur_pos=obj1.get_position()
    cur_aabb_center=obj1.aabb_center
    delta=new_center-cur_aabb_center
    return cur_pos+delta

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