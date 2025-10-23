import omni
import openai
import torch
from PIL import Image
import fcntl
import base64
from io import BytesIO
import numpy as np
import json
import re
import math
import copy
from scipy.spatial.transform import Rotation as R
import omnigibson.utils.transform_utils as T
from omnigibson.action_primitives.starter_semantic_action_primitives import StarterSemanticActionPrimitives
from omnigibson.macros import create_module_macros
import os
import pandas as pd
from omnigibson.object_states import Inside, OnTop, NextTo, ContactBodies
from omnigibson.objects.dataset_object import DatasetObject
from omnigibson.utils.constants import PrimType

nowpath = os.path.abspath(os.path.dirname(__file__))


m = create_module_macros(module_path=os.path.join(nowpath, "../action_primitives/starter_semantic_action_primitives.py"))

from omnigibson.utils.grasping_planning_utils import get_grasp_poses_for_object_sticky, get_grasp_position_for_open
import os
import traceback
import random
from omnigibson import object_states
from src.utils.config_loader import config
posecache_path = config['verification']['posecache_path']
llmplan_path = config['verification']['llmplan_path']
task_dir = config['task_generation']['output_dir']
scene_dir = config['scene_generation']['output_dir']
def get_scene_name(scene_file):

    scene_name_list = os.listdir("omnigibson/data/og_dataset/scenes")
    for scene_name in scene_name_list:
        if scene_name in scene_file:
            return scene_name

def get_pose_cache(scene, activity_name):
    """
    根据场景名称和活动名称获取姿态缓存文件
    
    Args:
        scene (str): 场景名称
        activity_name (str): 活动名称
        
    Returns:
        dict: 加载的姿态缓存数据
        
    Raises:
        FileNotFoundError: 如果找不到匹配的缓存文件
    """
    # 定义可能的路径前缀列表，按优先级排序
    posecache_prefix_list = [
        os.path.join(posecache_path,f"/{scene}_{activity_name[:30]}"),
    ]
    
    # 遍历所有可能的路径前缀
    for prefix in posecache_prefix_list:
        # 尝试直接加载 .json 文件
        json_path = f"{prefix}.json"
        if os.path.exists(json_path):
            try:
                return json.load(open(json_path, 'r'))
            except Exception as e:
                print(f"尝试加载 {json_path} 失败: {e}")
                continue
                
        # 如果直接加载失败，尝试查找以该前缀开头的所有 JSON 文件
        prefix_dir = os.path.dirname(prefix)
        if os.path.exists(prefix_dir):
            prefix_base = os.path.basename(prefix)
            for filename in os.listdir(prefix_dir):
                if filename.startswith(prefix_base) and filename.endswith(".json"):
                    full_path = os.path.join(prefix_dir, filename)
                    try:
                        return json.load(open(full_path, 'r'))
                    except Exception as e:
                        print(f"尝试加载 {full_path} 失败: {e}")
                        continue
    
    # 如果所有尝试都失败，抛出异常
    raise FileNotFoundError(f"找不到匹配的姿态缓存文件，尝试了以下前缀: {posecache_prefix_list}")

def get_obj_dic(scene,task,scene_file=None):
    if scene_file != None:
        json_path = scene_file
        with open(json_path, 'r') as f:
            data = json.load(f)
    else:
        json_path = [os.path.join(scene_dir,f"{scene}/{task}.json"),os.path.join(task_dir,f"{self.scene_name}/{self.task_name}/problem0.jsonl"),]
        for tmp_path in json_path:
            try:
                with open(tmp_path, 'r') as f:
                    data = json.load(f)
                break
            except:
                continue
    
    try:
        long_name_dic = data["metadata"]["task"]["inst_to_name"]
    except KeyError as e:
        raise KeyError(f"Invalid JSON structure, missing key: {e}")
    short_name_dic = {}
    for key, value in long_name_dic.items():
        word_part = key.split('.')[0]      # 提取第一个点前的单词（如 "desk"）
        index_part = key.split('_')[-1]    # 提取最后一个下划线后的序号（如 "1"）
        new_key = f"{word_part}_{index_part}"
        short_name_dic[new_key] = value 
    return long_name_dic ,short_name_dic

# from transformers import AutoModelForCausalLM, AutoTokenizer
class planner():
    def __init__(self, env, config,full_task_name=None,scene_file = None):
        #self.full_activity_name = full_activity_name
        self.env = env
        self.config = config
        self.task = env.task
        self.task_description = ' '.join(config['task']['activity_name'].split('_'))
        self.caption_prompt = config['planner']['caption_prompt']
        self._primitive_controller = StarterSemanticActionPrimitives(env, enable_head_tracking=False)
        #self.pred_map = {"ontop": object_states.OnTop, "inside": object_states.Inside}
        self.state_dict = {}
        self.action_name = ['LEFT_GRASP', 'RIGHT_GRASP', 'LEFT_PLACE_ONTOP', 'RIGHT_PLACE_ONTOP', 'LEFT_PLACE_INSIDE', 'RIGHT_PLACE_INSIDE', 'RIGHT_RELEASE', 'LEFT_RELEASE', 'OPEN', 'CLOSE', 'COOK', 'CLEAN', 'FREEZE', 'UNFREEZE', 'SLICE', 'SOAK', 'DRY', 'TOGGLE_ON', 'TOGGLE_OFF', 'LEFT_PLACE_NEXTTO', 'RIGHT_PLACE_NEXTTO', 'LEFT_TRANSFER_CONTENTS_INSIDE', 'RIGHT_TRANSFER_CONTENTS_INSIDE', 'LEFT_TRANSFER_CONTENTS_ONTOP', 'RIGHT_TRANSFER_CONTENTS_ONTOP', 'LEFT_PLACE_NEXTTO_ONTOP', 'RIGHT_PLACE_NEXTTO_ONTOP', 'LEFT_PLACE_UNDER', 'RIGHT_PLACE_UNDER']
        
        # self.cache_file = f"omnigibson/plannerdemo/plancache/{env.scene.scene_model}_{config['task']['activity_name']}.json"
        # print(f"Cache file: {self.cache_file}")
        self.reasonable_sample_pose = {}
        # try:
        #     self.reasonable_sample_pose = self.load_pose_cache()
        # except:
        #     pass
        self.objects_scope = self.get_objects_scope()

        scene_file = config['scene']['scene_file']
        self.scene_name = get_scene_name(scene_file)
        self.task_name = full_task_name
        self.sample_2d_cache = get_pose_cache(self.scene_name, self.task_name)
        # self.sample_2d_cache = {}
        print(f"sample_2d_cache: {self.sample_2d_cache}")
        self.long_obj_dic ,self.short_obj_dic = get_obj_dic(self.scene_name, self.task_name, scene_file)
        self.object_name = [k for k in self.sample_2d_cache["nearby"].keys()]
        self.reset_object_status()
        self.initial_goal_state = self.get_initial_goal_state()
        self.object_states = self.get_initial_obj_state()
    
    def get_initial_goal_state(self):
        jsonl_path_list = [
            f"bddl/bddl/activity_definitions/{self.task_name[:30]}/problem0.jsonl",
            os.path.join(task_dir,f"{self.scene_name}/{self.task_name}/problem0.jsonl")]


        def read_jsonl_to_list(file_path):
            """
            将 JSONL 文件读取为 Python 列表
            
            参数:
            file_path (str): JSONL 文件路径
            
            返回:
            list: 包含所有 JSON 对象的列表
            """
            data_list = []
            
            try:
                with open(file_path, 'r', encoding='utf-8') as file:
                    for line in file:
                        stripped_line = line.strip()
                        if stripped_line:  # 跳过空行
                            try:
                                data = json.loads(stripped_line)
                                data_list.append(data)
                            except json.JSONDecodeError:
                                print(f"警告: 跳过无效的 JSON 行: {line}")
            except FileNotFoundError:
                print(f"错误: 文件 '{file_path}' 不存在")
                return []
            except Exception as e:
                print(f"读取文件时出错: {str(e)}")
                return []
            
            return data_list
        for jsonl_path in jsonl_path_list:
            intial_goal =  read_jsonl_to_list(jsonl_path)
            if intial_goal != []:
                return intial_goal
        return []
            
        
    def get_initial_obj_state(self):
        self.initial_goal_state = self.get_initial_goal_state()
        data_dict = self.initial_goal_state[0]
        object_states = data_dict["state"]
        new_object_states = {}
        long_obj_dic = self.long_obj_dic
        for obj in object_states:
            if obj.split('_', 1)[0] in ['forall','forpairs','exists']:
                obj_name = obj.split('_', 1)[1]
                for key in long_obj_dic:
                    if obj_name in key:
                        new_object_states[long_obj_dic[key]] = object_states[obj]
            else:
                try:
                    new_object_states[long_obj_dic[obj]] = object_states[obj]
                except Exception as e:
                    print(f"发生未知错误: {str(e)}")
        print(new_object_states)
        return new_object_states
                

    def reset_sample_2d_cache(self):
        config = self.config
        scene_file = config['scene']['scene_file']
        scene = get_scene_name(scene_file)
        self.sample_2d_cache = get_pose_cache(scene, config['task']['activity_name'])

    def reset_object_status(self):
        self.object_status = {}
        try:
            for k, v in self.sample_2d_cache["nearby"].items():
                self.object_status[k] = "fixed"
        except KeyError:
            print("No nextto information in sample_2d_cache, initializing object_status as empty.")
        
    def get_objects_scope(self):
        object_scope = []
        for k, v in self.env.scene._scene_info_meta_inst_to_name.items():
            if 'robot' not in v:
                object_scope.append(v)
        return object_scope
    
    # # def load_pose_cache(self):
    # #     """尝试从文件加载缓存数据，若文件不存在则返回空字典。加共享锁保证读操作安全。"""
    # #     if os.path.exists(self.cache_file):
    # #         try:
    # #             with open(self.cache_file, 'r') as f:
    # #                 fcntl.flock(f, fcntl.LOCK_SH)  # 获取共享锁
    # #                 data = json.load(f)
    # #                 fcntl.flock(f, fcntl.LOCK_UN)
    # #                 return data  # 读取的数据结构为 {object_name: [[x, y, yaw], ...], ...}
    # #         except Exception as e:
    # #             print(f"加载缓存失败: {e}")
    # #             return {}
    # #     else:
    # #         return {}

    # # def save_pose_cache(self):
    # #     """
    # #     每次写入前先读取现有数据，然后更新后写入。
    # #     使用独占锁保证多个服务之间不会并发写入。
    # #     """
    # #     try:
    # #         # 以 a+ 模式打开文件，如果文件不存在则创建
    # #         with open(self.cache_file, 'a+') as f:
    # #             fcntl.flock(f, fcntl.LOCK_EX)  # 获取独占锁
    # #             f.seek(0)
    # #             try:
    # #                 existing_data = json.load(f)
    # #             except Exception:
    # #                 existing_data = {}

    # #             # 将 self.reasonable_sample_pose 中的 tensor 转换为列表（确保 JSON 可序列化）
    # #             serializable_cache = {}
    # #             for key, pose_list in self.reasonable_sample_pose.items():
    # #                 serializable_cache[key] = [
    # #                     pose.tolist() if hasattr(pose, "tolist") else pose for pose in pose_list
    # #                 ]
                
    # #             # 合并已有数据和新数据：如果 key 已存在则追加新采样值（避免重复可额外判断）
    # #             for key, new_poses in serializable_cache.items():
    # #                 if key in existing_data:
    # #                     for pose in new_poses:
    # #                         if pose not in existing_data[key]:
    # #                             existing_data[key].append(pose)
    # #                 else:
    # #                     existing_data[key] = new_poses
                
    #             # 回写文件：先清空再写入更新后的数据
    #             f.seek(0)
    #             f.truncate()
    #             json.dump(existing_data, f)
    #             f.flush()
    #             fcntl.flock(f, fcntl.LOCK_UN)
    #     except Exception as e:
    #         print(f"保存缓存失败: {e}")

    def parse_plan(self, plan):
        structured_plan = extract_and_load_json(plan)
        return structured_plan
        
# [COOK
# CLEAN
# FREEZE
# UNFREEZE
# SLICE
# SOAK
# DRY]
    def take_action(self, action):
        result = True
        try:
            object_list = action['object'].strip().split(",")
            object_list = [self.short_obj_dic[short_name] for short_name in object_list]
            action_word = action['action'].split('_')
    
            print(action_word)
            print(object_list)
            if (len(action_word)==2) and ("GRASP" in action['action']) and (action_word[0] in ['LEFT','RIGHT']):
                result = self.pick_up(object_list[0],action_word[0])
            elif (len(action_word)==3) and ("PLACE" in action['action']) and (action_word[0] in ['LEFT','RIGHT']) and (action_word[2] in ['ONTOP','INSIDE','NEXTTO','UNDER']):
                result = self.place(object_list[0], action_word[2], action_word[0])
            elif action['action'] == "LEFT_PLACE_NEXTTO_ONTOP" or action['action'] == "RIGHT_PLACE_NEXTTO_ONTOP":
                result = self.place_nextto_ontop(object_list[0], object_list[1], action_word[0])
            elif (len(action_word)==2) and ("RELEASE" in action['action']) and (action_word[0] in ['LEFT','RIGHT']):
                result = self.release(hand = action_word[0])
            elif action['action'] == "OPEN":
                result = self.open(object_list[0])
            elif action['action'] == "CLOSE":
                result = self.close(object_list[0])
            elif action['action'] == "TOGGLE_ON":
                result = self.toggle_on(object_list[0])
            elif action['action'] == "TOGGLE_OFF":
                result = self.toggle_off(object_list[0])
            elif (len(action_word)==4) and ("TRANSFER" in action['action']) and (action_word[0] in ['LEFT','RIGHT']) and (action_word[3] in ['ONTOP','INSIDE']):
                result = self.pour(object_list[0], action_word[3], action_word[0])
            elif action['action'] == "COOK":
                result = self.cook(object_list[0])
            elif action['action'] == "FREEZE":
                result = self.freeze(object_list[0],True)
            elif action['action'] == "UNFREEZE":
                result = self.freeze(object_list[0],False)
            elif action['action'] == "SLICE":
                result = self.slice(object_list[0])
            elif action['action'] == "SOAK":
                result = self.soak(object_list[0])
            elif action['action'] == "DRY":
                result = self.dry(object_list[0])
            elif action['action'] == "CLEAN":
                result = self.clean(object_list[0])
            else:
                print(f"Failed to take action: {action['action']}")
                result = False
        except Exception as e: 
            print(f"Failed to take action: {e}")
            result = False
        return result
    
    def check_done(self, action):
        termination, termination_condition = self.get_new_reward()
        return termination, termination_condition
            
    def get_reward(self, action):
        format_reward = 0.5
        reward, reward_dict = self.get_new_reward()
        reward += format_reward
        return reward, reward_dict
    
    def process_object_string(self, object_dict):
        new_dict = {}
        count_dict = {}
        for k, v in object_dict.items():
            parts = v.split('_')
            if parts[-1].isdigit() and int(parts[-1]) > 5:
                modified_number = 1
            else:
                modified_number = 2
            n = len(parts) - modified_number
            result = ' '.join(parts[:n])
            if result in count_dict:
                count_dict[result] += 1
            else:
                count_dict[result] = 1
                
            new_dict[k] = ' '.join([result, str(count_dict[result])])
        return ', '.join([f"object index {k}: {v}" for k,v in new_dict.items()])
    
    def get_plan_prompt(self, image_caption, task_description, object_dict):
        """
        Generate a prompt for the LLM to create a plan with supported actions, including newly added actions.

        Args:
            image_caption (str): Description of the scene.
            task_description (str): Description of the task to be performed.
            object_dict (dict): Dictionary mapping object indices to object names.

        Returns:
            str: A formatted prompt string for the LLM.
        """
        example = """json: {"action_sequence": [{"action": "...", "parameters": {...}}, ...], "task_status_summary": "..."}"""
        move_str = '''{"action": "move", "parameters": {"object_index": n}}'''
        turn_str = '''{"action": "turn", "parameters": {"yaw": y}}'''
        pick_up_str = '''{"action": "pick_up", "parameters": {"object_index": n}}'''
        place_str = '''{"action": "place", "parameters": {"object_index": n, "relation": r}}'''
        move_forward_str = '''{"action": "move_forward", "parameters": {"distance": x, "yaw": y}}'''
        open_str = '''{"action": "open", "parameters": {"object_index": n}}'''
        close_str = '''{"action": "close", "parameters": {"object_index": n}}'''
        toggle_on_str = '''{"action": "toggle_on", "parameters": {"object_index": n}}'''
        toggle_off_str = '''{"action": "toggle_off", "parameters": {"object_index": n}}'''
        heat_object_with_source_str = '''{"action": "heat_object_with_source", "parameters": {"object_index": n, "source_index": m}}'''
        cook_object_with_tool_str = '''{"action": "cook_object_with_tool", "parameters": {"object_index": n, "source_index": m}}'''
        froze_object_with_source_str = '''{"action": "froze_object_with_source", "parameters": {"object_index": n, "source_index": m}}'''
        go_to_room_str = '''{"action": "go_to_room", "parameters": {"room_name": s}}'''

        object_description_prompt = self.process_object_string(object_dict)
        prompt = f"You are a Fetch robot, a versatile mobile manipulator designed for indoor environments, particularly warehouses and distribution centers. Your capabilities include: 7-degree-of-freedom (DoF) arm: Capable of lifting up to 6 kilograms at full extension. Gripper: Maximum opening of 100 mm. Torso lift: Prismatic joint for vertical movement. Mobile base: Enables autonomous navigation for tasks such as picking and placing objects. You are tasked with: {task_description}. The objects in the scenarios include: {object_description_prompt}. Please generate a plan to accomplish the task, considering the following actions: 1. Move to a nearby location around an object. Call move by {move_str}, Specify the object index n. 2. Turn by a specific angle: Provide the angle in degrees. Call turn by {turn_str}, Specify the yaw y in radians. 3. Pick up an object. Call pick_up by {pick_up_str}, Specify the object index n. Note: Objects must be within a 0.5-meter reach. 4. Place an object: Specify the which object to place and the relationship (For example: Place inside trash can (object index 9)). Call place by {place_str}, Specify the object index n and relation r with ontop or inside. 5. Move forward to a specific location: Provide the distance in meters and yaw. Call move_forward by {move_forward_str}, Specify the distance x in meters and yaw y in radians. 6. Open an object (e.g., a door or container). Call open by {open_str}, Specify the object index n. 7. Close an object (e.g., a door or container). Call close by {close_str}, Specify the object index n. 8. Toggle on an object (e.g., a light or appliance). Call toggle_on by {toggle_on_str}, Specify the object index n. 9. Toggle off an object (e.g., a light or appliance). Call toggle_off by {toggle_off_str}, Specify the object index n. 10. Heat an object using a heat source (e.g., stove). Call heat_object_with_source by {heat_object_with_source_str}, Specify the object index n and source index m. 11. Cook an object using a cooking tool (e.g., microwave). Call cook_object_with_tool by {cook_object_with_tool_str}, Specify the object index n and source index m. 12. Freeze an object using a cold source (e.g., freezer). Call froze_object_with_source by {froze_object_with_source_str}, Specify the object index n and source index m. 13. Go to a specific room. Call go_to_room by {go_to_room_str}, Specify the room name s as a string. You should think step by step and give a action sequence to complete the task. After formulating the plan, provide a summary of the current task status to inform subsequent inferences. Your output should generate in a json format, below is a example: {example} Please be note that Your response should follow the json format strictly."
        return prompt

    def get_prompt(self):
        init_dict = self.initial_goal_state[0]
        goal_dict = self.initial_goal_state[1]
        long_obj_dic = self.long_obj_dic
        def get_short_name(key):
            word_part = key.split('.')[0]      # 提取第一个点前的单词（如 "desk"）
            index_part = key.split('_')[-1]    # 提取最后一个下划线后的序号（如 "1"）
            new_key = f"{word_part}_{index_part}"
            return new_key
        def trans(goal_dict):
            state_goals = goal_dict["state"]
            relation_goals = goal_dict["relation"]
            goal_str = ""
            
            for obj in state_goals: 
                for state in state_goals[obj]:
                    tmp_list = []
                    obj_name = ''
                    if obj.split('_', 1)[0] in ['forall','forpairs','exists']:
                        obj_name = obj.split('_', 1)[1]
                        tmp_list = tmp_list + [obj.split('_', 1)[0], obj_name, '-', obj_name]
                    else:
                        obj_name = get_short_name(obj)
                    if state_goals[obj][state]:
                        tmp_list.append('not')
                    tmp_list = tmp_list + [state, obj_name]
                    
                    goal_str = goal_str + str(tmp_list)+ '\n'
    
            for goal in relation_goals:
                tmp_list = []
                obj1 = goal[0]
                predicate = goal[1]
                obj2 = goal[2]
                obj1_name = ''
                if obj1.split('_', 1)[0] in ['forall','forpairs','exists']:
                    obj1_name = obj1.split('_', 1)[1]
                    tmp_list = tmp_list + [obj1.split('_', 1)[0], obj1_name, '-', obj1_name]
                else:
                    obj1_name = get_short_name(obj1)
                if obj2.split('_', 1)[0] in ['forall','forpairs','exists']:
                    obj2_name = obj2.split('_', 1)[1]
                    if tmp_list != []:
                        tmp_list.append('and')
                    tmp_list = tmp_list + [obj2.split('_', 1)[0], obj2_name, '-', obj2_name]
                else:
                    obj2_name = get_short_name(obj2)

                tmp_list = tmp_list + [predicate, obj1_name, obj2_name]
                goal_str = goal_str + str(tmp_list) + '\n'
                
            return goal_str

        init_str = trans(init_dict)
        goal_str = trans(goal_dict)
        obj_str = ""
        for k in long_obj_dic:
            tmp_dic = {'name': get_short_name(k), 'category': k}
            obj_str = obj_str + str(tmp_dic) + '\n'
        return init_str, goal_str, obj_str
         
    def get_plan_prompt_eai(self):
        prompt = '''\n\nProblem:\nYou are designing instructions for a household robot. \nThe goal is to guide the robot to modify its environment from an initial state to a desired final state. \nThe input will be the initial environment state, the target environment state, the objects you can interact with in the environment. \nThe output should be a list of action commands so that after the robot executes the action commands sequentially, the environment will change from the initial state to the target state. \n\nData format: After # is the explanation.\n\nFormat of the states:\nThe environment state is a list starts with a uniary predicate or a binary prediate, followed by one or two obejcts.\nYou will be provided with multiple environment states as the initial state and the target state.\nFor example:\n['inside', 'strawberry_0', 'fridge_97'] #strawberry_0 is inside fridge_97\n['not', 'sliced', 'peach_0'] #peach_0 is not sliced\n['ontop', 'jar_1', 'countertop_84'] #jar_1 is on top of countertop_84\n\nFormat of the action commands:\nAction commands is a dictionary with the following format:\n{\n        \"action\": \"action_name\", \n        \"object\": \"target_obj_name\",\n}\n\nor \n\n{\n        \"action\": \"action_name\", \n        \"object\": \"target_obj_name1,target_obj_name2\",\n}\n\nThe action_name must be one of the following:\nLEFT_GRASP # the robot grasps the object with its left hand, to execute the action, the robot's left hand must be empty, e.g. {'action': 'LEFT_GRASP', 'object': 'apple_0'}.\nRIGHT_GRASP # the robot grasps the object with its right hand, to execute the action, the robot's right hand must be empty, e.g. {'action': 'RIGHT_GRASP', 'object': 'apple_0'}.\nLEFT_PLACE_ONTOP # the robot places the object in its left hand on top of the target object and release the object in its left hand, e.g. {'action': 'LEFT_PLACE_ONTOP', 'object': 'table_1'}.\nRIGHT_PLACE_ONTOP # the robot places the object in its right hand on top of the target object and release the object in its left hand, e.g. {'action': 'RIGHT_PLACE_ONTOP', 'object': 'table_1'}.\nLEFT_PLACE_INSIDE # the robot places the object in its left hand inside the target object and release the object in its left hand, to execute the action, the robot's left hand must hold an object, and the target object can't be closed e.g. {'action': 'LEFT_PLACE_INSIDE', 'object': 'fridge_1'}.\nRIGHT_PLACE_INSIDE # the robot places the object in its right hand inside the target object and release the object in its left hand, to execute the action, the robot's right hand must hold an object, and the target object can't be closed, e.g. {'action': 'RIGHT_PLACE_INSIDE', 'object': 'fridge_1'}.\nRIGHT_RELEASE # the robot directly releases the object in its right hand, to execute the action, the robot's left hand must hold an object, e.g. {'action': 'RIGHT_RELEASE', 'object': 'apple_0'}.\nLEFT_RELEASE # the robot directly releases the object in its left hand, to execute the action, the robot's right hand must hold an object, e.g. {'action': 'LEFT_RELEASE', 'object': 'apple_0'}.\nOPEN # the robot opens the target object, to execute the action, the target object should be openable and closed, also, toggle off the target object first if want to open it, e.g. {'action': 'OPEN', 'object': 'fridge_1'}.\nCLOSE # the robot closes the target object, to execute the action, the target object should be openable and open, e.g. {'action': 'CLOSE', 'object': 'fridge_1'}.\nCOOK # the robot cooks the target object, to execute the action, the target object should be put in a pan, e.g. {'action': 'COOK', 'object': 'apple_0'}.\nCLEAN # the robot cleans the target object, to execute the action, the robot should have a cleaning tool such as rag, the cleaning tool should be soaked if possible, or the target object should be put into a toggled on cleaner like a sink or a dishwasher, e.g. {'action': 'CLEAN', 'object': 'window_0'}.\nFREEZE # the robot freezes the target object e.g. {'action': 'FREEZE', 'object': 'apple_0'}.\nUNFREEZE # the robot unfreezes the target object, e.g. {'action': 'UNFREEZE', 'object': 'apple_0'}.\nSLICE # the robot slices the target object, to execute the action, the robot should have a knife in hand, e.g. {'action': 'SLICE', 'object': 'apple_0'}.\nSOAK # the robot soaks the target object, to execute the action, the target object must be put in a toggled on sink, e.g. {'action': 'SOAK', 'object': 'rag_0'}.\nDRY # the robot dries the target object, e.g. {'action': 'DRY', 'object': 'rag_0'}.\nTOGGLE_ON # the robot toggles on the target object, to execute the action, the target object must be closed if the target object is openable and open e.g. {'action': 'TOGGLE_ON', 'object': 'light_0'}.\nTOGGLE_OFF # the robot toggles off the target object, e.g. {'action': 'TOGGLE_OFF', 'object': 'light_0'}.\nLEFT_PLACE_NEXTTO # the robot places the object in its left hand next to the target object and release the object in its left hand, e.g. {'action': 'LEFT_PLACE_NEXTTO', 'object': 'table_1'}.\nRIGHT_PLACE_NEXTTO # the robot places the object in its right hand next to the target object and release the object in its right hand, e.g. {'action': 'RIGHT_PLACE_NEXTTO', 'object': 'table_1'}.\nLEFT_TRANSFER_CONTENTS_INSIDE # the robot transfers the contents in the object in its left hand inside the target object, e.g. {'action': 'LEFT_TRANSFER_CONTENTS_INSIDE', 'object': 'bow_1'}.\nRIGHT_TRANSFER_CONTENTS_INSIDE # the robot transfers the contents in the object in its right hand inside the target object, e.g. {'action': 'RIGHT_TRANSFER_CONTENTS_INSIDE', 'object': 'bow_1'}.\nLEFT_TRANSFER_CONTENTS_ONTOP # the robot transfers the contents in the object in its left hand on top of the target object, e.g. {'action': 'LEFT_TRANSFER_CONTENTS_ONTOP', 'object': 'table_1'}.\nRIGHT_TRANSFER_CONTENTS_ONTOP # the robot transfers the contents in the object in its right hand on top of the target object, e.g. {'action': 'RIGHT_TRANSFER_CONTENTS_ONTOP', 'object': 'table_1'}.\nLEFT_PLACE_NEXTTO_ONTOP # the robot places the object in its left hand next to target object 1 and on top of the target object 2 and release the object in its left hand, e.g. {'action': 'LEFT_PLACE_NEXTTO_ONTOP', 'object': 'window_0, table_1'}.\nRIGHT_PLACE_NEXTTO_ONTOP # the robot places the object in its right hand next to object 1 and on top of the target object 2 and release the object in its right hand, e.g. {'action': 'RIGHT_PLACE_NEXTTO_ONTOP', 'object': 'window_0, table_1'}.\nLEFT_PLACE_UNDER # the robot places the object in its left hand under the target object and release the object in its left hand, e.g. {'action': 'LEFT_PLACE_UNDER', 'object': 'table_1'}.\nRIGHT_PLACE_UNDER # the robot places the object in its right hand under the target object and release the object in its right hand, e.g. {'action': 'RIGHT_PLACE_UNDER', 'object': 'table_1'}.\n\nFormat of the interactable objects:\nInteractable object will contain multiple lines, each line is a dictionary with the following format:\n{\n    \"name\": \"object_name\",\n    \"category\": \"object_category\"\n}\nobject_name is the name of the object, which you must use in the action command, object_category is the category of the object, which provides a hint for you in interpreting initial and goal condtions.\n\nPlease pay specail attention:\n1. The robot can only hold one object in each hand.\n2. Action name must be one of the above action names, and the object name must be one of the object names listed in the interactable objects.\n3. All PLACE actions will release the object in the robot's hand, you don't need to explicitly RELEASE the object after the PLACE action.\n4. For LEFT_PLACE_NEXTTO_ONTOP and RIGHT_PLACE_NEXTTO_ONTOP, the action command are in the format of {'action': 'action_name', 'object': 'obj_name1, obj_name2'}\n5. If you want to perform an action to an target object, you must make sure the target object is not inside a closed object.\n6. For actions like OPEN, CLOSE, SLICE, COOK, CLEAN, SOAK, DRY, FREEZE, UNFREEZE, TOGGLE_ON, TOGGLE_OFF, at least one of the robot's hands must be empty, and the target object must have the corresponding property like they're openable, toggleable, etc.\n7. For PLACE actions and RELEASE actions, the robot must hold an object in the corresponding hand.\n8. Before slicing an object, the robot can only interact with the object (e.g. peach_0), after slicing the object, the robot can only interact with the sliced object (e.g. peach_0_part_0).\n\n\nExamples: after# is the explanation.\n\nExample 1:\nInput:\ninitial environment state:\n['stained', 'sink_7']\n['stained', 'bathtub_4']\n['not', 'soaked', 'rag_0']\n['onfloor', 'rag_0', 'room_floor_bathroom_0']\n['inside', 'rag_0', 'cabinet_1']\n['not', 'open', 'cabinet_1']\n\n\ntarget environment state:\n['not', 'stained', 'bathtub_4']\n['not', 'stained', 'sink_7']\n['and', 'soaked', 'rag_0', 'inside', 'rag_0', 'bucket_0']\n\n\ninteractable objects:\n{'name': 'sink_7', 'category': 'sink.n.01'}\n{'name': 'bathtub_4', 'category': 'bathtub.n.01'}\n{'name': 'bucket_0', 'category': 'bucket.n.01'}\n{'name': 'rag_0', 'category': 'rag.n.01'}\n{'name': 'cabinet_1', 'category': 'cabinet.n.01'}\n\n\nPlease output the list of action commands (in the given format) so that after the robot executes the action commands sequentially, the current environment state will change to target environment state. Usually, the robot needs to execute multiple action commands consecutively to achieve final state. Please output multiple action commands rather than just one. Only output the list of action commands with nothing else.\n\nOutput:\n[\n    {\n        \"action\": \"OPEN\",\n        \"object\": \"cabinet_1\"\n    }, # you want to get the rag_0 from cabinet_1, should open it first\n    {\n        \"action\": \"RIGHT_GRASP\",\n        \"object\": \"rag_0\"\n    }, # you want to clean the sink_7 and bathtub_4, you found them stained, so you need to soak the rag_0 first\n    {\n        \"action\": \"RIGHT_PLACE_INSIDE\",\n        \"object\": \"sink_7\"\n    }, # to soak the rag_0, you need to place it inside the sink_7\n    {\n        \"action\": \"TOGGLE_ON\",\n        \"object\": \"sink_7\"\n    }, # to soak the rag_0, you need to toggle on the sink_7\n    {\n        \"action\": \"SOAK\",\n        \"object\": \"rag_0\"\n    }, # now you can soak the rag_0\n    {\n        \"action\": \"TOGGLE_OFF\",\n        \"object\": \"sink_7\"\n    }, # after soaking the rag_0, you need to toggle off the sink_7\n    {\n        \"action\": \"LEFT_GRASP\",\n        \"object\": \"rag_0\"\n    }, # now you can grasp soaked rag_0 to clean stain\n    {\n        \"action\": \"CLEAN\",\n        \"object\": \"sink_7\"\n    }, # now you clean the sink_7\n    {\n        \"action\": \"CLEAN\",\n        \"object\": \"bathtub_4\"\n    }, # now you clean the bathtub_4\n    {\n        \"action\": \"LEFT_PLACE_INSIDE\",\n        \"object\": \"bucket_0\"\n    } # after cleaning the sink_7, you need to place the rag_0 inside the bucket_0\n]\n\n'''
        
        init_state, target_state, object_list = self.get_prompt()
        add_prompt = f'''Your task:\nInput:\ninitial environment state:\n{init_state}\n\n\ntarget environment state:\n{target_state}\n\n\ninteractable objects:\n{object_list}\n\n\nPlease output the list of action commands (in the given format) so that after the robot executes the action commands sequentially, the current environment state will change to target environment state. Usually, the robot needs to execute multiple action commands consecutively to achieve final state. Please output multiple action commands rather than just one. Only output the list of action commands with nothing else.\n\nOutput:\n'''
        return prompt + add_prompt


        

    def observation_caption_prompt(self, obs, object_dict):
        obs = self.env.get_obs()
        obs_rgb_img = obs[0]['robot0::robot0:eyes:Camera:0::rgb'][:, :, :3]
        image_caption = "test"
        # return obs_rgb_img, object_list, image_caption
        return image_caption

    def generate_image_caption(self, image_tensor, object_dict, api_key):
        # 转换图像
        base64_image = tensor_to_base64(image_tensor)
        
        # 构建提示词
        system_prompt = f"Generate detailed English visual description for robotic action planning. \nFocus on object positions, spatial relationships, material properties, and environmental context. The robot's task is {self.task_description}. You should pay attention to relative information. \nKeep sentences concise and factual."
        system_prompt = self.caption_prompt
        
        object_description_prompt = ', '.join([f"object index {k}: {v.split('_')[0]}" for k,v in object_dict.items()])
        
        user_prompt = f"Visible objects: {', '.join(object_description_prompt)}. Describe scene details relevant for physical interaction:"
        
        # 调用API
        openai.api_key = api_key
        response = openai.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user_prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}",
                                "detail": "high"
                            }
                        }
                    ]
                }
            ],
            max_tokens=1000
        )
        gpt_caption = response.choices[0].message.content
        caption = gpt_caption + f"\n\nAll objects in the scene include: {object_description_prompt}"
        return caption

    def check_interactability(self,obj_name):
        target_object = self.env.scene.object_registry("name", obj_name)
        for other_object_name in self.object_name:
            if other_object_name != obj_name:
                other_object = self.env.scene.object_registry("name", other_object_name)
                try:
                    inside_state = target_object.states[object_states.Inside].get_value(other_object)
                    open_state = other_object.states[object_states.Open].get_value()
                    if inside_state and not open_state:
                        print(f'{obj_name} is inside closed {other_object_name}')
                        return False
                except KeyError:
                    continue
        return True

    def move_to_object(self, object_name):
        target_object = self.env.scene.object_registry("name", object_name)
        sampled_pose_2d = None
        
        if object_name in self.sample_2d_cache["nearby"]:
            if self.object_status[object_name] == "fixed":
                # sampled_pose_2d = random.choice(self.sample_2d_cache[object_name])
                sampled_pose_2d = self.sample_2d_cache["nearby"][object_name]
    
        # if object_name in self.sample_2d_cache:
        #     if self.object_status[object_name] == "fixed":
        #         sampled_pose_2d = random.choice(self.sample_2d_cache[object_name])

        #for obj in self.env.scene.objects:
            #if (target_object.states[Inside].get_value(obj) and obj.name != target_object.name) or (target_object.states[OnTop].get_value(target_object) and "floors_" not in obj.name):
                # 如果目标对象在另一个对象内部，则不允许移动
                #return self.move_to_object(obj.name)

        if sampled_pose_2d is None:
            while sampled_pose_2d is None:
                sampled_pose_2d = self._primitive_controller._sample_pose_near_object(
                    target_object, pose_on_obj=None, distance_lo=0.1, distance_hi=1.5, yaw_lo=-3.1416, yaw_hi=3.1416
                )
                if sampled_pose_2d is not None:
                    # 如果该对象已有列表，则追加；否则新建列表
                    if object_name in self.reasonable_sample_pose:
                        self.reasonable_sample_pose[object_name].append(sampled_pose_2d)
                    else:
                        self.reasonable_sample_pose[object_name] = [sampled_pose_2d]
                    # 保存更新后的缓存到文件
                    # self.save_pose_cache()
                else:
                    # 如果采样失败，可以考虑休眠或重试
                    pass

                # except Exception as e:
                    # print(f"Failed to sample pose: {e}")
        robot_pose = self.env.robots[0].get_position_orientation()
        # sampled_pose_2d is x,y,yaw, robot_position is x,y,z, quaternion. Please modify robot position accroding to sampled_pose_2d
        try:
            sampled_x, sampled_y, sampled_yaw = sampled_pose_2d
        except:
            sampled_x, sampled_y, sampled_yaw = sampled_pose_2d[0]
        robot_x, robot_y, robot_z = robot_pose[0]
        robot_orientation = robot_pose[1]
        robot_rotation = R.from_quat(robot_orientation)  # 创建四元数对象
        robot_euler_angles = robot_rotation.as_euler('xyz', degrees=False)  # 获取欧拉角

        # 修改 yaw 值
        robot_euler_angles[2] = sampled_yaw  # 这里的[2]是yaw轴，可能需要根据实际情况调整

        # 将修改后的欧拉角转换回四元数
        new_robot_rotation = R.from_euler('xyz', robot_euler_angles, degrees=False)
        new_robot_orientation = new_robot_rotation.as_quat()

        # 更新机器人的位置和姿态
        new_robot_position = torch.cat([torch.tensor([sampled_x]), torch.tensor([sampled_y]), torch.tensor([robot_z])])  # 这里假设 z 不变
        new_robot_position_and_orientation = (new_robot_position, torch.tensor(new_robot_orientation, dtype=torch.float32))
        self.env.robots[0].set_position_orientation(position = new_robot_position_and_orientation[0], orientation = new_robot_position_and_orientation[1])
        # 计算机器人和目标物体之间的距离
        robot_position = self.env.robots[0].get_position()
        target_position = target_object.get_position()
        distance = torch.norm(robot_position[:2] - target_position[:2])
        print(f"Robot successfully moved. Distance to {object_name}: {distance:.2f} meters")
        print(f"Robot position: {robot_position}, Target position: {target_position}")
        return True
    
    def navigate_to_if_needed(self,obj_name):
        target_object = self.env.scene.object_registry("name", obj_name)
        if not self.if_robot_close_to_obj(target_object):
            self.move_to_object(obj_name)

    def pick_up(self, object_name, hand):
        if not self.check_interactability(object_name):
            return False
        if "grasped" in self.state_dict and hand in self.state_dict["grasped"]:
            print(f"{hand} is grasped")
            return False
        self.navigate_to_if_needed(object_name)
        target_object = self.env.scene.object_registry("name", object_name)
        # for action in self._primitive_controller._grasp(target_object):
        #     if action is not None:
        #         self.env.step(action)
        #     else:
        #         break
        # return f"pick up object {object_name}"
        result = self.god_grasp(target_object, hand)
        #if not result:
            #for obj in self.env.scene.objects:
                #if (target_object.states[Inside].get_value(obj) and obj.name != target_object.name) or (target_object.states[OnTop].get_value(target_object) and "floors_" not in obj.name):
                    # 如果目标对象在另一个对象内部，则不允许移动
                    #result_vo = self.god_grasp(obj)
                    #result = result_vo
                    #break
        if result:
            self.object_status[object_name] = "picked"
            
        return result
    
    def god_grasp(self, target_object, hand):
        if "grasped" not in self.state_dict:
            self.state_dict["grasped"] = {}
        self.state_dict["grasped"][hand] = target_object.name
        self.object_status[target_object.name] = "picked"
        self.object_status[target_object.name] = "picked"
        print(f'成功拿起物体：{target_object.name}')
        return True
        
    def if_robot_close_to_obj(self, target_object):

        robot_position = self.env.robots[0].get_position()
        obj_position = target_object.get_position()
        distance = torch.norm(robot_position[:2] - obj_position[:2])
        _,_,box_extent = self.get_relative_aabb_information(target_object)

        # 如果距离大于0.5米(机器人手臂最大可达范围),则无法抓取
        if distance > 3 + (1/2)*max(box_extent):
            print(f"'{target_object.name}' is too far from the robot, distance is {distance:.2f}m. The available reach is {3 + (1/2)*max(box_extent)}m.")
            return False
        else:
            return True
    def get_obj_in_hand(self):
        in_hand = []
        if "grasped" not in self.state_dict:
            print("Not grasped")
        else:
            for k,v in self.state_dict["grasped"].items():
                in_hand.append(v)
        return in_hand

    def get_new_reward(self):
        all_goal_num = 0
        all_success_num = 0
        goal_dict = self.initial_goal_state[1]
        state_goals = goal_dict["state"]
        relation_goals = goal_dict["relation"]
        long_obj_dic = self.long_obj_dic
        info = {'state':{},'relation':{}}
        for obj in state_goals:
            obj_list = []
            if obj.split('_', 1)[0] in ['forall','forpairs','exists']:
                obj_name = obj.split('_', 1)[1]
                for key in long_obj_dic:
                    if obj_name in key:
                        obj_list.append(long_obj_dic[key])
            else:
                obj_list.append(long_obj_dic[obj])
            
            if obj.split('_', 1)[0] == "forall":
                goal_num = len(obj_list)
            elif obj.split('_', 1)[0] == "forpairs":
                goal_num = 2
            else:
                goal_num = 1
            
            for state in state_goals[obj]:
                success_num = 0
                for obj_name in obj_list:
                    if self.check_state(obj_name, state, state_goals[obj][state]):
                        success_num = success_num + 1
                all_goal_num = all_goal_num + goal_num
                all_success_num = all_success_num + min(success_num, goal_num)
                info['state'][(obj, state, state_goals[obj][state])] = {"goal_num":goal_num, "success_num":min(success_num, goal_num)}

        for goal in relation_goals:
            obj1 = goal[0]
            predicate = goal[1]
            obj2 = goal[2]

            obj1_list = []
            if obj1.split('_', 1)[0] in ['forall','forpairs','exists']:
                obj1_name = obj1.split('_', 1)[1]
                for key in long_obj_dic:
                    if obj1_name in key:
                        obj1_list.append(long_obj_dic[key])
            else:
                obj1_list.append(long_obj_dic[obj1])
            
            if obj1.split('_', 1)[0] == "forall":
                goal_num1 = len(obj1_list)
            elif obj1.split('_', 1)[0] == "forpairs":
                goal_num1 = 2
            else:
                goal_num1 = 1

            obj2_list = []
            if obj2.split('_', 1)[0] in ['forall','forpairs','exists']:
                obj2_name = obj2.split('_', 1)[1]
                for key in long_obj_dic:
                    if obj2_name in key:
                        obj2_list.append(long_obj_dic[key])
            else:
                obj2_list.append(long_obj_dic[obj2])
            
            if obj2.split('_', 1)[0] == "forall":
                goal_num2 = len(obj2_list)
            elif obj2.split('_', 1)[0] == "forpairs":
                goal_num2 = 2
            else:
                goal_num2 = 1
            
            goal_num = goal_num1 * goal_num2

            success_num = 0
            for obj1_name in obj1_list:
                for obj2_name in obj2_list:
                    if self.check_relation(obj1_name, predicate, obj2_name):
                        success_num = success_num + 1

            all_goal_num = all_goal_num + goal_num
            all_success_num = all_success_num + min(success_num, goal_num)
            info['relation'][(obj1, predicate, obj2)] = {"goal_num":goal_num, "success_num":min(success_num, goal_num)}  

        if all_goal_num == 0:
            success_rate = 0
        else:
            success_rate = all_success_num / all_goal_num
        return success_rate, info

    def check_state(self, obj_name, state, true_or_false):
        obj = self.env.scene.object_registry("name", obj_name)
        new_object_states = self.object_states
        if state == "cooked":
            if object_states.Cooked in obj.states:
                return (obj.states[object_states.Cooked].get_value() == true_or_false)
        elif state == "frozen":
            if object_states.Frozen in obj.states:
                return (obj.states[object_states.Frozen].get_value() == true_or_false)
        elif state == "open":
            if object_states.Open in obj.states:
                return (obj.states[object_states.Open].get_value() == true_or_false)
        elif state == "toggled_on":
            if object_states.ToggledOn in obj.states:
                return (obj.states[object_states.ToggledOn].get_value() == true_or_false)
        elif state == "dusty":
            if "dusty" in new_object_states[obj_name]:
                return (new_object_states[obj_name]["dusty"] == true_or_false)
        elif state == "sliced":
            if "sliced" in new_object_states[obj_name]:
                return (new_object_states[obj_name]["sliced"] == true_or_false)
        elif state == "soaked":
            if "soaked" in new_object_states[obj_name]:
                return (new_object_states[obj_name]["soaked"] == true_or_false)
        elif state == "stained":
            if "stained" in new_object_states[obj_name]:
                return (new_object_states[obj_name]["stained"] == true_or_false)
        return False

    def check_relation(self, obj1_name, predicate, obj2_name):
        obj1 = self.env.scene.object_registry("name", obj1_name)
        obj2 = self.env.scene.object_registry("name", obj2_name)
        if predicate == "inside":
            return obj1.states[object_states.Inside].get_value(obj2)
        elif predicate == "ontop":
            return obj1.states[object_states.OnTop].get_value(obj2)
        elif predicate == "nextto":
            return obj1.states[object_states.NextTo].get_value(obj2)
        elif predicate == "under":
            return obj2.states[object_states.Under].get_value(obj2)
        return False



    def slice(self, object_name):
        if not self.check_interactability(object_name):
            return False
        target_object = self.env.scene.object_registry("name", object_name)
        allow_slicer = ["table_knife.n.01", "cleaver.n.01"]
        reversed_dict = {v: k for k, v in self.long_obj_dic.items()}
        has_slicer = False
        obj_in_hand = self.get_obj_in_hand()
        for obj in obj_in_hand:
            for slicer in allow_slicer:
                if slicer in reversed_dict[obj]:
                    has_slicer = True
                    break
            if has_slicer:
                break
        if not has_slicer:
            return False
        self.navigate_to_if_needed(object_name)
        self.object_states[object_name]["sliced"] = True
        return True

    def soak(self, object_name):
        if not self.check_interactability(object_name):
            return False
        target_object = self.env.scene.object_registry("name", object_name)
        allowed_soakers=["sink","teapot"]
        in_soaker = False
        for other_obj_name in self.object_name:
            if other_obj_name != object_name:
                for soaker_name in allowed_soakers:
                    if soaker_name in other_obj_name:
                        soaker = self.env.scene.object_registry("name", other_obj_name)
                        if (object_states.ToggledOn in soaker.states and soaker.states[object_states.ToggledOn].get_value() or object_states.ToggledOn not in soaker.states):
                            if target_object.states[object_states.Inside].get_value(soaker) or target_object.states[object_states.OnTop].get_value(soaker) or target_object.states[object_states.NextTo].get_value(soaker):
                                in_soaker = True
                                break
            if in_soaker:
                break
        if not in_soaker:
            print("not in soaker")
            return False
        self.navigate_to_if_needed(object_name)
        self.object_states[object_name]["soaked"] = True
        return True
        
    def dry(self, object_name):
        if not self.check_interactability(object_name):
            return False
        target_object = self.env.scene.object_registry("name", object_name)
        allowed_soakers=["sink","teapot"]
        self.navigate_to_if_needed(object_name)
        self.object_states[object_name]["soaked"] = False
        return True

    def clean(self, object_name):
        # clean will clean both dust and stain
        flag1=False
        flag2=False
        try_clean_dust=False
        try_clean_stain=False
        if not ("dusty" in self.object_states[object_name] and self.object_states[object_name]["dusty"]==False):
            flag1=self.clean_dust(object_name)
            try_clean_dust=True
        if not ("stained" in self.object_states[object_name] and self.object_states[object_name]["stained"]==False):
            flag2=self.clean_stain(object_name)
            try_clean_stain=True
        if not (try_clean_dust or try_clean_stain):
            print("Clean failed, object is already clean")
            return False
        return flag1 or flag2

    def clean_dust(self, object_name):
        in_cleaner=False
        if not self.check_interactability(object_name):
            return False
        target_object = self.env.scene.object_registry("name", object_name)
        allowed_cleaners=["dishwasher","sink"]
        for other_obj_name in self.object_name:
            if other_obj_name != object_name:
                for soaker_name in allowed_cleaners:
                    if soaker_name in other_obj_name:
                        soaker = self.env.scene.object_registry("name", other_obj_name)
                        if (object_states.ToggledOn in soaker.states and soaker.states[object_states.ToggledOn].get_value() or object_states.ToggledOn not in soaker.states):
                            if target_object.states[object_states.Inside].get_value(soaker) or target_object.states[object_states.OnTop].get_value(soaker) or target_object.states[object_states.NextTo].get_value(soaker):
                                in_cleaner = True
                                break
            if in_cleaner:
                break
        
        if not ("dusty" in self.object_states[object_name]):
            print("Clean-dust failed, object cannot be clean-dusted")
            return False
        
        # check if cleaner in inventory
        has_cleaner=False
        allowed_cleaning_tool=["vacuum","brush","piece_of_cloth","rag","towel"]
        reversed_dict = {v: k for k, v in self.long_obj_dic.items()}

        obj_in_hand = self.get_obj_in_hand()
        for obj in obj_in_hand:
            for cleaning_tool in allowed_cleaning_tool:
                if cleaning_tool in reversed_dict[obj]:
                    has_cleaner = True
                    break
            if has_cleaner:
                break

        if not in_cleaner and not has_cleaner:
            print("Clean-dust failed, please place object in a toggled on cleaner or get a cleaner first")
            return False

        ## post effect
        self.navigate_to_if_needed(object_name)
        self.object_states[object_name]["dusty"] = False
        print(f"Clean-dust {object_name} success")
        return True
    
    def clean_stain(self, object_name):
        in_cleaner=False
        if not self.check_interactability(object_name):
            return False
        target_object = self.env.scene.object_registry("name", object_name)
        allowed_cleaners=["sink"]
        for other_obj_name in self.object_name:
            if other_obj_name != object_name:
                for soaker_name in allowed_cleaners:
                    if soaker_name in other_obj_name:
                        soaker = self.env.scene.object_registry("name", other_obj_name)
                        if (object_states.ToggledOn in soaker.states and soaker.states[object_states.ToggledOn].get_value() or object_states.ToggledOn not in soaker.states):
                            if target_object.states[object_states.Inside].get_value(soaker) or target_object.states[object_states.OnTop].get_value(soaker) or target_object.states[object_states.NextTo].get_value(soaker):
                                in_cleaner = True
                                break
            if in_cleaner:
                break
        
        if not ("stained" in self.object_states[object_name]):
            print("Clean-dust failed, object cannot be clean-dusted")
            return False
        
        # check if cleaner in inventory
        has_soaked_cleaner=False
        allowed_cleaning_tool=["vacuum","brush","piece_of_cloth","rag","towel"]
        reversed_dict = {v: k for k, v in self.long_obj_dic.items()}
        allowed_new_cleaners=["detergent"]
        obj_in_hand = self.get_obj_in_hand()
        for obj in obj_in_hand:
            for cleaning_tool in allowed_cleaning_tool:
                if cleaning_tool in reversed_dict[obj]:
                    if 'soaked' in self.object_states[obj] and self.object_states[obj]["soaked"]:
                        has_soaked_cleaner = True
                        break
            
            for new_cleaner in allowed_new_cleaners:
                if new_cleaner in obj:
                    has_soaked_cleaner = True

            if has_soaked_cleaner:
                break 

        if not in_cleaner and not has_soaked_cleaner:
            print("Clean-dust failed, please place object in a toggled on cleaner or get a cleaner first")
            return False

        ## post effect
        self.navigate_to_if_needed(object_name)
        self.object_states[object_name]["stained"] = False
        print(f"Clean-dust {object_name} success")
        return True

    def open(self, object_name):
        if not self.check_interactability(object_name):
            return False
        target_object = self.env.scene.object_registry("name", object_name)
        if not target_object:
            print(f"Failed to open: Object '{object_name}' not found in scene")
            return False
        # 1. 判断物体的 object_states 列表中是否包含 "open" 状态。
        try:
            open_state = target_object.states[object_states.Open].get_value()
            # 如果是关着的物体就维持原状
            if open_state:
                print(f"No need to close: Object '{object_name}' is already closed")
                return True
        except:
            print(f"Failed to close: Object '{object_name}' does not support 'Open' state")
            return False
        
        self.navigate_to_if_needed(object_name)
        # 3. 执行打开操作：
        # 使用 state.set_value(True) 设置状态为 "open"。
        if "grasped" in self.state_dict and 'LEFT' in self.state_dict["grasped"] and 'RIGHT' in self.state_dict["grasped"]:
            print("both hand is full")
            return False
        try:
            toggle_on_state = target_object.states[object_states.ToggledOn].get_value()
            if toggle_on_state:
                print(f"{object_name} is toggled on, cannot be opened")
                return False
        except:
            pass
        target_object.states[object_states.Open].set_value(True)
        # 返回打开后的状态：state.get_value()。
        print(f'尝试打开{target_object.name},结果为{target_object.states[object_states.Open].get_value()}')
        print(f'尝试打开{target_object.name},结果为{target_object.states[object_states.Open].get_value()}')
        return target_object.states[object_states.Open].get_value()

    def close(self, object_name):
        if not self.check_interactability(object_name):
            return False
        target_object = self.env.scene.object_registry("name", object_name)
        if not target_object:
            print(f"Failed to close: Object '{object_name}' not found in scene")
            return False
        # 1. 判断物体的 object_states 列表中是否包含 "open" 状态。
        try:
            open_state = target_object.states[object_states.Open].get_value()
            # 如果是关着的物体就维持原状
            if not open_state:
                print(f"No need to close: Object '{object_name}' is already closed")
                return True
        except:
            print(f"Failed to close: Object '{object_name}' does not support 'Open' state")
            return False
        self.navigate_to_if_needed(object_name)

        if "grasped" in self.state_dict and 'LEFT' in self.state_dict["grasped"] and 'RIGHT' in self.state_dict["grasped"]:
            print("both hand is full")
            return False
        # 3. 执行关闭操作：
        # 使用 state.set_value(False) 设置状态为 "closed"。
        target_object.states[object_states.Open].set_value(False)
        print(f'尝试关闭{target_object.name},结果为{not target_object.states[object_states.Open].get_value()}')
        # 返回关闭后的状态：state.get_value()。
        return not target_object.states[object_states.Open].get_value()

    def toggle_on(self, object_name):
        if not self.check_interactability(object_name):
            return False
        target_object = self.env.scene.object_registry("name", object_name)
        if not target_object:
            print(f"Failed to toggle on: Object '{object_name}' not found in scene")
            return False
        try:
            open_state = target_object.states[object_states.ToggledOn].get_value()
            # 如果是开着的物体就维持原状
            if open_state:
                print(f"No need to toggle on: Object '{object_name}' is already toggled on")
                return True
        except:
            print(f"Failed to toggle on: Object '{object_name}' does not support 'ToggledOn' state")
            return False
        self.navigate_to_if_needed(object_name)
        # 3. 执行打开操作：
        # 使用 state.set_value(True) 设置状态为 "open"。
        if "grasped" in self.state_dict and 'LEFT' in self.state_dict["grasped"] and 'RIGHT' in self.state_dict["grasped"]:
            print("both hand is full")
            return False
        try:
            open_state = target_object.states[object_states.Open].get_value()
            if open_state:
                print(f"{object_name} is opened, cannot be toggled on")
                return False
        except:
            pass
        target_object.states[object_states.ToggledOn].set_value(True)
        # 返回打开后的状态：state.get_value()。
        print(f'尝试开启{target_object.name},结果为{target_object.states[object_states.ToggledOn].get_value()}')
        return target_object.states[object_states.ToggledOn].get_value()

    def toggle_off(self, object_name):
        if not self.check_interactability(object_name):
            return False
        target_object = self.env.scene.object_registry("name", object_name)
        if not target_object:
            print(f"Failed to toggle off: Object '{object_name}' not found in scene")
            return False
        try:
            open_state = target_object.states[object_states.ToggledOn].get_value()
            # 如果是关着的物体就维持原状
            if not open_state:
                print(f"No need to toggle off: Object '{object_name}' is already toggled off")
                return True
        except:
            print(f"Failed to toggle off: Object '{object_name}' does not support 'ToggledOn' state")
            return False
        self.navigate_to_if_needed(object_name)

        if "grasped" in self.state_dict and 'LEFT' in self.state_dict["grasped"] and 'RIGHT' in self.state_dict["grasped"]:
            print("both hand is full")
            return False
        target_object.states[object_states.ToggledOn].set_value(False)
        # 返回关闭后的状态：state.get_value()。
        print(f'尝试关闭{target_object.name},结果为{not target_object.states[object_states.ToggledOn].get_value()}')
        return not target_object.states[object_states.ToggledOn].get_value()
    
    
    
    def cook(self, target_object_name):
        if not self.check_interactability(target_object_name):
            return False
        allowered_cookers=["saucepan","frying_pan"]
        target_object = self.env.scene.object_registry("name", target_object_name)
        if object_states.Cooked not in target_object.states:
            print("The target object can't be cooked")
            return False
        in_cooker = False
        for cooker in allowered_cookers:
            for obj_name in self.long_obj_dic:
                if cooker in obj_name:
                    cooker_obj = self.env.scene.object_registry("name", self.long_obj_dic[obj_name])
                    if target_object.states[object_states.Inside].get_value(cooker_obj) or target_object.states[object_states.OnTop].get_value(cooker_obj) or target_object.states[object_states.NextTo].get_value(cooker_obj):
                        in_cooker = True
                        break
            if in_cooker:
                break
        if not in_cooker:
            print("Cook failed, please place object in a cooker first")
            return False

        self.navigate_to_if_needed(target_object_name)
        target_object.states[object_states.Cooked].set_value(True)

        return target_object.states[object_states.Cooked].get_value() 

    def freeze(self, target_object_name, freeze_or_unfreeze):
        if not self.check_interactability(object_name):
            return False
        
        target_object = self.env.scene.object_registry("name", target_object_name)
        if object_states.Frozen not in target_object.states:
            print("The target object can't be cooked")
            return 
            
        self.navigate_to_if_needed(target_object_name)
        target_object.states[object_states.Frozen].set_value(freeze_or_unfreeze)

        return target_object.states[object_states.Frozen].get_value()  

    
    def _get_pose_in_robot_frame(self, pose):
        """
        Converts the pose in the world frame to the robot frame

        Args:
            pose_2d (Iterable): (x, y, yaw) 2d pose

        Returns:
            2-tuple:
                - 3-array: (x,y,z) Position in the world frame
                - 4-array: (x,y,z,w) Quaternion orientation in the world frame
        """
        body_pose = self.env.robots[0].get_position_orientation()
        return T.relative_pose_transform(*pose, *body_pose)
    
    def move_forward(self, distance_m, yaw_degrees):
        robot_pose = self.env.robots[0].get_position_orientation() # position, quaternion
        robot_position = robot_pose[0]
        robot_orientation = robot_pose[1]
        robot_rotation = R.from_quat(robot_orientation)  # 创建四元数对象
        robot_euler_angles = robot_rotation.as_euler('xyz', degrees=True)  # 获取欧拉角
        # 修改 yaw 值
        robot_euler_angles[2] += yaw_degrees  # 这里的[2]是yaw轴，可能需要根据实际情况调整
        # 将修改后的欧拉角转换回四元数
        new_robot_rotation = R.from_euler('xyz', robot_euler_angles, degrees=True)
        # 计算新的位置
        new_robot_position = robot_position + torch.tensor([distance_m * math.cos(math.radians(yaw_degrees)), distance_m * math.sin(math.radians(yaw_degrees)), 0])
        # 更新机器人的位置和姿态
        self.env.robots[0].set_position_orientation(position = new_robot_position, orientation = torch.tensor(new_robot_rotation.as_quat(), dtype=torch.float32))
        return f"move forward {distance_m} meters"
    
    def parse_response(self,response):
        # find [ and ]
        try:
            start_idx=response.find("[")
            end_idx=response.find("]")
            action_list=eval(response[start_idx:end_idx+1])
            new_action=[]
            for action in action_list:
                if isinstance(action,dict):
                    if "action" in action and "object" in action:
                        new_action.append(action)
        except Exception as e:
            print(e)
            new_action=[]
        return new_action

    def evaluate_format_reward(self,actions):
        related_reward = 0
        if len(actions)==0:
            print("No actions found")
            return related_reward, False
        for action in actions:
            if "action" not in action or "object" not in action:
                print("action or object not found")
                return related_reward, False
        old_obj = []
        for action in actions:
            action_name=action["action"]
            if action_name not in self.action_name:
                print(f"action {action_name} not found")
            for obj in action["object"].strip().split(","):
                obj_name=obj.strip()
                if obj_name not in self.short_obj_dic:
                    print(f"object {obj_name} not found")
                elif obj_name not in old_obj:
                    old_obj.append(obj_name)
                    related_reward += 0.1
        return related_reward ,True

    def get_reward_dict(self, llm_plan):
        """
        Evaluate the reward for a given LLM plan, logging each step's success and reward.

        Args:
            llm_plan (str or dict): The LLM-generated plan, either as a JSON string or a dictionary.

        Returns:
            list: A list containing a dictionary with action sequence, reward, LLM plan, and done status.
        """
        print(f"Processing LLM plan: {llm_plan}")

        # Initialize sample data and default action
        sample_data = []
        init_action = torch.tensor([0.0] * 11)

        actions=self.parse_response(llm_plan)
        related_reward, correct_format = self.evaluate_format_reward(actions)
        if not correct_format:
            print(f"Error format")
            sample_data.append({
                "action_sequence": None,
                "reward": -1,
                "llm_plan": llm_plan,
                "done": False
            })
            return sample_data

        # Main execution
        try:
            # Reset environment and object status
            self.env.reset()
            self.reset_object_status()
            self.reset_object_status()
            self.reset_sample_2d_cache()
            self.state_dict = {}
            self.object_states = self.get_initial_obj_state()

            # Build object dictionary
            #object_dict = {index: key for index, key in enumerate(self.sample_2d_cache["nextto"].keys())}
            #print(f"Object dictionary created with {len(object_dict)} objects")

            action_sequence = actions
            step_count = len(action_sequence)
            print(f"Executing {step_count} actions in sequence")

            # Execute actions
            for i, action in enumerate(action_sequence, 1):
                print(f"\nStep {i}: Executing action {action}")
                result = self.take_action(action)
                if not result:
                    print(f"Step {i}: Action failed, terminating sequence")
                    break
                print(f"Step {i}: Action succeeded")

            # Check completion and compute rewards
            done, termination_dict = self.check_done(init_action)
            reward, reward_dict = self.get_reward(init_action)
            step_penalty = step_count * 0.01
            
            total_reward = reward - step_penalty + related_reward
            print(f"\nReward calculation:")
            print(f"  Base reward: {reward:.3f}")
            print(f"  Step penalty: -{step_penalty:.3f} ({step_count} steps * 0.01)")
            print(f"  Related reward: +{related_reward:.3f}")
            print(f"  Total reward: {total_reward:.3f}")
            print(f"Task completed: {done}")
            print(f"Termination details: {termination_dict}")

            sample_data.append({
                "action_sequence": action_sequence,
                "reward": total_reward,
                "llm_plan": llm_plan,
                "done": done
            })

        except Exception as e:
            error_trace = traceback.format_exc()
            print(f"\nFailed to process plan: {error_trace}")
            print(f"LLM plan: {llm_plan}")
            sample_data.append({
                "action_sequence": None,
                "reward": -1,
                "llm_plan": llm_plan,
                "done": False
            })

        return sample_data

    # def get_reward_dict(self, llm_plan):
    #     print(f"llm_plan: {llm_plan}")
    #     # open("omnigibson/plannerdemo/llm_plan_log.txt", "a").write(f"llm_plan: {llm_plan}")
    #     init_action = torch.tensor([0.0 for i in range(11)])
    #     # llm_plan = json.loads(llm_plan)
    #     # print(f"llm_plan after jsonload is {llm_plan}")
    #     if isinstance(llm_plan, str):
    #         try:
    #             # 尝试将字符串转换为字典
    #             llm_plan = json.loads(llm_plan)
    #             llm_plan_str = llm_plan['llm_plans']
    #         except:
    #             llm_plan_str = llm_plan
    #             llm_plan = {}
    #         llm_plan["structured_plan"] = self.parse_plan(llm_plan_str)
        
    #     sample_data = []
        
    #     try:
    #         # for llm_plan in llm_plans:
    #         self.env.reset()
    #         self.reset_object_status()
    #         self.reset_object_status()
    #         object_dict = {}
    #         sample_data = []
    #         self.state_dict = {}
    #         # for index, item in enumerate(self.env.scene.objects):
    #         for index, key in enumerate(self.sample_2d_cache.keys()):
    #             object_dict.update({index: key})
    #         structured_plan = llm_plan['structured_plan']
    #         if structured_plan is None:
    #             sample_data.append({"action_sequence": None, "reward": -1, "llm_plan": llm_plan, "done": False})
    #         else: 
    #             action_sequence = llm_plan['structured_plan']['action_sequence']
    #             for action in action_sequence:
    #                 result = self.take_action(action, object_dict)
    #                 # self.env.step(init_action)
    #                 if not result:
    #                     break
    #             # obs = self.env.step(init_action)
    #             # check done
    #             step_count = len(action_sequence)
    #             done, termination_dict = self.check_done(init_action) 
    #             reward, reward_dict = self.get_reward(init_action)
    #             reward -= step_count * 0.01
    #             reward += self.related_reward(action_sequence, object_dict)
    #             sample_data.append({"action_sequence": action_sequence, "reward": reward, "llm_plan": llm_plan, "done": done})
    #     except Exception as e:
    #         # 使用 traceback 获取完整的错误堆栈
    #         error_trace = traceback.format_exc()
    #         print(f"Failed to get reward: \n{error_trace}")  # 打印完整错误堆栈
    #         print(f"llm plan is {llm_plan}")
    #         sample_data.append({
    #             "action_sequence": None,
    #             "reward": -1,
    #             "llm_plan": llm_plan,
    #             "done": False
    #         })
    #     return sample_data

    def related_reward(self, action_sequence, object_dict):
        object_scope_count = {k: 0 for k in self.objects_scope}
        extra_reward = 0
        mentioned_objects = []
        for action in action_sequence:
            try:
                mentioned_objects.append(action['parameters']['object_index'])
            except:
                pass
        for mentioned_object in mentioned_objects:
            try:
                if object_dict[mentioned_object] in self.objects_scope:
                    if object_scope_count[object_dict[mentioned_object]] <=2:
                        extra_reward += 0.05
                        object_scope_count[object_dict[mentioned_object]] += 1
            except:
                continue
                
        return extra_reward
            
    def release(self, obj_name, hand):
        if "grasped" not in self.state_dict:
            print("Not grasped")
            return False
        elif hand not in self.state_dict["grasped"]:
            print(f"{hand} not grasped")
            return False
        elif obj_name is not None and self.state_dict["grasped"][hand]!=obj_name:
            print(f"{obj_name} is not in hand")
            return False
        else:
            self.state_dict["grasped"].pop(hand)
            return True
    def place_nextto_ontop(self, obj1_name, obj2_name, predicate, hand):
        if "grasped" not in self.state_dict:
            print("Not grasped")
            return False
        elif hand not in self.state_dict["grasped"]:
            print(f"{hand} not grasped")
            return False
        if not self.check_interactability(obj1_name):
            return False
        if not self.check_interactability(obj2_name):
            return False
        obj_in_hand_list = self.get_obj_in_hand()
        if obj1_name in obj_in_hand_list or obj2_name in obj_in_hand_list:
            print(f"{obj1_name} or {obj2_name} in hand")
            return False
        self.navigate_to_if_needed(obj2_name)
        obj_in_hand_name = self.state_dict["grasped"][hand]
        obj_in_hand = self.env.scene.object_registry("name", obj_in_hand_name)
        obj1 = self.env.scene.object_registry("name", obj1_name)
        obj2 = self.env.scene.object_registry("name", obj2_name)
        try:
            if obj_in_hand_name in self.sample_2d_cache["ontop"]:
                position_and_orientation = self.sample_2d_cache["ontop"][obj_in_hand_name][obj2_name]
                obj_in_hand.set_position_orientation(position = position_and_orientation[0], orientation = position_and_orientation[1])
            else:
                print("can not place correctly, no sampled pose")
                return False
            self.sample_2d_cache["nearby"][obj_in_hand_name] = self.sample_2d_cache["nearby"][obj2_name]
            self.state_dict["grasped"].pop(hand)
        except Exception as e:
            print(f"Error placing object: {e}")
        return obj_in_hand.states[object_states.OnTop].get_value(obj2) and obj_in_hand.states[object_states.NextTo].get_value(obj1)

    def place(self, obj_name, predicate, hand):
        predicate = predicate.lower()
        if "grasped" not in self.state_dict:
            print("Not grasped")
            return False
        elif hand not in self.state_dict["grasped"]:
            print(f"{hand} not grasped")
            return False
        if not self.check_interactability(obj_name):
            return False
        else:
            obj_in_hand_name = self.state_dict["grasped"][hand]
            obj_in_hand = self.env.scene.object_registry("name", obj_in_hand_name)
            obj = self.env.scene.object_registry("name", obj_name)

            self.navigate_to_if_needed(obj_name)
            # if relation == "inside":
            #     result = self.place_inside(object_in_hand, object)
            # elif relation == "ontop":
            #     result = self.place_ontop(object_in_hand, object)
            try:
                #if predicate == "inside":
                    #obj_in_hand.set_position(obj.get_position())
                    #self.state_dict.pop("grasped")
                    #return True
                #elif predicate == "ontop":
                    # print("step1")
                    # obj_in_hand.states[self.pred_map["ontop"]].set_value(obj,True)
                    #try:
                        #print("step2")
                        #obj_in_hand_ontop_pos = self._primitive_controller._sample_pose_with_object_and_predicate(self.pred_map[predicate], obj_in_hand, obj)
                        # obj_z = obj.aabb[1][2]
                        # obj_xy = obj.aabb_center[:2]
                        # obj_in_hand_height = obj_in_hand.aabb_center[2] - obj_in_hand.aabb[0][2]
                        # obj_in_hand_z = obj_z + obj_in_hand_height
                        # obj_ontop_pos = torch.cat([obj_xy, torch.tensor([obj_in_hand_z])])
                        #print("step3")
                        #obj_in_hand.set_position_orientation(obj_in_hand_ontop_pos[0], obj_in_hand_ontop_pos[1])
                        #print("step4")
                        #if not obj_in_hand.states[self.pred_map["ontop"]].get_value(obj):
                            #print("step5")
                            #obj_in_hand.states[self.pred_map["ontop"]].set_value(obj,True)
                    #except:
                        #print("step5")
                        #obj_in_hand.states[self.pred_map["ontop"]].set_value(obj,True)
                    #if obj_in_hand.states[self.pred_map["ontop"]].get_value(obj):
                        #self.state_dict.pop("grasped")
                        #return True
                    #else:
                        #print("can not place correctly")
                        #return False
                if predicate == "inside":
                    try:
                        open_state = obj.states[object_states.Open].get_value()
                        # 如果是开着的物体就维持原状
                        if not open_state:
                            print(f"{obj_name} is closed")
                            return False
                    except:
                        pass
                if obj_in_hand_name in self.sample_2d_cache[predicate]:
                    position_and_orientation = self.sample_2d_cache[predicate][obj_in_hand_name][obj_name]
                    obj_in_hand.set_position_orientation(position = position_and_orientation[0], orientation = position_and_orientation[1])
                else:
                    print("can not place correctly, no sampled pose")
                    return False
                self.sample_2d_cache["nearby"][obj_in_hand_name] = self.sample_2d_cache["nearby"][obj_name]
                self.state_dict["grasped"].pop(hand)
                if predicate == "inside":
                    #if not obj_in_hand.states[object_states.Inside].get_value(obj):
                        #obj_in_hand.states[object_states.Inside].set_value(obj,True)
                    print(f'实际放置inside结果为：{obj_in_hand.states[object_states.Inside].get_value(obj)}')
                    print(f'实际放置inside位置为：{position_and_orientation[0]}')
                    return obj_in_hand.states[object_states.Inside].get_value(obj)
                elif predicate == "ontop":
                    #if not obj_in_hand.states[object_states.OnTop].get_value(obj):
                        #obj_in_hand.states[object_states.OnTop].set_value(obj,True)
                    print(f'实际放置ontop结果为：{obj_in_hand.states[object_states.OnTop].get_value(obj)}')
                    print(f'实际放置ontop位置为：{position_and_orientation[0]}')
                    return obj_in_hand.states[object_states.OnTop].get_value(obj)
                elif predicate == "nextto":
                    #if not obj_in_hand.states[object_states.OnTop].get_value(obj):
                        #obj_in_hand.states[object_states.OnTop].set_value(obj,True)
                    print(f'实际放置nextto结果为：{obj_in_hand.states[object_states.NextTo].get_value(obj)}')
                    print(f'实际放置nextto位置为：{position_and_orientation[0]}')
                    return obj_in_hand.states[object_states.NextTo].get_value(obj)
                elif predicate == "under":
                    #if not obj_in_hand.states[object_states.OnTop].get_value(obj):
                        #obj_in_hand.states[object_states.OnTop].set_value(obj,True)
                    print(f'实际放置under结果为：{obj_in_hand.states[object_states.Under].get_value(obj)}')
                    print(f'实际放置under位置为：{position_and_orientation[0]}')
                    return obj_in_hand.states[object_states.Under].get_value(obj)
            except:
                print("have problem when placing")
                return False
            
    def pour(self,obj_name,predicate,hand):
        predicate = predicate.lower()

        if "grasped" not in self.state_dict or hand not in self.state_dict["grasped"]:
            print(f"Error: {hand} is not grasping an object to pour from.")
            return False
        
        container_in_hand_name = self.state_dict["grasped"][hand]
        container_in_hand = self.env.scene.object_registry("name", container_in_hand_name)
        target_obj = self.env.scene.object_registry("name", obj_name)

        content_objects = []
        try:
            all_scene_objects = self.env.scene.objects.object_registry()
        except AttributeError:
            print("错误: 无法获取场景对象列表。请确认 `self.env.scene.objects` 是否是正确的API。")
            return False

        for obj_in_scene in all_scene_objects:

            if obj_in_scene.name == container_in_hand.name:
                continue
            

            if obj_in_scene.states[object_states.Inside].get_value(container_in_hand):
                content_objects.append(obj_in_scene)


        if not content_objects:
            print(f"'{container_in_hand_name}' is empty, nothing to pour.")
            return False

        self.navigate_to_if_needed(obj_name)

        all_poured_successfully = True
        for content_obj in content_objects:
            try:
                content_obj_name = content_obj.name
                if predicate == "inside":
                    try:
                        if not target_obj.states[object_states.Open].get_value():
                            print(f"Cannot pour inside closed object: '{obj_name}'")
                            all_poured_successfully = False
                            continue
                    except:
                        pass
                
                if content_obj_name in self.sample_2d_cache[predicate] and obj_name in self.sample_2d_cache[predicate][content_obj_name]:
                    pos_and_ori = self.sample_2d_cache[predicate][content_obj_name][obj_name]
                    content_obj.set_position_orientation(position=pos_and_ori[0], orientation=pos_and_ori[1])
                    print(f"Poured '{content_obj_name}' {predicate} '{obj_name}'.")
                else:
                    print(f"Cannot pour '{content_obj_name}', no sampled pose found.")
                    all_poured_successfully = False

            except Exception as e:
                print(f"An error occurred while pouring '{content_obj.name}': {e}")
                all_poured_successfully = False
        

        if all_poured_successfully:
            print(f"Successfully poured contents from '{container_in_hand_name}'.")
        else:
            print(f"Pouring from '{container_in_hand_name}' was not fully successful.")
            
        return all_poured_successfully
    def get_relative_aabb_information(self,objA):
        if isinstance(objA, DatasetObject) and objA.prim_type == PrimType.RIGID:
            # Retrieve base CoM frame-aligned bounding box parallel to the XY plane
            parallel_bbox_center, parallel_bbox_orn, parallel_bbox_extents, _ = objA.get_base_aligned_bbox(
                xy_aligned=True
            )
        else:
            try:
                aabb_lower, aabb_upper = objA.states[AABB].get_value()
                parallel_bbox_center = (aabb_lower + aabb_upper) / 2.0
                parallel_bbox_orn = th.tensor([0.0, 0.0, 0.0, 1.0])
                parallel_bbox_extents = aabb_upper - aabb_lower
            except:
                print(f"{objA.name} has no AABB states!")
                return None, None, None

        return parallel_bbox_center.numpy()[:2], parallel_bbox_orn, parallel_bbox_extents.numpy()[:2]        
    def test_tool(self, action_sequence):
        action_sequence[0] = {'action': 'move', 'parameters': {'object_index': 95}}
        action_sequence[1] = {'action': 'pick_up', 'parameters': {'object_index': 95}}
        action_sequence[2] = {'action': 'move', 'parameters': {'object_index': 94}}
        action_sequence[3] = {'action': 'place', 'parameters': {'object_index': 94, 'relation': 'inside'}}
        return action_sequence
        
def tensor_to_base64(tensor):
    # 转换torch tensor为numpy数组
    if tensor.max() <= 1.0:
        tensor = tensor * 255
    numpy_array = tensor.cpu().numpy().astype(np.uint8)
    
    # 确保通道顺序正确（H,W,3）
    pil_image = Image.fromarray(numpy_array)
    
    # 转换为JPEG格式的base64
    buffered = BytesIO()
    pil_image.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

def extract_and_load_json(input_string):
    # 使用正则表达式匹配包含 "json" 的行后的内容
    
    pattern = r'json:\n*(\{[\s\S]*\})'
    match = re.search(pattern, input_string)
    
    if match:
        json_str = match.group(1).strip()
        try:
            # 解析 JSON 字符串并返回
            return json.loads(json_str)
        except:
            return None
    else:
        print("没有找到有效的 JSON 数据")
        return None
    
def get_info_in_synset(df, name):
    info = None
    for i in range(len(df)):
        item = df["synset"][i]
        if item.split('.')[0] in name:
            info = df.iloc[i]
    return info   