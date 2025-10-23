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
if not os.path.exists(os.path.join(nowpath, "../action_primitives/starter_semantic_action_primitives.py")):
    nowpath = "/GPFS/rhome/yuanzhuoding/planner-bench/planner_bench"

m = create_module_macros(module_path=os.path.join(nowpath, "../action_primitives/starter_semantic_action_primitives.py"))

from omnigibson.utils.grasping_planning_utils import get_grasp_poses_for_object_sticky, get_grasp_position_for_open
import os
import traceback
import random
from omnigibson import object_states

def get_scene_name(scene_file):
    """从场景文件路径中提取场景名称"""
    # 提取场景名称 - 适应新的路径格式
    # 全体scene_name列表：
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
        f"omnigibson/plannerdemo/simulation_tools/posecache_0502/{scene}/{activity_name}",
        f"omnigibson/plannerdemo/simulation_tools/posecache0502/{scene}/{activity_name}",
        f"omnigibson/plannerdemo/simulation_tools/posecache0502/{scene}_{activity_name}",
        f"omnigibson/plannerdemo/simulation_tools/posecache_0502/{scene}_{activity_name[:30]}",
        f"omnigibson/plannerdemo/simulation_tools/posecache0502/{scene}_{activity_name[:30]}",
        f"omnigibson/plannerdemo/simulation_tools/posecache/our/{scene}_{activity_name[:30]}",
        f"omnigibson/plannerdemo/simulation_tools/posecache/layoutgpt/{scene}_{activity_name[:30]}",
        f"omnigibson/plannerdemo/simulation_tools/posecache/noholodeck/{scene}_{activity_name[:30]}",
        f"omnigibson/plannerdemo/simulation_tools/posecache/notree/{scene}_{activity_name[:30]}"
    ]
    
    # 遍历所有可能的路径前缀
    for prefix in posecache_prefix_list:
        # 尝试直接加载 .json 文件
        json_path = f"{prefix}.json"
        if os.path.exists(json_path):ok
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


# from transformers import AutoModelForCausalLM, AutoTokenizer
class planner():
    def __init__(self, env, config,full_activity_name=None):
        self.full_activity_name = full_activity_name
        self.env = env
        self.config = config
        self.task = env.task
        self.task_description = ' '.join(config['task']['activity_name'].split('_'))
        self.caption_prompt = config['planner']['caption_prompt']
        self._primitive_controller = StarterSemanticActionPrimitives(env, enable_head_tracking=False)
        self.pred_map = {"ontop": object_states.OnTop, "inside": object_states.Inside}
        self.state_dict = {}
        # self.cache_file = f"omnigibson/plannerdemo/plancache/{env.scene.scene_model}_{config['task']['activity_name']}.json"
        # print(f"Cache file: {self.cache_file}")
        self.reasonable_sample_pose = {}
        # try:
        #     self.reasonable_sample_pose = self.load_pose_cache()
        # except:
        #     pass
        self.objects_scope = self.get_objects_scope()

        scene_file = config['scene']['scene_file']
        scene = get_scene_name(scene_file)
        self.sample_2d_cache = get_pose_cache(scene, config['task']['activity_name'])
        # self.sample_2d_cache = {}
        print(f"sample_2d_cache: {self.sample_2d_cache}")
        self.reset_object_status()

    def reset_sample_2d_cache(self):
        config = self.config
        scene_file = config['scene']['scene_file']
        scene = get_scene_name(scene_file)
        self.sample_2d_cache = get_pose_cache(scene, config['task']['activity_name'])

    def reset_object_status(self):
        self.object_status = {}
        try:
            for k, v in self.sample_2d_cache["nextto"].items():
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
           
    
    def take_action(self, action, object_dict):
        result = True
        try:
            if action['action'] == "move":
                result = self.move_to_object(object_dict[action['parameters']['object_index']])
            elif action['action'] == "pick_up":
                result = self.pick_up(object_dict[action['parameters']['object_index']])
            elif action['action'] == "turn":
                result = self.turn(action['parameters']['yaw'])
            elif action['action'] == "move_forward":
                result = self.move_forward(action['parameters']['distance'], action['parameters']['yaw'])
            elif action['action'] == "place":
                result = self.place(action['parameters']['object_index'], action['parameters']['relation'], object_dict)
            elif action['action'] == "open":
                result = self.open(object_dict[action['parameters']['object_index']])
            elif action['action'] == "close":
                result = self.close(object_dict[action['parameters']['object_index']])
            elif action['action'] == "toggle_on":
                result = self.toggle_on(object_dict[action['parameters']['object_index']])
            elif action['action'] == "toggle_off":
                result = self.toggle_off(object_dict[action['parameters']['object_index']])
            elif action['action'] == "heat_object_with_source":
                result = self.heat_object_with_source(object_dict[action['parameters']['object_index']],object_dict[action['parameters']['source_index']])
            elif action['action'] == "cook_object_with_tool":
                result = self.cook_object_with_tool(object_dict[action['parameters']['object_index']],object_dict[action['parameters']['source_index']])
            elif action['action'] == "froze_object_with_source":
                result = self.froze_object_with_source(object_dict[action['parameters']['object_index']],object_dict[action['parameters']['source_index']])
            elif action['action'] == "go_to_room":
                result = self.go_to_room(object_dict[action['parameters']['room_name']])
            else:
                print(f"Failed to take action: {action['action']}")
                result = False
        except Exception as e: 
            print(f"Failed to take action: {e}")
            result = False
        return result
    
    def check_done(self, action):
        termination, termination_condition = self.env.task._step_reward(self.env, action)
        return termination, termination_condition
            
    def get_reward(self, action):
        format_reward = 0.5
        reward, reward_dict = self.env.task._step_reward(self.env, action)
        reward += format_reward
        return reward, reward_dict

    # def process_object_string(self, s):
    #     # Split by '_'
    #     parts = s.split('_')
        
    #     # 如果最后一部分是大于5的数字
    #     if parts[-1].isdigit() and int(parts[-1]) > 5:
    #         modified_number = 1
    #     else:
    #         modified_number = 2
        
    #     # Find the position of the numeric part (the last part)
    #     n = len(parts) - modified_number
        
    #     # Join all items before the number
    #     result = ' '.join(parts[:n])
        
    #     return result
    
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

    # def get_plan_prompt(self, image_caption, task_description, object_dict):
    #     example = """
    #         json:
    #         {
    #             "action_sequence": [
    #                 {
    #                     "action": "...",
    #                     "parameters": {
    #                         ...
    #                     }
    #                 },
    #                 ...
    #             ],
    #             "task_status_summary": "..."
    #         }

    #     """
    #     move_str = '''{'action': 'move', 'parameters': {'object_index': n}}'''
    #     turn_str = '''{'action': 'turn', 'parameters': {'yaw': y}}'''
    #     pick_up_str = '''{'action': 'pick_up', 'parameters': {'object_index': n}}'''
    #     place_str = '''{'action': 'place', 'parameters': {'object_index': n, "relation": r}}'''
    #     move_forward_str = '''{'action': 'place', 'parameters': {'distance': x, 'yaw': y}}'''
        
    #     # object_description_prompt = ', '.join([f"object index {k}: {self.process_object_string(v)}" for k,v in object_dict.items()])
    #     object_description_prompt = self.process_object_string(object_dict)
    #     prompt = f"You are a Fetch robot, a versatile mobile manipulator designed for indoor environments, particularly warehouses and distribution centers. Your capabilities include: \n7-degree-of-freedom (DoF) arm: Capable of lifting up to 6 kilograms at full extension. \nGripper: Maximum opening of 100 mm. \nTorso lift: Prismatic joint for vertical movement. \nMobile base: Enables autonomous navigation for tasks such as picking and placing objects. \nYou are tasked with: {task_description}. \nThe objects in the scenarios include: {object_description_prompt}. \nPlease generate a plan to accomplish the task, considering the following actions: \n1. Move to a nearby location around an object. Call move by {move_str}, Specify the object index n. \n2. Turn by a specific angle: Provide the angle in degrees. Call turn by {turn_str}, Specify the yaw y in radius. \n3. Pick up an object. Call pick_up by {pick_up_str}, Specify the object index n. Note: Objects must be within a 0.5-meter reach. \n4. Place an object: Specify the which object to place and the relationship(For example: Place inside trash can(object index 9)). Call place by {place_str}, Specify the object index n and relation s with ontop or inside. \n5. Move forward to a specific location: Provide the distance in meters and yaw. Call move_forward by {move_forward_str}, Specify the distance x in meter and yaw y in radius. \nYou should think step by step and give a action sequence to complete the task. \nAfter formulating the plan, provide a summary of the current task status to inform subsequent inferences .\nYour output should generate in a json format, below is a example: \n{example} \nPlease be note that Your response should follow the json format strictly."
    #     return prompt
    
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
    
    def move_to_object(self, object_name):
        target_object = self.env.scene.object_registry("name", object_name)
        sampled_pose_2d = None
        
        if object_name in self.sample_2d_cache["nextto"]:
            if self.object_status[object_name] == "fixed":
                # sampled_pose_2d = random.choice(self.sample_2d_cache[object_name])
                sampled_pose_2d = self.sample_2d_cache["nextto"][object_name]

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
    
    def pick_up(self, object_name):
        target_object = self.env.scene.object_registry("name", object_name)
        # for action in self._primitive_controller._grasp(target_object):
        #     if action is not None:
        #         self.env.step(action)
        #     else:
        #         break
        # return f"pick up object {object_name}"
        result = self.god_grasp(target_object)
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
    
    def god_grasp(self, target_object):
        grasp_poses = get_grasp_poses_for_object_sticky(target_object)
        grasp_pose, object_direction = random.choice(grasp_poses)

        # Prepare data for the approach later.
        # approach_pos = grasp_pose[0] + object_direction * m.GRASP_APPROACH_DISTANCE
        approach_pos = grasp_pose[0] + object_direction * 1
        approach_pose = (approach_pos, grasp_pose[1])
        # if not self._primitive_controller._target_in_reach_of_robot(grasp_pose):
        # 计算机器人当前位置与抓取位置之间的距离
        robot_pos = self.env.robots[0].get_position()
        grasp_pos = grasp_pose[0]
        distance = torch.norm(robot_pos - grasp_pos)
        _,_,box_extent = self.get_relative_aabb_information(target_object)
        robot_extent = self.env.robots[0].aabb_extent[:2]

        # 如果距离大于0.5米(机器人手臂最大可达范围),则无法抓取
        if distance > 3.5 + (1)*max(box_extent) + max(robot_extent):
            print(f"'{target_object.name}' is too far from the robot, distance is {distance:.2f}m. The available reach is {3 + (1)*max(box_extent) + max(robot_extent)}m.")
            return False
        else:
            self.state_dict["grasped"] = target_object.name
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
        
    def turn(self, angle_degrees):
        robot_pose = self.env.robots[0].get_position_orientation()[1] # quaternion
        robot_rotation = R.from_quat(robot_pose)  # 创建四元数对象
        robot_euler_angles = robot_rotation.as_euler('xyz', degrees=True)  # 获取欧拉角
        # 修改 yaw 值
        robot_euler_angles[2] += angle_degrees  # 这里的[2]是yaw轴，可能需要根据实际情况调整
        # 将修改后的欧拉角转换回四元数
        new_robot_rotation = R.from_euler('xyz', robot_euler_angles, degrees=True)
        # self.env.robots[0].set_position_orientation(orientation = torch.tensor(new_robot_rotation.as_quat(), dtype=torch.float32))
        return f"turn {angle_degrees} degrees"
    
    def open(self, object_name):
        target_object = self.env.scene.object_registry("name", object_name)
        if not target_object:
            print(f"Failed to open: Object '{object_name}' not found in scene")
            return False
        # 1. 判断物体的 object_states 列表中是否包含 "open" 状态。
        try:
            open_state = target_object.states[object_states.Open].get_value()
            # 如果是开着的物体就维持原状
            if open_state:
                print(f"No need to open: Object '{object_name}' is already open")
                return True
        except:
            print(f"Failed to open: Object '{object_name}' does not support 'Open' state")
            return False
        # 2. 判断 agent 是否在物体旁边（是否满足 "nextto" 条件）。
        if not self.if_robot_close_to_obj(target_object):
            print(f"Failed to open: Robot is not close to object '{object_name}'")
            return False
        # 3. 执行打开操作：
        # 使用 state.set_value(True) 设置状态为 "open"。
        target_object.states[object_states.Open].set_value(True)
        # 返回打开后的状态：state.get_value()。
        print(f'尝试打开{target_object.name},结果为{target_object.states[object_states.Open].get_value()}')
        print(f'尝试打开{target_object.name},结果为{target_object.states[object_states.Open].get_value()}')
        return target_object.states[object_states.Open].get_value()

    def close(self, object_name):
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
        # 2. 判断 agent 是否在物体旁边（是否满足 "nextto" 条件）。
        if not self.if_robot_close_to_obj(target_object):
            print(f"Failed to close: Robot is not close to object '{object_name}'")
            return False
        # 3. 执行关闭操作：
        # 使用 state.set_value(False) 设置状态为 "closed"。
        target_object.states[object_states.Open].set_value(False)
        print(f'尝试关闭{target_object.name},结果为{not target_object.states[object_states.Open].get_value()}')
        # 返回关闭后的状态：state.get_value()。
        return not target_object.states[object_states.Open].get_value()

    def toggle_on(self, object_name):
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
        if not self.if_robot_close_to_obj(target_object):
            print(f"Failed to toggle on: Robot is not close to object '{object_name}'")
            return False
        # 3. 执行打开操作：
        # 使用 state.set_value(True) 设置状态为 "toggled on"。
        target_object.states[object_states.ToggledOn].set_value(True)
        # 返回打开后的状态：state.get_value()。
        print(f'尝试开启{target_object.name},结果为{target_object.states[object_states.ToggledOn].get_value()}')
        return target_object.states[object_states.ToggledOn].get_value()

    def toggle_off(self, object_name):
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
        if not self.if_robot_close_to_obj(target_object):
            print(f"Failed to toggle off: Robot is not close to object '{object_name}'")
            return False
        # 3. 执行关闭操作：
        # 使用 state.set_value(False) 设置状态为 "toggled off"。
        target_object.states[object_states.ToggledOn].set_value(False)
        # 返回关闭后的状态：state.get_value()。
        print(f'尝试关闭{target_object.name},结果为{not target_object.states[object_states.ToggledOn].get_value()}')
        return not target_object.states[object_states.ToggledOn].get_value()

    
    def go_to_room(self, room_name):
        #寻找所有的floor物体
        target_floors = self.env.scene.object_registry("category", 'floors') 
        # 初始化一个变量来存储匹配的对象
        matching_floor = None

        # 遍历 target_floors 集合
        for obj in target_floors:
            # 获取当前对象的 _in_rooms 属性
            rooms = obj.in_rooms
            # 检查目标房间名称是否在当前对象的 _in_rooms 列表中
            if room_name in rooms:
                matching_floor = obj
                break  # 找到匹配的对象后退出循环

        # 检查是否找到匹配的对象
        if matching_floor is not None:
            print(f"找到匹配的对象: {matching_floor.name}")
        else:
            print(f"没有找到包含房间 '{room_name}' 的目标对象")
            return False
        #将agent放置到相应的房间地板上
        self.env.robots[0].states[object_states.OnTop].set_value(matching_floor, True)
        return self.env.robots[0].states[object_states.OnTop].get_value(matching_floor)
    
    
    def heat_object_with_source(self, target_object_name, heatsource_name):
        target_object = self.env.scene.object_registry("name", target_object_name)
        heatsource = self.env.scene.object_registry("name", heatsource_name)

        if object_states.Heated not in target_object.states:
            print("The target object can't be heated")
            return False

        heatsource_df = pd.read_csv('omnigibson/plannerdemo/data/heatSource.csv')
        info = get_info_in_synset(heatsource_df, heatsource_name)
        
        if info is None:
            print("The object isn't heatsource")
            return False
        else:
            heat_source_name = info['synset'].split('.')[0]

        if info['requires_toggled_on']:
            if not heatsource.states[object_states.ToggledOn].get_value():
                print(f"Toggle on the {heat_source_name} first")
                return False
               
        if info['requires_inside']:
            if not target_object.states[object_states.Inside].get_value(heatsource):
                f"Put the object in the {heat_source_name} first"
                return False
        else:
            if (not target_object.states[object_states.OnTop].get_value(heatsource)):
                print(f"Put the object on or near the {heat_source_name} first")
                return False
        
        # if info['requires_closed']:
        #     if heatsource.states[object_states.Open].get_value():
        #         print(f"The {heat_source_name} should be closed")
        #         return False
            
        if not self.if_robot_close_to_obj(heatsource):
            return False

        target_object.states[object_states.Heated].set_value(True)

        return target_object.states[object_states.Heated].get_value()        

    def cook_object_with_tool(self, target_object_name, cooktool_name):
        target_object = self.env.scene.object_registry("name", target_object_name)
        cook_tool = self.env.scene.object_registry("name", cooktool_name)

        if object_states.Cooked not in target_object.states:
            print("The target object can't be cooked")
            return False

        cooktool_df = pd.read_csv('omnigibson/plannerdemo/data/heatSource.csv')
        info = get_info_in_synset(cooktool_df, cooktool_name)
        
        if info is None:
            print("The object isn't cooktool")
            return False
        else:
            cook_tool_name = info['synset'].split('.')[0]

        if info['requires_toggled_on']:
            if not cook_tool.states[object_states.ToggledOn].get_value():
                print(f"Toggle on the {cook_tool_name} first")
                return False
               
        if info['requires_inside']:
            if not target_object.states[object_states.Inside].get_value(cook_tool):
                print(f"Put the object in the {cook_tool_name} first")
                return False
        else:
            if (not target_object.states[object_states.OnTop].get_value(cook_tool)):
                print(f"Put the object on or near the {cook_tool_name} first")
                return False
        
        # if info['requires_closed']:
        #     if cook_tool.states[object_states.Open].get_value():
        #         print(f"The {cook_tool_name} should be closed")
        #         return False
            
        if not self.if_robot_close_to_obj(cook_tool):
            return False

        target_object.states[object_states.Cooked].set_value(True)

        return target_object.states[object_states.Cooked].get_value()

    def froze_object_with_source(self, target_object_name, coldsource_name):
        target_object = self.env.scene.object_registry("name", target_object_name)
        coldsource = self.env.scene.object_registry("name", coldsource_name)

        if object_states.Frozen not in target_object.states:
            print("The target object can't be frozen")
            return False

        coldsource_df = pd.read_csv('omnigibson/plannerdemo/data/coldSource.csv')
        info = get_info_in_synset(coldsource_df, coldsource_name)
        
        if info is None:
            print("The object isn't coldsource")
            return False
        else:
            cold_source_name = info['synset'].split('.')[0]

        if info['requires_toggled_on']:
            if not coldsource.states[object_states.ToggledOn].get_value():
                print(f"Toggle on the {cold_source_name} first")
                return False
               
        if info['requires_inside']:
            if not target_object.states[object_states.Inside].get_value(coldsource):
                print(f"Put the object in the {cold_source_name} first")
                return False
        else:
            if (not target_object.states[object_states.OnTop].get_value(coldsource)):
                print(f"Put the object on or near the {cold_source_name} first")
                return False
        
        # if info['requires_closed']:
        #     if coldsource.states[object_states.Open].get_value():
        #         print(f"The {cold_source_name} should be closed")
        #         return False
            
        if not self.if_robot_close_to_obj(coldsource):
            return False

        target_object.states[object_states.Frozen].set_value(True)

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

        # Parse LLM plan
        if isinstance(llm_plan, str):
            try:
                llm_plan = json.loads(llm_plan)
                llm_plan_str = llm_plan.get('llm_plans', llm_plan)
            except json.JSONDecodeError as e:
                print(f"Warning: Failed to parse LLM plan as JSON: {e}")
                llm_plan_str = llm_plan
                llm_plan = {}
        else:
            llm_plan_str = llm_plan.get('llm_plans', '')
        
        try:
            llm_plan["structured_plan"] = self.parse_plan(llm_plan_str)
        except Exception as e:
            print(f"Error parsing plan: {e}")
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

            # Build object dictionary
            object_dict = {index: key for index, key in enumerate(self.sample_2d_cache["nextto"].keys())}
            print(f"Object dictionary created with {len(object_dict)} objects")

            structured_plan = llm_plan.get('structured_plan')
            if not structured_plan:
                print("No structured plan found, returning default failure")
                sample_data.append({
                    "action_sequence": None,
                    "reward": -1,
                    "llm_plan": llm_plan,
                    "done": False
                })
                return sample_data

            action_sequence = structured_plan.get('action_sequence', [])
            step_count = len(action_sequence)
            print(f"Executing {step_count} actions in sequence")

            # Execute actions
            for i, action in enumerate(action_sequence, 1):
                print(f"\nStep {i}: Executing action {action}")
                result = self.take_action(action, object_dict)
                if not result:
                    print(f"Step {i}: Action failed, terminating sequence")
                    break
                print(f"Step {i}: Action succeeded")

            # Check completion and compute rewards
            done, termination_dict = self.check_done(init_action)
            reward, reward_dict = self.get_reward(init_action)
            step_penalty = step_count * 0.01
            related_reward = self.related_reward(action_sequence, object_dict)
            
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
            
            
    def place(self, obj_idx, predicate, object_dict):
        if "grasped" not in self.state_dict:
            print("Not grasped")
            return False
        else:
            obj_in_hand_name = self.state_dict["grasped"]
            obj_name = object_dict[obj_idx]
            obj_in_hand = self.env.scene.object_registry("name", obj_in_hand_name)
            obj = self.env.scene.object_registry("name", obj_name)

            robot_position = self.env.robots[0].get_position()
            obj_position = obj.get_position()
            distance = torch.norm(robot_position[:2] - obj_position[:2])
            _,_,box_extent = self.get_relative_aabb_information(obj)
            robot_extent = self.env.robots[0].aabb_extent[:2]
            if distance > 3.5 + (1)*max(box_extent) + max(robot_extent):
                print(f"Too far from the object, the distance is {distance:.2f}m, the available distance is {3 + (1)*max(box_extent)+max(robot_extent)}m.")
                return False
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
                if obj_in_hand_name in self.sample_2d_cache[predicate]:
                    position_and_orientation = self.sample_2d_cache[predicate][obj_in_hand_name][obj_name]
                    obj_in_hand.set_position_orientation(position = position_and_orientation[0], orientation = position_and_orientation[1])
                else:
                    print("can not place correctly, no sampled pose")
                    return False
                self.sample_2d_cache["nextto"][obj_in_hand_name] = self.sample_2d_cache["nextto"][obj_name]
                self.state_dict.pop("grasped")
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
            except:
                print("have problem when placing")
                return False
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