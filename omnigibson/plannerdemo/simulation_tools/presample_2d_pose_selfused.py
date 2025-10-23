import json
import os
import time
import torch
import ast
import argparse
import yaml
import sys
sys.path.append("/GPFS/rhome/yuanzhuoding/planner-bench/planner_bench/")
import omnigibson as og
import re
import math
from omnigibson.action_primitives.starter_semantic_action_primitives import StarterSemanticActionPrimitives
from omnigibson.object_states import Inside, OnTop
from tqdm import tqdm

def extract_task(s):
    # Use regular expression to find the content between 'task_' and '_0'
    match = re.search(r'task_(.*?)_0', s)
    if match:
        return match.group(1)
    return None
def ensure_json_serializable(obj, path="root"):
    """
    递归确保对象obj可被json.dumps()直接序列化：
    - 如果检测到不可序列化的类型，就尝试做特殊转换（比如torch.Tensor -> list）；
    - 如果仍无法转换，则抛出TypeError。
    
    函数最终返回一个“已处理”或“可直接被json.dumps”的新对象。
    """
    # None, bool, int, float, str 都是可直接序列化的
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj

    # torch.Tensor -> list
    if isinstance(obj, torch.Tensor):
        return obj.tolist()
    
    # list 或 tuple
    if isinstance(obj, (list, tuple)):
        return [ensure_json_serializable(item, path=f"{path}[{idx}]")
                for idx, item in enumerate(obj)]

    # dict
    if isinstance(obj, dict):
        new_dict = {}
        for k, v in obj.items():
            # 如果键不是字符串，在严格模式下也会有问题
            if not isinstance(k, str):
                raise TypeError(
                    f"Key '{k}' at '{path}' is not a string, not JSON-serializable by default."
                )
            new_dict[k] = ensure_json_serializable(v, path=f"{path}['{k}']")
        return new_dict

    # 如果走到这里，说明仍是其他类型，且我们没有提供转换逻辑
    raise TypeError(f"Object of type {type(obj)} at '{path}' is not JSON serializable.")

def safe_dump_to_json(cache_dict, filename, if_replace=True):
    # 先对整个数据做可序列化处理
    cache_dict_serializable = ensure_json_serializable(cache_dict)
    # 再把处理后的结果dump到文件
    if if_replace:
        filename = filename.replace("omnigibson/","/GPFS/rhome/yuanzhuoding/planner-bench/planner_bench/omnigibson/")

    parent_dir = os.path.dirname(filename)
    
    # 检查文件夹是否存在
    if not os.path.exists(parent_dir):
        # 如果不存在，创建文件夹（包括必要的父目录）
        os.makedirs(parent_dir)
        print(f"Created directory: {parent_dir}")
    else:
        print(f"Directory already exists: {parent_dir}")

    try:
        with open(filename, 'w') as f:
            json.dump(cache_dict_serializable, f)
        print(f"Plan saved successfully to {filename}")
    except:

        import hashlib
        dir_path, long_filename = os.path.split(filename)
        short_filename = f"plan_{hashlib.md5(long_filename.encode()).hexdigest()[:8]}.json"
        short_plan_file = os.path.join(dir_path, short_filename)
        
        # 写入文件
        with open(short_plan_file, 'w') as f:
            json.dump(cache_dict_serializable, f)
        print(f"Plan saved successfully to {short_plan_file}")

def get_object_pose(file_path, object_name):
    """
    从指定路径中提取 scene_name 和 activity_name，找到对应的 sample_pose 文件，
    检查是否存在 object_name 的键，并返回其值（如果非 None），否则返回 None。

    参数:
        file_path (str): 输入的文件路径
        object_name (str): 要查找的对象名称

    返回:
        dict 或 None: object_name 对应的值（如果存在且非 None），否则返回 None
    """
    # 分割路径以提取 scene_name 和 activity_name
    path_parts = file_path.split('/')
    scene_name = path_parts[-2]  # e.g., "Beechwood_0_garden"
    
    # 提取 activity_name（假设是倒数第 2 个部分，去掉 .json 后缀）
    activity_name = path_parts[-1].replace('.json', '')  # e.g., "please_bring_the_loaf_of..."

    # 构造 sample_pose 文件路径
    sample_pose_dir = os.path.join(
        os.path.dirname(os.path.dirname(file_path)),
        "sample_pose",
        scene_name,
        activity_name  # 去掉文件名部分
    )
    sample_pose_file = os.path.join(sample_pose_dir, "sample_pose.txt")

    # 检查 sample_pose 文件是否存在
    if not os.path.isfile(sample_pose_file):
        print(f"Error: sample_pose file not found at {sample_pose_file}")
        return None

    # 读取 JSON 文件
    # try:
    #     with open(sample_pose_file, 'r') as f:
    #         data = json.load(f)
    # except json.JSONDecodeError:
    #     print(f"Error: Failed to decode JSON from {sample_pose_file}")
    #     return None
    # except Exception as e:
    #     print(f"Error: Unable to read {sample_pose_file} - {str(e)}")
    #     return None

    # # 检查 object_name 是否存在且值非 None
    # if object_name in data and data[object_name] is not None:
    #     return data[object_name]
    # else:
    #     print(f"Object '{object_name}' not found or value is None in {sample_pose_file}")
    #     return None

    try:
        # 初始化字典存储解析结果
        data = {}
        
        # 读取 txt 文件
        with open(sample_pose_file, 'r') as f:
            for line in f:
                # 去除空白字符
                line = line.strip()
                if not line:  # 跳过空行
                    continue
                
                # 分割 key 和 value（假设格式为 "key: value"）
                if ': ' in line:
                    key, value = line.split(': ', 1)  # 仅分割第一个 ": "
                    # 使用 ast.literal_eval 安全解析 value（字符串表示的 Python 列表）
                    try:
                        data[key] = ast.literal_eval(value)
                    except (ValueError, SyntaxError) as e:
                        print(f"Error: Failed to parse value for '{key}' in {sample_pose_file} - {str(e)}")
                        continue
        
        # 检查 object_name 是否存在且值非 None
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
    """
    从指定 JSON 文件路径读取 'inst_to_name' 中的物体名称（排除 floor 和 agent），返回列表。
    
    参数:
        json_path (str): JSON 文件的路径
    
    返回:
        list: 包含物体名称的列表
    """
    # 检查文件是否存在
    if not os.path.isfile(json_path):
        raise FileNotFoundError(f"JSON file not found at: {json_path}")
    
    # 读取 JSON 文件
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # 获取 inst_to_name 字典
    try:
        inst_to_name = data["metadata"]["task"]["inst_to_name"]
    except KeyError as e:
        raise KeyError(f"Invalid JSON structure, missing key: {e}")
    
    # 过滤掉 floor 和 agent，提取 value
    exclude_keys = ["floors_", "robot"]
    object_names = []
    key_names = []
    for key, value in inst_to_name.items():
        flag = True
        for tmp_key in exclude_keys:
            if tmp_key in value:
                flag = False
                break
        if flag:
            object_names.append(value)
            key_names.append(key)

    return object_names

def run_experiment(scene_file, gpu_id, mode='train'):
    

    task_description = scene_file.split('/')[-1].replace('.json', '').replace('_', ' ')
    activity_name = scene_file.split('/')[-1].replace('.json', '')
    
    bddl_file = scene_file.replace('.json', '.bddl')
    # bddl_file = f"bddl/bddl/activity_definitions/{activity_name}/problem0.bddl"
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
    
    scene_name = scene_file.split('/')[-2]

    if scene_name == activity_name:
        scene_name = scene_file.split('/')[-3]

    #import pdb; pdb.set_trace()
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
    og.macros.gm.GPU_ID = gpu_id
    og.macros.gm.USE_GPU_DYNAMICS = True
    og.macros.gm.ENABLE_OBJECT_STATES = True
    og.macros.gm.ENABLE_TRANSITION_RULES = False

    # 先看是否需要启动环境
    cache_dict = {}
    task_object = get_task_object_names(scene_file)
    real_task_object = []
    for object_name in task_object:

        query_pose = get_object_pose(scene_file, object_name)
        if query_pose is not None:
            print(f"Found pose for {object_name}: {query_pose}")
            cache_dict[object_name] = query_pose
            continue
        else:
            real_task_object.append(object_name)

    if real_task_object != []:

        # 初始化环境
        env = og.Environment(configs=cfg)
        env.reset()

        index_name_dict = {}
        for k, v in env.scene._scene_info_meta_inst_to_name.items():
            index_name_dict[v] = k
            
        print(cache_dict)
        
        _primitive_controller = StarterSemanticActionPrimitives(env, enable_head_tracking=False)
        for object_name in real_task_object:

            object = env.scene.object_registry("name", object_name)
            sampled_pose_2d = None
            print("Try to sample pose on other object")
            for obj in env.scene.objects:
                if (object.states[Inside].get_value(obj) and obj.name != object.name) or (object.states[OnTop].get_value(obj) and "floors_" not in obj.name):

                    if obj.name in cache_dict.keys():
                        query_pose = cache_dict[obj.name]
                    else:
                        query_pose = get_object_pose(scene_file, obj.name)
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
                        break
            if sampled_pose_2d is not None:
                query_pose = sampled_pose_2d
                print(f"Found pose for {object.name}: {query_pose}")
                cache_dict[object.name] = query_pose
                continue

            if 'floors' not in object.name and 'window' not in object.name and 'wall' not in object.name and 'ceilings' not in object.name:
                cache_dict[object.name] = object
                print(object.name)
                sampled_pose_2d = None
                # while sampled_pose_2d is None:
                for _ in range(2):
                    try:
                        sampled_pose_2d = _primitive_controller._sample_pose_near_object(object, pose_on_obj=None, distance_lo=0.1, distance_hi=1.5, yaw_lo=-math.pi, yaw_hi=math.pi) 
                        print(f"{object.name}: {sampled_pose_2d}")
                        break
                    except:
                        pass
                    
                if sampled_pose_2d is None:
                    print(f"Failed to sample pose for {object.name}! Can not complete the task !")
                    sys.exit(1)
                else:
                    cache_dict[object.name] = sampled_pose_2d
        
        # for object in tqdm(env.scene.objects):

        #     if object.name in task_object:
        #         continue

        #     print(object.name)

        #     query_pose = get_object_pose(scene_file, object.name)
        #     if query_pose is not None:
        #         print(f"Found pose for {object.name}: {query_pose}")
        #         cache_dict[object.name] = query_pose
        #         continue

        #     if 'floors' not in object.name and 'window' not in object.name and 'wall' not in object.name and 'ceilings' not in object.name:
        #         cache_dict[object.name] = object
        #         print(object.name)
        #         sampled_pose_2d = None
        #         # while sampled_pose_2d is None:
        #         for _ in range(2):
        #             try:
        #                 sampled_pose_2d = _primitive_controller._sample_pose_near_object(object, pose_on_obj=None, distance_lo=0.1, distance_hi=0.7, yaw_lo=-math.pi, yaw_hi=math.pi) 
        #                 print(f"{object.name}: {sampled_pose_2d}")
        #                 break
        #             except:
        #                 pass
        #         if sampled_pose_2d is None:
        #             print(f"Failed to sample pose for {object.name}")
        #             cache_dict[object.name] = sampled_pose_2d
        #             continue
        #         else:
        #             cache_dict[object.name] = sampled_pose_2d`
    

    if not os.path.exists(f"omnigibson/plannerdemo/simulation_tools/posecache/{scene_name}"):
        os.makedirs(f"omnigibson/plannerdemo/simulation_tools/posecache/{scene_name}")

    try:
        safe_dump_to_json(
            cache_dict,
            f"omnigibson/plannerdemo/simulation_tools/posecache/{scene_name}/{activity_name}.json",
        )
        print("成功写入 JSON 文件!")
    except TypeError as e:
        print(f"写入失败: {e}")
    

if __name__ == "__main__":
    scene_file = "/GPFS/rhome/yuanzhuoding/planner-bench/planner_bench/omnigibson/shengyin/results0318/Benevolence_1_int/place_basket_on_shelf_shelf_in_corridor.json"
    gpu_id = 2
    run_experiment(scene_file, gpu_id, mode='train')
    # test_data = {
    #     "float_number": 1.23,
    #     "text": "hello",
    #     "tensor_data": torch.tensor([[1,2,3],[4,5,6]]),
    #     "list_of_tensors": [torch.tensor([1,2]), torch.tensor([3,4])]
    # }

    # # 尝试输出到 test.json，观察结果
    # safe_dump_to_json(test_data, "/GPFS/rhome/yuanzhuoding/planner-bench/planner_bench/omnigibson/feasible_scene/test.json")
    # print("JSON dump success!")