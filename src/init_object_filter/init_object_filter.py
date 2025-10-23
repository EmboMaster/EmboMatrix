import json
import os
import sys
from src.llm_selection import get_gpt_response_by_request
from src.utils.config_loader import config
def get_scene_room_init_obj(scene_name="Beechwood_0_garden", room_name="kitchen_0"):
    # load best.json and get room information
    best_json_path = f"omnigibson/data/og_dataset/scenes/{scene_name}/json/{scene_name}_best.json"
    best_origin_json_path = f"omnigibson/data/og_dataset/scenes/{scene_name}/json/{scene_name}_best_origin.json"
    if os.path.exists(best_origin_json_path):
        with open(best_origin_json_path, "r") as file:
            data = json.load(file)
    else:
        with open(best_json_path, "r") as file:
            data = json.load(file)

    init_info = data['objects_info']['init_info']
    room_objects = {}
    for obj_name, obj_info in init_info.items():
        if 'in_rooms' in obj_info['args'].keys() and room_name in obj_info['args']['in_rooms']:
            room_objects[obj_name] = obj_info

    return room_objects

def init_obj_filter(official_api,room_objects_name, room_image_path_list, room_name):

    exclude_obj_list = ['walls', 'floors', 'window', 'door']
    exclude_room_objects_name = [obj_name for obj_name in room_objects_name if any(exclude_obj in obj_name for exclude_obj in exclude_obj_list)]
    room_objects_name = [obj_name for obj_name in room_objects_name if obj_name not in exclude_room_objects_name]


    prompt = f"""
    You are a helpful assistant as a home decorator and have been asked to decide which objects in the room are convient and worth to change the positions. The inputs include (1) a list of objects in the room. Please generate a list of objects that you think need to move.

    There are some rules:
    1. if the object is big and heavy, it can not be moved.
    2. you need to move at least two objects but not move no more than 5 objects in the room.
    3. do not move floors, doors, windows, walls, ceilings, roofs, or other objects fixed with the above objects like electric_switches, electric_outlets, etc.
    4. the length of output list is at least 2 but no more than 5.
    5. if there are no suitable objects to move, just return an empty list.
    6. do not add any explanations or other information, just list the objects that need to move.
    

    ### example output:
    ['countertop_jveutp_0']
    
    The objects in the room are: {room_objects_name}

    ### Output:
    """

    from openai import OpenAI
    import base64
    use_official_api = official_api
    model = config['scene_generation']['model']
    api_key = config['scene_generation']['api_key']
    base_url = config['scene_generation']['base_url']

    # 准备消息内容
    messages = [{
        "role": "user",
        "content": [{
            "type": "text",
            "text": prompt
        }]
    }]

    # gpt_exclude_obj_list = response.choices[0].message.content.strip()
    gpt_exclude_obj_list = get_gpt_response_by_request(model=model, api_key=api_key,base_url = base_url, messages=messages, image_paths = room_image_path_list, max_try=3, use_official_api=use_official_api)
    print(f"{room_name} moved objects: ", gpt_exclude_obj_list)
    # 使用正则表达式提取列表部分
    import ast, re
    match = re.search(r"\[(.*?)\]", gpt_exclude_obj_list)
    if match:
        list_str = match.group(0)  # 包括方括号
        try:
            # 使用 ast.literal_eval 安全解析列表字符串
            extracted_list = ast.literal_eval(list_str)
        except (ValueError, SyntaxError):
            raise ValueError("无法解析列表内容")
    else:
        raise ValueError("未找到列表")

    real_exclude_obj_list = [obj_name for obj_name in room_objects_name if obj_name not in extracted_list]
    exclude_room_objects_name.extend(real_exclude_obj_list)

    return exclude_room_objects_name


def step1_main(official_api,scene_name="Beechwood_0_garden", room_name="kitchen_0", room_image_path_list = [], skip_step_1 = False):

    room_objects = get_scene_room_init_obj(scene_name, room_name)
    room_objects_name = room_objects.keys()

    if skip_step_1:
        exclude_obj_list = room_objects_name
    else:
        exclude_obj_list = init_obj_filter(official_api,room_objects_name, [], room_name)
    return room_objects_name, exclude_obj_list
    

if __name__ == "__main__":
    scene_name = "Beechwood_0_garden"
    room_name = "kitchen_0"
    initial_room_image_path_list = [f"omnigibson/data/camera/initial_bbox_objectname/{scene_name}/{room_name}_1_top_view_.png", f"omnigibson/data/camera/initial_bbox_objectname/{scene_name}/{room_name}_1_front_view_.png"]
    room_objects_name, exclude_obj_list = step1_main(scene_name, room_name, initial_room_image_path_list)
    print(exclude_obj_list)
