import omnigibson as og
from omnigibson.macros import gm
from omnigibson.utils.asset_utils import get_available_g_scenes, get_available_og_scenes
from omnigibson.utils.ui_utils import choose_from_options
from omnigibson.object_states import AttachedTo, Covered, Draped, Filled, Inside, OnTop, Overlaid, Saturated, Under

# Configure macros for maximum performance

import sys
import os

# 添加项目路径到 sys.path
project_path = "omnigibson"
sys.path.append(project_path)

# 导入函数
from src.init_object_filter.init_object_filter import step1_main
from src.room_creater.room_creater import get_init_obj_pos_rot, load_init_object_from_scene

def scene_creater(official_api,scene_name, result, image_height, image_width, if_reset=False, reset_env=None, skip_step_1 = False, scene_file = None):

    if scene_file == None:
        cfg = {
            "scene": {
                "type": 'InteractiveTraversableScene',
                "scene_model": scene_name,
                "not_load_object_categories": ["roof", "downlight", "room_ventilator", "square_light", "fire_alarm", "range_hood", "room_light", "fire_sprinkler", "signpost"]
            }
        }

        cfg["objects"] = [
            {
                "type": "LightObject",
                "name": "Light_test",
                "light_type": "Distant",
                "intensity": 514,
                "radius": 10,
                "position": [0, 0, 30],
            }
        ]
    else:
        cfg = {
            "scene": {
                "type": 'InteractiveTraversableScene',
                "scene_model": scene_name,
                "scene_file": scene_file,
                "not_load_object_categories": ["roof", "downlight", "room_ventilator", "square_light", "fire_alarm", "range_hood", "room_light", "fire_sprinkler", "signpost"]
            }
        }

    room_all_obj_name_list_dict, unincluded_obj_list_dict = {}, {}
    
    print("Printing room info ...")
    for room in result['rooms']:
        # print(room)
        print(result['rooms'][room])
        room_name = result['rooms'][room]["room_name"]
        initial_room_image_path_list = [f"omnigibson/data/camera/initial_bbox_objectname/{scene_name}/{room}_1_front_view_.png",f"omnigibson/data/camera/initial_bbox_objectname/{scene_name}/{room}_1_top_view_.png"]
        new_added_tree = result['rooms'][room]["new_added_tree"]
        new_added_objects = result['rooms'][room]["new_added_objects"]
        new_added_objects_list = result['rooms'][room]["new_added_objects_list"]
        single_states = result['rooms'][room]["single_obj_state"]

        # step 1: identity which objects don't move
        room_all_obj_name_list, unincluded_obj_list = step1_main(official_api,scene_name, room_name, initial_room_image_path_list, skip_step_1 = skip_step_1)
        room_all_obj_name_list_dict[room_name] = room_all_obj_name_list
        unincluded_obj_list_dict[room_name] = unincluded_obj_list

        init_obj_pos_rot = get_init_obj_pos_rot(scene_name, room_name)

        modified_obj_info = load_init_object_from_scene(scene_model=scene_name, excluded_obj_list=unincluded_obj_list, room_name=room_name)

        if scene_file == None:
            cfg['objects'].extend(new_added_objects_list)

        # add modified obj
        # for obj in modified_obj_info.keys():
        #     tmp = {
        #         "type": "DatasetObject",
        #         "name": obj,
        #         "category": modified_obj_info[obj]["args"]["category"],
        #         "model": modified_obj_info[obj]["args"]["model"],
        #         "scale": modified_obj_info[obj]["args"]["scale"],
        #         "in_rooms": modified_obj_info[obj]["args"]["in_rooms"],
        #     }
        #     cfg['objects'].append(tmp)

    cfg["render"] = {
        "viewer_width": image_width,
        "viewer_height": image_height,
    }
    # Load the environment
    if not if_reset:

        print(cfg)
        env = og.Environment(configs=cfg)
        
    else:
        env = reset_env
        env.reset()

    return env,room_all_obj_name_list_dict, unincluded_obj_list_dict