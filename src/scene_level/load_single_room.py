import sys

import omnigibson as og
from omnigibson.macros import gm
from omnigibson.utils.asset_utils import get_available_g_scenes, get_available_og_scenes
from omnigibson.utils.ui_utils import choose_from_options
from camera_utils import camera_for_scene_room
# Configure macros for maximum performance
gm.USE_GPU_DYNAMICS = True
gm.ENABLE_FLATCACHE = True

import os
import re
import json
import torch as th
from omnigibson.objects.dataset_object import DatasetObject
from utils import read_position_config,update_position_config,quat2euler,euler2quat
import numpy as np
import matplotlib.pyplot as plt
import base64
from PIL import Image, ImageDraw
import io
from ModifyAgent import ObjectPositionModifyAgent
import time

def main(random_selection=False, headless=False, short_exec=False):
    """
    Prompts the user to select any available interactive scene and loads a turtlebot into it.
    It steps the environment 100 times with random actions sampled from the action space,
    using the Gym interface, resetting it 10 times.
    """
    og.log.info(f"Demo {__file__}\n    " + "*" * 80 + "\n    Description:\n" + main.__doc__ + "*" * 80)

    # Choose the scene type to load
    scene_options = {
        "InteractiveTraversableScene": "Procedurally generated scene with fully interactive objects",
        # "StaticTraversableScene": "Monolithic scene mesh with no interactive objects",
    }
    scene_type = choose_from_options(options=scene_options, name="scene type", random_selection=random_selection)

    # Choose the scene model to load
    scenes = get_available_og_scenes() if scene_type == "InteractiveTraversableScene" else get_available_g_scenes()
    scene_model = choose_from_options(options=scenes, name="scene model", random_selection=random_selection)

    # cfg = {
    #     "scene": {
    #         "type": scene_type,
    #         "scene_model": scene_model,
    #     },
    #     "robots": [
    #         {
    #             "type": "Turtlebot",
    #             "obs_modalities": ["scan", "rgb", "depth"],
    #             "action_type": "continuous",
    #             "action_normalize": True,
    #         },
    #     ],
    # }
    cfg = {
        "scene": {
            "type": scene_type,
            "scene_model": scene_model,
            "not_load_object_categories": ["roof", "downlight", "room_ventilator", "square_light", "fire_alarm", "range_hood", "room_light", "fire_sprinkler", "signpost"]
        
        } 
    }
    cfg['objects'] = [
        {
            "type": "LightObject",
            "name": "Light_test",
            "light_type": "Distant",
            "intensity": 514,
            "radius": 10,
            "position": [0, 0, 30],
        }
    ]
    
    
    # If the scene type is interactive, also check if we want to quick load or full load the scene
    # if scene_type == "InteractiveTraversableScene":
    #     load_options = {
    #         "Quick": "Only load the building assets (i.e.: the floors, walls, ceilings)",
    #         "Full": "Load all interactive objects in the scene",
    #     }
    #     load_mode = choose_from_options(options=load_options, name="load mode", random_selection=random_selection)
    #     if load_mode == "Quick":
    #         cfg["scene"]["load_object_categories"] = ["floors", "walls", "ceilings"]
    load_mode = "Full"
    #create a folder to save the images
    # scene 1 example:
    # suggestions = "Move the bottom_cabinet_qacthv_0 close to the shelf_owvfik_1"
    # target_name = "bottom_cabinet_qacthv_0"
    suggestions = "Move the bottom_cabinet_qacthv_0 up a bit in the picture "
    target_name = "bottom_cabinet_qacthv_0"
    # scene 17 example:
    # suggestions = "Move the shelf_owvfik_0 close to the bottom_cabinet_jhymlr_0,don't let it collide with the bottom_cabinet_jhymlr_0"
    # target_name = "shelf_owvfik_0"
    # suggestions = "Move the shelf_njwsoa_1 to the position on the opposite wall and reverse its orientation,don't let it collide with the wall"
    # target_name = "shelf_njwsoa_1"
    included_obj_type = re.findall(r'\b([a-zA-Z]+(?:_[a-zA-Z]+)*)_[a-zA-Z0-9]+_\d+', suggestions)
    included_obj = re.findall(r'\b[a-zA-Z]+(?:_[a-zA-Z0-9]+)+_\d+\b', suggestions)
    suggestions = re.sub(r'[<>:"/\\|?*\s]', '_', suggestions)  # 替换特殊字符
    base_path = "./overhead_image_withbbox3"
    path = os.path.join(base_path, scene_model, suggestions)
    os.makedirs(path, exist_ok=True)
    # Load the ceilings info to get the center of the room easily
    json_name = "omnigibson/data/og_dataset/scenes/"+scene_model+"/json/" + scene_model + "_best.json"
    with open(json_name, 'r') as file:
        data = json.load(file)
    objects = data["state"]["object_registry"]
    room_info = data["objects_info"]["init_info"]
    all_obj_type = []
    for key in objects.keys():
        match = re.match(r'([a-zA-Z]+(?:_[a-zA-Z]+)*)_.*_\d+', key)
        if match:
            all_obj_type.append(match.group(1))
    not_included_obj_type = list(set(all_obj_type) - set(included_obj_type))
    
    
    ceiling_list = []

    for key in objects.keys():
        if "ceilings" in key:
            model = key.split('_')[-2]
            pos = objects[key]["root_link"]["pos"]

            room = room_info[key]["args"]["in_rooms"]
            if room != "":
                ceiling_list.append((pos, room[0], model))

    # Load the environment
    
    env = og.Environment(configs=cfg)
    ceilings = env.scene.object_registry("category", "ceilings")
    for ceiling in ceilings:
        ceiling.visible = False
    # objs = [obj for obj in env.scene.objects if obj.category != 'light']
    # obj_models = [obj.model for obj in objs]
    # obj_bbox = [obj.native_bbox for obj in objs]
    # bdict = dict(zip(obj_models, obj_bbox))
    # obj_dict = {obj.model: obj for obj in objs}
    # # Allow user to move camera more easily
    # if not gm.HEADLESS:
    #     og.sim.enable_viewer_camera_teleoperation()
    
    # og.sim.viewer_camera.image_height = 720
    # og.sim.viewer_camera.image_width = 1280
    # for s in ["rgb", "bbox_2d_tight", "bbox_2d_loose", "bbox_3d", "seg_semantic", "seg_instance", "seg_instance_id"]:
    #     og.sim.viewer_camera.add_modality(s)
    # og.sim.viewer_camera.focal_length = 15
    # Run a simple loop and reset periodically
    
    

    which_room_to_load = "living_room_1"
    center = [item[0] for item in ceiling_list if item[1] == which_room_to_load]
    center[0][2] +=5
    og.sim.viewer_camera.set_position_orientation(center[0], [0, 0, -0.707, 0.707])

    
    ModifyAgent = ObjectPositionModifyAgent()
    
    obj_config = env.scene.object_registry.get_dict("name")
    # import pdb;pdb.set_trace()
    # print("obj_config",obj_config)
    all_obj = [k for k,v in obj_config.items()]
    not_included_obj = list(set(all_obj) - set(included_obj))
    obj_inroom_config = {}
    if which_room_to_load not in obj_inroom_config:
        obj_inroom_config[which_room_to_load] = {}

    
    for j in range(100):
        og.log.info("Resetting environment")
        env.reset()
        
        for i in range(100):
            action = env.action_space.sample()
            state, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                og.log.info("Episode finished after {} timesteps".format(i + 1))
                break
        
    
    og.clear()

if __name__ == "__main__":
    main()
