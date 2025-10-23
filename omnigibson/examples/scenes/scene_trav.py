import sys
sys.path.insert(0, '/GPFS/rhome/yuanzhuoding/planner-bench/planner_bench')
import omnigibson as og
from omnigibson.macros import gm
from omnigibson.utils.asset_utils import get_available_g_scenes, get_available_og_scenes
from omnigibson.utils.ui_utils import choose_from_options
from camera_utils import camera_for_scene_room 
import json
import copy
# Configure macros for maximum performance
gm.USE_GPU_DYNAMICS = True
gm.ENABLE_FLATCACHE = True
gm.ENABLE_OBJECT_STATES = False
gm.ENABLE_TRANSITION_RULES = False
import os
import re
import json
import torch as th
from omnigibson.objects.dataset_object import DatasetObject
from camera_utils import camera_for_scene_room
import numpy as np
import matplotlib.pyplot as plt
import base64
from PIL import Image, ImageDraw
import io
import time
import argparse
def main(scene_model,random_selection=False, headless=False, short_exec=False):
    """
    Prompts the user to select any available interactive scene and loads a turtlebot into it.
    It steps the environment 100 times with random actions sampled from the action space,
    using the Gym interface, resetting it 10 times.
    """
    og.log.info(f"Demo {__file__}\n    " + "*" * 80 + "\n    Description:\n" + main.__doc__ + "*" * 80)
     
    result = dict()
    cfg = {
        "scene": {
            "type": "InteractiveTraversableScene",
            "scene_model": scene_model,
            "not_load_object_categories": ["roof", "downlight", "room_ventilator", "square_light", "fire_alarm", "range_hood", "room_light", "fire_sprinkler", "signpost"],
            
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
    json_name = "omnigibson/data/og_dataset/scenes/"+scene_model+"/json/" + scene_model + "_best.json"
    with open(json_name, 'r') as file:
        data = json.load(file)
    objects = data["state"]["object_registry"]
    room_info = data["objects_info"]["init_info"]
    ceiling_list = []

    for key in objects.keys():
        if "ceilings" in key:
            model = key.split('_')[-2]
            pos = objects[key]["root_link"]["pos"]

            room = room_info[key]["args"]["in_rooms"]
            if room != "":
                ceiling_list.append((pos, room[0], model))

    room_list = set([item[1] for item in ceiling_list])

    # Load the environment
    env = og.Environment(configs=cfg)
    # ceilings = env.scene.object_registry("category", "ceilings")
    # for ceiling in ceilings:
    #     ceiling.visible = False
  
    if not gm.HEADLESS:
        og.sim.enable_viewer_camera_teleoperation()


    # Run a simple loop and reset periodically
   
    # for scene_model in scenes:
        
    # env.reload_model(scene_model)
    result[scene_model] = dict()
    for room in room_list: 
        og.log.info("Resetting environment")
        env.reset()
        for i in range(100):
            action = env.action_space.sample()
            state, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                og.log.info("Episode finished after {} timesteps".format(i + 1))
                break
        bbox ,_ , area= camera_for_scene_room(scene_model, room, "top_view", "bbox_2d_tight", uninclude_list=["floor"], env=env, image_height=1440, image_width=2560, focal_length=14)
        # obs, info = og.sim.viewer_camera.get_obs()
        # rgb = obs['rgb'].numpy()
        result[scene_model][room] = area
        
     # 读取现有的 area.json 文件内容
    area_file_path = "omnigibson/examples/scenes/overhead_view/area.json"
    if os.path.exists(area_file_path):
        with open(area_file_path, 'r') as f:
            try:
                existing_data = json.load(f)
            except json.JSONDecodeError:
                existing_data = {}
    else:
        existing_data = {}

    # 合并现有内容和新结果
    existing_data.update(result)

    # 写回文件
    with open(area_file_path, 'w') as f:
        json.dump(existing_data, f, indent=4)
    og.clear()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run scene_trav.py with different scene models.")
    parser.add_argument("--scene_model", type=str, required=True, help="The scene model to use.")
    args = parser.parse_args()
    main(scene_model=args.scene_model)
