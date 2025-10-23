# from xvfbwrapper import Xvfb

# vdisplay = Xvfb()
# vdisplay.start()
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
    #scene_type = choose_from_options(options=scene_options, name="scene type", random_selection=random_selection)
    scene_type = "InteractiveTraversableScene"
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

    # If the scene type is interactive, also check if we want to quick load or full load the scene
    # if scene_type == "InteractiveTraversableScene":
    #     load_options = {
    #         "Quick": "Only load the building assets (i.e.: the floors, walls, ceilings)",
    #         "Full": "Load all interactive objects in the scene",
    #     }
    #     load_mode = choose_from_options(options=load_options, name="load mode", random_selection=random_selection)
    #     if load_mode == "Quick":
    #         cfg["scene"]["load_object_categories"] = ["floors", "walls", "ceilings"]
    
    
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
    which_room_to_load = "living_room_0"
    center = [item[0] for item in ceiling_list if item[1] == which_room_to_load]
    center[0][2] +=5
    room_list = set([item[1] for item in ceiling_list])

    # Load the environment
    env = og.Environment(configs=cfg)
    # ceilings = env.scene.object_registry("category", "ceilings")
    # for ceiling in ceilings:
    #     ceiling.visible = False
    # # Allow user to move camera more easily
    if not gm.HEADLESS:
        og.sim.enable_viewer_camera_teleoperation()
    og.sim.viewer_camera.focal_length = 15

    # Run a simple loop and reset periodically
    max_iterations = 100
    # og.sim.viewer_camera.set_position_orientation([30.1026, 16.7447,  2.1116], [0.5, 0.5, 0.4999999999999999, 0.5000000000000001])

    import pdb;pdb.set_trace()
    
    og.sim.viewer_camera.set_position_orientation(center[0], [0, 0, -0.707, 0.707])
    for room in room_list:
        
        
        center = [item[0] for item in ceiling_list if item[1] == room]
        center[0][2] +=5
        og.sim.viewer_camera.set_position_orientation(center[0], [0, 0, -0.707, 0.707])
        
        og.log.info("Resetting environment")
        env.reset()
        for i in range(100):
            action = env.action_space.sample()
            state, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                og.log.info("Episode finished after {} timesteps".format(i + 1))
                break
        bbox = camera_for_scene_room(scene_model, which_room_to_load, "top_view", "bbox_2d_tight", uninclude_list=["floor"], env=env, image_height=1440, image_width=2560, focal_length=14)
        # obs, info = og.sim.viewer_camera.get_obs()
        # rgb = obs['rgb'].numpy()
        import pdb;pdb.set_trace()
        image = Image.fromarray(bbox[0][0].astype('uint8'))
        # 将图像保存到内存中的字节流
        save_path = path
        # 定义文件名
        file_name = f"omnigibson/examples/scenes/overhead_view/overhead_image_iteration{room}.png"
        full_path = os.path.join(save_path, file_name)

        # 保存图像
        image.save(full_path)
    # Always close the environment at the end
    og.clear()

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


if __name__ == "__main__":
    main()
# vdisplay.stop()