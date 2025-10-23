import omnigibson as og
from omnigibson.macros import gm
from omnigibson.utils.asset_utils import get_available_g_scenes, get_available_og_scenes
from omnigibson.utils.ui_utils import choose_from_options

import json
import os
import matplotlib.pyplot as plt
import matplotlib
import collections
import math
import torch as th
import numpy as np
from scipy.spatial.transform import Rotation as R
from PIL import Image, ImageDraw
import numpy as np
import sys
from collections import Counter
from omnigibson.utils.vision_utils import colorize_bboxes_3d


# Configure macros for maximum performance
gm.USE_GPU_DYNAMICS = True
gm.ENABLE_FLATCACHE = True
gm.ENABLE_OBJECT_STATES = False
gm.ENABLE_TRANSITION_RULES = False


def main(random_selection=False, headless=False, short_exec=False):
    """
    Prompts the user to select any available interactive scene and loads a turtlebot into it.
    It steps the environment 100 times with random actions sampled from the action space,
    using the Gym interface, resetting it 10 times.
    """
    scenes = get_available_og_scenes()
    scene_id = int(sys.argv[1])
    scene_model = scenes[scene_id - 1]

    cfg = {
        "scene": {
            "type": "InteractiveTraversableScene",
            "scene_model": scene_model,
            "not_load_object_categories": ["roof", "downlight", "room_ventilator", "square_light", "fire_alarm", "range_hood", "room_light", "fire_sprinkler", "signpost"]
        },
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

    # Load the environment
    env = og.Environment(configs = cfg)

    from omnigibson.utils.camera_utils import camera_for_scene_room
    top_view_outcome, _ = camera_for_scene_room(scene_model=scene_model, room_name='', type="top_view", caption="bbox_2d_tight", uninclude_list=[], env=env, image_height=1080, image_width=1920, focal_length=14, if_all_room=True)

    front_view_outcome, front_view_objs = camera_for_scene_room(scene_model=scene_model, room_name='', type="front_view", caption="bbox_2d_tight", uninclude_list=[], env=env, image_height=1080, image_width=1920, focal_length=14, if_all_room=True)

    direction3 = "/home/magic-amd/embodied/OmniGibson/omnigibson/examples/scenes/initial_bbox_objectname" 
    folder_path3 = os.path.join(direction3, scene_model)
    os.makedirs(folder_path3, exist_ok=True)

    for tmp_room in top_view_outcome.keys():
        matplotlib.use('Agg')
        plt.imshow(top_view_outcome[tmp_room][0])
        plt.axis('off')
        img_name = "/" + tmp_room + "_top_view_" + ".png"
        
        plt.savefig(folder_path3 + img_name, bbox_inches='tight', transparent=True, pad_inches=0, dpi = 300)
        plt.close()


    for tmp_room in front_view_outcome.keys():

        room_imgs = front_view_outcome[tmp_room]
        room_objs = front_view_objs[tmp_room]

        idx = np.argmax(np.array([len(tmp) for tmp in room_objs]))

        matplotlib.use('Agg')
        plt.imshow(room_imgs[idx])
        plt.axis('off')
        img_name = "/" + tmp_room + "_front_view_" + ".png"
        
        plt.savefig(folder_path3 + img_name, bbox_inches='tight', transparent=True, pad_inches=0, dpi = 300)
        plt.close()
    
    og.clear()

if __name__ == "__main__":
    main()