import omnigibson as og
from omnigibson.macros import gm
from omnigibson.utils.asset_utils import get_available_g_scenes, get_available_og_scenes
from omnigibson.utils.ui_utils import choose_from_options
import json
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
            "load_room_instances": ["living_room_0"]
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
    
    
    json_name = "/media/magic-4090/44ee543b-82db-4f62-aa8d-c1ad5dd806dc/shengyin/embodied-bench/omnigibson/data/og_dataset/scenes/"+scene_model+"/json/" + scene_model + "_best.json"
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


    # Load the environment
    env = og.Environment(configs=cfg)
    ceilings = env.scene.object_registry("category", "ceilings")
    for ceiling in ceilings:
        ceiling.visible = False
    # # Allow user to move camera more easily
    if not gm.HEADLESS:
        og.sim.enable_viewer_camera_teleoperation()
    og.sim.viewer_camera.focal_length = 15

    # Run a simple loop and reset periodically
    max_iterations = 100
    
    og.sim.viewer_camera.set_position_orientation(center[0], [0, 0, -0.707, 0.707])
    for j in range(max_iterations):
        og.log.info("Resetting environment")
        env.reset()
        for i in range(100):
            action = env.action_space.sample()
            state, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                og.log.info("Episode finished after {} timesteps".format(i + 1))
                break

    # Always close the environment at the end
    og.clear()

if __name__ == "__main__":
    main()