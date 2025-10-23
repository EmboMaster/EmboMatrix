"""
Example training code using stable-baselines3 PPO for one BEHAVIOR activity.
Note that due to the sparsity of the reward, this training code will not converge and achieve task success.
This only serves as a starting point that users can further build upon.
"""

import argparse
import os
import time
import sys
sys.path.insert(0, './')
import yaml

import omnigibson as og
from omnigibson import example_config_path
from omnigibson.macros import gm
from omnigibson.utils.python_utils import meets_minimum_version
from omnigibson.model.RDT import RDTModel, CustomRDTPolicy
import re
import json


try:
    import gymnasium as gym
    import tensorboard
    import torch as th
    import torch.nn as nn
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback, EvalCallback
    from stable_baselines3.common.evaluation import evaluate_policy
    from stable_baselines3.common.preprocessing import maybe_transpose
    from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
    from stable_baselines3.common.utils import set_random_seed

except ModuleNotFoundError:
    og.log.error(
        "torch, stable-baselines3, or tensorboard is not installed. "
        "See which packages are missing, and then run the following for any missing packages:\n"
        "pip install stable-baselines3[extra]\n"
        "pip install tensorboard\n"
        "pip install shimmy>=0.2.1\n"
        "Also, please update gym to >=0.26.1 after installing sb3: pip install gym>=0.26.1"
    )
    exit(1)

assert meets_minimum_version(gym.__version__, "0.28.1"), "Please install/update gymnasium to version >= 0.28.1"

# We don't need object states nor transitions rules, so we disable them now, and also enable flatcache for maximum speed
gm.ENABLE_OBJECT_STATES = True
gm.ENABLE_TRANSITION_RULES = True
gm.ENABLE_FLATCACHE = True
gm.USE_GPU_DYNAMICS=True
gm.ENABLE_FLATCACHE=False

def main():
    # Parse args
    parser = argparse.ArgumentParser(description="Train or evaluate a PPO agent in BEHAVIOR")

    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Absolute path to desired PPO checkpoint to load for evaluation",
    )

    parser.add_argument(
        "--eval",
        action="store_true",
        help="If set, will evaluate the PPO agent found from --checkpoint",
    )

    args = parser.parse_args()
    tensorboard_log_dir = os.path.join("log_dir", time.strftime("%Y%m%d-%H%M%S"))
    os.makedirs(tensorboard_log_dir, exist_ok=True)
    prefix = ""
    seed = 0

    # Load config
    with open(f"{example_config_path}/fetch_behavior_recycling_office_papers.yaml", "r") as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)

    # Make sure flattened obs and action space is used
    cfg["env"]["flatten_action_space"] = True
    cfg["env"]["flatten_obs_space"] = True

    # Only use RGB obs
    cfg["robots"][0]["obs_modalities"] = ["rgb"]

    # If we're not eval, turn off the start / goal markers so the agent doesn't see them
    if not args.eval:
        cfg["task"]["visualize_goal"] = False

    env = og.Environment(configs=cfg)
    
    print("load environment successful")
    # load subgoals bddl files
    task_name = cfg['task']['activity_name']
    bddl_subgoals_path = f"./bddl_subgoals/bddl_subgoals/activity_definitions/{task_name}/problem0.bddl" 
    subgoals_list = load_subgoal_bddl(bddl_subgoals_path)
    scene_json = json.load(open(env._scene.scene_file))
    inst_to_name = scene_json["metadata"]["task"]["inst_to_name"]
    for subgoal in subgoals_list:
        instruction = subgoal["instruction"]
        relation = subgoal["relation"]
        objects = subgoal["objects"]
        objects_name = [inst_to_name[obj] for obj in objects]
        objects_info = env.scene.get_objects_info()
        print(f"Instruction: {instruction}")
        print(f"Relation: {relation}")
        print(f"Objects: {objects_name}")
        if relation in ['NextTo', 'InFrontOf', 'Behind']:
            type = "navigation"
        else:
            type = "grasp"

def load_subgoal_bddl(file_path):
    """
    Load a BDDL subgoal file and parse it into a list of subgoal dictionaries.

    Args:
        file_path (str): Path to the BDDL file.

    Returns:
        list[dict]: A list of subgoals, each represented as a dictionary with keys
                    {"instruction": str, "relation": str, "objects": list}.
    """
    subgoals = []

    # Open and read the BDDL file
    with open(file_path, 'r') as file:
        content = file.read()

    # Use regular expressions to find subgoal definitions
    subgoal_pattern = re.compile(
        r"\(:subgoal\d+\s+(.*?)\.\s*\((\w+)\s+([^\)]+)\)",  # Matches instruction, relation, and objects
        re.DOTALL  # Enables multi-line matching
    )
    matches = subgoal_pattern.findall(content)

    # Parse each subgoal
    for match in matches:
        instruction = match[0].strip()
        relation = match[1].strip()
        objects = [obj.strip() for obj in match[2].split()]

        # Append the parsed subgoal as a dictionary
        subgoals.append({
            "instruction": instruction,
            "relation": relation,
            "objects": objects
        })

    return subgoals


if __name__ == "__main__":
    main()
