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
# from omnigibson.model.RDT import RDTModel, CustomRDTPolicy
from omnigibson.model.baseline_discrete.baseline_model_IL_grasp import MultiModalRobotPolicy_Discrete
from omnigibson.model.baseline_discrete.IL_trainer_grasp import imitation_learning_trainner

try:
    import gymnasium as gym
    import tensorboard
    import torch as th
    import torch.nn as nn
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback, EvalCallback
    from stable_baselines3.common.evaluation_baseline import evaluate_policy, visualize_result
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
gm.ENABLE_TRANSITION_RULES = False
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
    
    parser.add_argument(
        "--subgoals",
        action="store_true",
        help="If set, the training process will use subgoals reward function",
    )

    # simulator GPU ID
    parser.add_argument(
        "--gpu_id",
        type=int,
        default=6,
        help="GPU ID for the simulator",
    )

    # Network GPU ID
    parser.add_argument(
        "--nn_gpu_id",
        type=int,
        default=7,
        help="GPU ID for the network",
    )

    # Number of transformer layers
    parser.add_argument(
        "--num_transformer_block",
        type=int,
        default=6,
        help="GPU ID for the network",
    )

    # Number of discrete action dimension
    parser.add_argument(
        "--discrete_action_dim",
        type=int,
        default=5,
        help="Number of discrete action dimension",
    )

    args = parser.parse_args()

    # set device id
    gm.GPU_ID = args.gpu_id
    device = f"cuda:{args.nn_gpu_id}"


    tensorboard_log_dir = os.path.join("log_dir", time.strftime("%Y%m%d-%H%M%S"))
    os.makedirs(tensorboard_log_dir, exist_ok=True)
    prefix = ""
    seed = 0

    # Load config
    with open(f"{example_config_path}/fetch_grasp_behavior.yaml", "r") as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)

    # Make sure flattened obs and action space is used
    cfg["env"]["flatten_action_space"] = True
    cfg["env"]["flatten_obs_space"] = True
    # cfg["env"]["action_frequency"] = 5
    # cfg["env"]["rendering_frequency"] = 5
    cfg["env"]["initial_pos_z_offset"] = 0.2
    cfg["scene"]["not_load_object_categories"] = ["door"]

    # Only use RGB obs
    cfg["robots"][0]["obs_modalities"] = ["rgb"]
    
    if args.subgoals:
        cfg["subgoals"] = True

    # If we're not eval, turn off the start / goal markers so the agent doesn't see them
    if not args.eval:
        cfg["task"]["visualize_goal"] = False

    env = og.Environment(configs=cfg)

    # If we're evaluating, hide the ceilings and enable camera teleoperation so the user can easily
    # visualize the rollouts dynamically
    if args.eval:
        # ceiling = env.scene.object_registry("name", "ceilings")
        # ceiling.visible = False
        og.sim.enable_viewer_camera_teleoperation()

    # Set the set
    set_random_seed(seed)
    env.reset()

    os.makedirs(tensorboard_log_dir, exist_ok=True)
    if not args.eval:
        iltrainer = imitation_learning_trainner(args, env)
        iltrainer.train(env, tensorboard_log_dir, device, args)

    
if __name__ == "__main__":
    main()
