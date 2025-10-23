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

    args = parser.parse_args()

    # set device id
    gm.GPU_ID = args.gpu_id
    device = f"cuda:{args.nn_gpu_id}"


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
    
    if args.subgoals:
        cfg["subgoals"] = True

    # If we're not eval, turn off the start / goal markers so the agent doesn't see them
    if not args.eval:
        cfg["task"]["visualize_goal"] = False

    env = og.Environment(configs=cfg)

    # If we're evaluating, hide the ceilings and enable camera teleoperation so the user can easily
    # visualize the rollouts dynamically
    if args.eval:
        ceiling = env.scene.object_registry("name", "ceilings")
        ceiling.visible = False
        og.sim.enable_viewer_camera_teleoperation()

    # Set the set
    set_random_seed(seed)
    env.reset()

    os.makedirs(tensorboard_log_dir, exist_ok=True)

    if args.eval:
        assert args.checkpoint is not None, "If evaluating a PPO policy, @checkpoint argument must be specified!"
        model = PPO.load(args.checkpoint)
        og.log.info("Starting evaluation...")
        mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=50)
        og.log.info("Finished evaluation!")
        og.log.info(f"Mean reward: {mean_reward} +/- {std_reward:.2f}")

    else:
        # model = PPO(
        #     "MultiInputPolicy",
        #     env,
        #     verbose=1,
        #     tensorboard_log=tensorboard_log_dir,
        #     policy_kwargs=policy_kwargs,
        #     n_steps=20 * 10,
        #     batch_size=8,
        #     device="cuda",
        # )
        model = PPO(
            CustomRDTPolicy,
            env,
            verbose=1,
            tensorboard_log=tensorboard_log_dir,
            # policy_kwargs=policy_kwargs,
            n_steps=20 * 10,
            batch_size=4,
            device=device,
            policy_kwargs={"device": device},
        )

        checkpoint_callback = CheckpointCallback(save_freq=1000, save_path=tensorboard_log_dir, name_prefix=prefix)
        eval_callback = EvalCallback(eval_env=env, eval_freq=1000, n_eval_episodes=20)
        # callback = CallbackList([checkpoint_callback, eval_callback])
        callback = CallbackList([checkpoint_callback])

        og.log.debug(model.policy)
        og.log.info(f"model: {model}")

        og.log.info("Starting training...")
        model.learn(
            total_timesteps=10000000,
            callback=callback,
            progress_bar=True
        )
        og.log.info("Finished training!")


if __name__ == "__main__":
    main()
