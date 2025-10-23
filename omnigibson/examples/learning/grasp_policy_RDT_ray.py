"""
Example training code using stable-baselines3 PPO for one BEHAVIOR activity.
Note that due to the sparsity of the reward, this training code will not converge and achieve task success.
This only serves as a starting point that users can further build upon.
"""

import argparse
import os
import time

import yaml

import omnigibson as og
from omnigibson import example_config_path
from omnigibson.macros import gm
from omnigibson.utils.python_utils import meets_minimum_version
from omnigibson.model.RDT_ray import RDTrayTrainner


try:
    import gymnasium as gym
    import tensorboard
    import torch as th
    import torch.nn as nn
    # from stable_baselines3 import PPO
    # from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback, EvalCallback
    # from stable_baselines3.common.evaluation import evaluate_policy
    # from stable_baselines3.common.preprocessing import maybe_transpose
    # from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
    # from stable_baselines3.common.utils import set_random_seed

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

import ray
from ray.rllib.algorithms.ppo import PPOConfig
from ray import tune

from ray.tune.registry import register_env
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
import torch
import torch.nn as nn
from ray.rllib.models import ModelCatalog

def behavior_1k_env_creator(config):
    return og.Environment(configs=config)

register_env("behavior_1k_env", behavior_1k_env_creator)
ModelCatalog.register_custom_model("large_scale_diffusion_model", RDTrayTrainner)

# We don't need object states nor transitions rules, so we disable them now, and also enable flatcache for maximum speed
gm.ENABLE_OBJECT_STATES = True
gm.ENABLE_TRANSITION_RULES = True
gm.ENABLE_FLATCACHE = True


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
    with open(f"{example_config_path}/tiago_primitives.yaml", "r") as f:
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

    # If we're evaluating, hide the ceilings and enable camera teleoperation so the user can easily
    # visualize the rollouts dynamically
    if args.eval:
        ceiling = env.scene.object_registry("name", "ceilings")
        ceiling.visible = False
        og.sim.enable_viewer_camera_teleoperation()

    env.reset()

    os.makedirs(tensorboard_log_dir, exist_ok=True)

    ray.init()
    # model = RDTrayTrainner()
    # 配置训练的超参数
    ppo_config = (
        PPOConfig()
        .environment(
            env="behavior_1k_env",
            env_config={"configs": f"{example_config_path}/tiago_primitives.yaml"},
        )
        .framework("torch")
        .rollouts(
            num_env_runners=7,  # 使用多个工作线程来并行处理环境
            rollout_fragment_length=200
        )
        .training(
            train_batch_size=4,
            minibatch_size=2,
            num_sgd_iter=10,
            model={"custom_model": "RDTrayTrainner"}
        )
        .resources(
            num_gpus=8  # 使用 8 张 GPU
        )
    )
    
    
    trainer = ppo_config.build()

    # 训练过程
    for i in range(100):
        result = trainer.train()
        print(f"Iteration {i}: reward={result['episode_reward_mean']}")

        # 保存模型检查点
        if i % 10 == 0:
            checkpoint_path = trainer.save()
            print(f"Checkpoint saved at {checkpoint_path}")

    # 训练完成后关闭 Ray
    ray.shutdown()
        
    
    

if __name__ == "__main__":
    main()
