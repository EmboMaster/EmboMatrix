#======================= 实验脚本 experiment_script.py =======================
import os
import time
import argparse
import yaml
import omnigibson as og
from omnigibson import example_config_path
from omnigibson.model.baseline_discrete.IL_trainer import imitation_learning_trainner

def run_experiment():
    parser = argparse.ArgumentParser()
    # parser.add_argument("--scene_path", type=str, required=True)
    # parser.add_argument("--task_file", type=str, required=True) 
    # parser.add_argument("--gpu_id", type=int, required=True)
    parser.add_argument(
        "--num_transformer_block",
        type=int,
        default=1,
        help="GPU ID for the network",
    )
    args = parser.parse_args()

    # GPU配置
    og.macros.gm.GPU_ID = args.gpu_id
    device = f"cuda:{args.gpu_id}"

    # 加载配置文件
    scene_file = os.path.join(args.scene_path, args.task_file).replace(".bddl", ".json")
    bddl_file = scene_file.replace('.json', '.bddl')
    
    with open(f"{example_config_path}/fetch_discrete_behavior.yaml", "r") as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)

    # cfg["env"]["flatten_action_space"] = True
    # cfg["env"]["flatten_obs_space"] = True
    # 动态配置参数
    cfg['scene'].update({
        # "scene_model": os.path.basename(args.scene_path),
        # "scene_instance": args.task_file,
        "scene_file": scene_file,
        "not_load_object_categories": ["door", "blanket","carpet","bath_rug","mat","place_mat","yoga_mat"],
        "waypoint_resolution": 0.1,
        "trav_map_resolution": 0.05,
    })

    cfg['task'].update({
        "activity_name": args.task_file.split('.')[0],
        "problem_filename": bddl_file
    })

    cfg['env'].update({
        "action_frequency": 120,
        "rendering_frequency": 120,
        "flatten_action_space": True,
        "flatten_obs_space": True,
    })
    # cfg.update({
    #     "scene": {
    #         "scene_model": os.path.basename(args.scene_path),
    #         "scene_instance": args.task_file,
    #         "scene_file": scene_file,
    #         "not_load_object_categories": ["door", "blanket","carpet","bath_rug","mat","place_mat","yoga_mat"]
    #     },
    #     "task": {
    #         "activity_name": args.task_file.split('.')[0],
    #         "problem_filename": bddl_file
    #     },
    #     "env": {
    #         "action_frequency": 120,
    #         "rendering_frequency": 120
    #     }
    # })

    # 初始化环境
    env = og.Environment(configs=cfg)
    env.reset()

    # 训练流程
    try:
        iltrainer = imitation_learning_trainner(args)
        iltrainer.train(env, "", device, args)
        print("Success")  # 重要结束标记
    except Exception as e:
        print(f"Failed: {str(e)}")
    finally:
        og.sim.close()

if __name__ == "__main__":
    run_experiment()