import os
import json

# 定义需要插入的数据
task_data = {
    "agent.n.01_1": "robot0"
}

init_info_data = {"robot0": {
                "class_module": "omnigibson.robots.fetch_discrete_mobile",
                "class_name": "Fetch_fixedcamera",
                "args": {
                    "name": "robot0",
                    "obs_modalities": [
                        "rgb",
                        "seg_instance",
                        "proprio"
                    ],
                    "grasping_mode": "physical",
                    "default_reset_mode": "tuck",
                    "default_arm_pose": "diagonal30",
                    "uuid": 3143690,
                    "proprio_obs": [
                        "base_qpos",
                        "camera_qpos",
                        "arm_0_qpos",
                        "gripper_0_qpos"
                    ],
                    "sensor_config": {"VisionSensor": {"sensor_kwargs": {"image_height": 384, "image_width": 384}}}
                        
                }
            }
        }

state_data = {
    "robot0": {
                "root_link": {
                    "pos": [
                        0.7449771761894226,
                        -1.6200429201126099,
                        -0.0004126131534576416
                    ],
                    "ori": [
                        0.00021816078515257686,
                        0.012888547964394093,
                        0.016867680475115776,
                        -0.9997747540473938
                    ],
                    "lin_vel": [
                        -0.000668405438773334,
                        -0.00013205420691519976,
                        -0.00013995477638673037
                    ],
                    "ang_vel": [
                        0.0004121083766222,
                        -0.000927913817577064,
                        0.0003866233746521175
                    ]
                },
                "joints": {
                    "l_wheel_joint": {
                        "pos": [
                            0.049843378365039825
                        ],
                        "vel": [
                            0.01298891194164753
                        ],
                        "effort": [
                            0.0
                        ],
                        "target_pos": [
                            0.049843378365039825
                        ],
                        "target_vel": [
                            0.0
                        ]
                    },
                    "r_wheel_joint": {
                        "pos": [
                            0.018020078539848328
                        ],
                        "vel": [
                            -0.001455148565582931
                        ],
                        "effort": [
                            0.0
                        ],
                        "target_pos": [
                            0.018020078539848328
                        ],
                        "target_vel": [
                            0.0
                        ]
                    },
                    "torso_lift_joint": {
                        "pos": [
                            0.003877718700096011
                        ],
                        "vel": [
                            0.00028331123758107424
                        ],
                        "effort": [
                            282.5425109863281
                        ],
                        "target_pos": [
                            0.003877718700096011
                        ],
                        "target_vel": [
                            0.00028331123758107424
                        ]
                    },
                    "head_pan_joint": {
                        "pos": [
                            4.463301195301028e-07
                        ],
                        "vel": [
                            -0.0010744701139628887
                        ],
                        "effort": [
                            0.0
                        ],
                        "target_pos": [
                            -8.222376772504258e-13
                        ],
                        "target_vel": [
                            -0.0010744701139628887
                        ]
                    },
                    "shoulder_pan_joint": {
                        "pos": [
                            1.260060429573059
                        ],
                        "vel": [
                            0.00013285758905112743
                        ],
                        "effort": [
                            -0.03979064151644707
                        ],
                        "target_pos": [
                            1.260060429573059
                        ],
                        "target_vel": [
                            0.00013285758905112743
                        ]
                    },
                    "head_tilt_joint": {
                        "pos": [
                            4.5714777741068247e-08
                        ],
                        "vel": [
                            -0.00020620427676476538
                        ],
                        "effort": [
                            0.0
                        ],
                        "target_pos": [
                            -1.1043024827905867e-10
                        ],
                        "target_vel": [
                            -0.00020620427676476538
                        ]
                    },
                    "shoulder_lift_joint": {
                        "pos": [
                            1.487149953842163
                        ],
                        "vel": [
                            -0.0007608061423525214
                        ],
                        "effort": [
                            12.475908279418945
                        ],
                        "target_pos": [
                            1.487149953842163
                        ],
                        "target_vel": [
                            -0.0007608061423525214
                        ]
                    },
                    "upperarm_roll_joint": {
                        "pos": [
                            -0.3527311682701111
                        ],
                        "vel": [
                            -0.00308566028252244
                        ],
                        "effort": [
                            -0.8831924200057983
                        ],
                        "target_pos": [
                            -0.3527311682701111
                        ],
                        "target_vel": [
                            -0.00308566028252244
                        ]
                    },
                    "elbow_flex_joint": {
                        "pos": [
                            1.7124614715576172
                        ],
                        "vel": [
                            -0.00018616065790411085
                        ],
                        "effort": [
                            16.523845672607422
                        ],
                        "target_pos": [
                            1.7124614715576172
                        ],
                        "target_vel": [
                            -0.00018616065790411085
                        ]
                    },
                    "forearm_roll_joint": {
                        "pos": [
                            0.011939816176891327
                        ],
                        "vel": [
                            -0.0023790765553712845
                        ],
                        "effort": [
                            0.11616577953100204
                        ],
                        "target_pos": [
                            0.011939816176891327
                        ],
                        "target_vel": [
                            -0.0023790765553712845
                        ]
                    },
                    "wrist_flex_joint": {
                        "pos": [
                            1.5164238214492798
                        ],
                        "vel": [
                            0.001471543568186462
                        ],
                        "effort": [
                            -0.01947609893977642
                        ],
                        "target_pos": [
                            1.5164238214492798
                        ],
                        "target_vel": [
                            0.001471543568186462
                        ]
                    },
                    "wrist_roll_joint": {
                        "pos": [
                            -0.04538358002901077
                        ],
                        "vel": [
                            -0.004079216625541449
                        ],
                        "effort": [
                            0.0017939602257683873
                        ],
                        "target_pos": [
                            -0.04538358002901077
                        ],
                        "target_vel": [
                            -0.004079216625541449
                        ]
                    },
                    "l_gripper_finger_joint": {
                        "pos": [
                            0.05000000074505806
                        ],
                        "vel": [
                            7.110803608156857e-07
                        ],
                        "effort": [
                            0.0
                        ],
                        "target_pos": [
                            0.05000000074505806
                        ],
                        "target_vel": [
                            7.110803608156857e-07
                        ]
                    },
                    "r_gripper_finger_joint": {
                        "pos": [
                            0.04999999701976776
                        ],
                        "vel": [
                            -6.173167435008509e-07
                        ],
                        "effort": [
                            0.0
                        ],
                        "target_pos": [
                            0.05000000074505806
                        ],
                        "target_vel": [
                            -6.173167435008509e-07
                        ]
                    }
                },
                "controllers": {
                    "base": {
                        "goal_is_valid": True,
                        "goal": {
                            "vel": [
                                0.0,
                                0.0
                            ]
                        }
                    },
                    "camera": {
                        "goal_is_valid": True,
                        "goal": {
                            "target": [
                                -8.222376772504258e-13,
                                -1.1043024827905867e-10
                            ]
                        }
                    },
                    "arm_0": {
                        "goal_is_valid": True,
                        "goal": {
                            "target_pos": [
                                0.0793778107640577,
                                -0.17895500174888923,
                                0.7638931311167472
                            ],
                            "target_quat": [
                                0.4877191775044889,
                                -0.5081529072124557,
                                0.5111108627578225,
                                0.4926218760437143
                            ]
                        },
                        "control_filter": {
                            "past_samples": [
                                [
                                    0.0038760926108807325,
                                    1.2599955797195435,
                                    1.4871435165405273,
                                    -0.3527546226978302,
                                    1.7124736309051514,
                                    0.01194410864263773,
                                    1.5164153575897217,
                                    -0.04534482583403587
                                ],
                                [
                                    0.003877718700096011,
                                    1.260060429573059,
                                    1.487149953842163,
                                    -0.3527311682701111,
                                    1.7124614715576172,
                                    0.011939816176891327,
                                    1.5164238214492798,
                                    -0.04538358002901077
                                ]
                            ],
                            "current_idx": 0,
                            "fully_filled": True
                        }
                    },
                    "gripper_0": {
                        "goal_is_valid": True,
                        "goal": {
                            "target": [
                                0.0
                            ]
                        }
                    }
                },
                "non_kin": {}
            }
}

def modify_json(file_path):
    # 打开文件并加载内容
    with open(file_path, 'r') as f:
        data = json.load(f)

    # 添加 task 中的数据
    if "task" in data["metadata"] and "inst_to_name" in data["metadata"]["task"]:
        data["metadata"]["task"]["inst_to_name"].update(task_data)

    # 添加 state 中的数据
    if "state" in data and "object_registry" in data["state"]:
        data["state"]["object_registry"].update(state_data)

    # data["objects_info"]["init_info"].update(init_info_data)
    try:
        data["objects_info"]["init_info"].pop("robot0")
    except:
        pass
        # 将修改后的数据保存回文件
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=4)

def process_directory(directory):
    # 遍历目录下所有的 json 文件
    for root, dirs, files in os.walk(directory):
        for file in files:
            print(file)
            if file.endswith('.json'):
                file_path = os.path.join(root, file)
                modify_json(file_path)
                print(f"修改文件: {file_path}")

# 调用函数遍历指定的子目录
directory_path = "/data/zxlei/embodied/embodied-bench/omnigibson/shengyin/ycx/"  # 请替换成实际的子目录路径
process_directory(directory_path)
