import torch as th

import omnigibson as og
from omnigibson.robots import REGISTERED_ROBOTS
import numpy as np
import time

class ZeroGravityWalker:
    def __init__(self):
        # 初始化步态周期的参数
        self.current_step = "left"  # 初始状态为左腿先迈步
        self.walk_cycle_frequency = 0.5  # 每次步态的频率，单位为秒
    
    def generate_walk_command(self):
        """
        Generates control signals for continuous walking by alternating legs.
        """
        command = np.zeros(15)  # 15个自由度（0-14）

        if self.current_step == "left":
            # 左腿迈步
            command[5] = -0.3  # 左髋关节向后摆动
            command[6] = 0.2   # 左膝关节弯曲
            command[7] = -0.1  # 左踝关节向下调整
            command[10] = 0.3  # 右髋关节向前摆动
            command[11] = -0.2 # 右膝关节弯曲
            command[12] = 0.1  # 右踝关节调整
            self.current_step = "right"  # 下一步换右腿迈步

        elif self.current_step == "right":
            # 右腿迈步
            command[10] = -0.3  # 右髋关节向后摆动
            command[11] = 0.2   # 右膝关节弯曲
            command[12] = -0.1  # 右踝关节向下调整
            command[5] = 0.3    # 左髋关节向前摆动
            command[6] = -0.2   # 左膝关节弯曲
            command[7] = 0.1    # 左踝关节调整
            self.current_step = "left"  # 下一步换左腿迈步

        # 躯干控制保持平衡
        command[4] = 0.1  # 躯干轻微调整

        return th.tensor(command)
    
    def walk_forward_continuous(self, controller):
        """
        Makes the robot continuously walk forward by alternating between left and right steps.
        """
        while True:
            # 生成当前步态的控制命令
            walk_command = self.generate_walk_command()

            # 发送控制命令
            controller.send_command(dof_idx=controller["dof_idx"], command=walk_command)

            # 控制步态频率，等待下一个步态周期
            time.sleep(self.walk_cycle_frequency)

def main(random_selection=False, headless=False, short_exec=False):
    """
    Robot demo
    Loads all robots in an empty scene, generate random actions
    """
    og.log.info(f"Demo {__file__}\n    " + "*" * 80 + "\n    Description:\n" + main.__doc__ + "*" * 80)
    # Create empty scene with no robots in it initially
    cfg = {
        "scene": {
            "type": "InteractiveTraversableScene",
            "scene_model": "Rs_int",
            "not_load_object_categories": "null",
        },
        "robots": [
            {
                "type": "G1",
                "name": "G1",
                "obs_modalities": [],
                "position": [0, 0, 0.8],
            },
        ],
    }

    env = og.Environment(configs=cfg)
    walker = ZeroGravityWalker()
    og.sim.stop()
    og.sim.play()


    if not headless:
        # Set viewer in front facing robot
        og.sim.viewer_camera.set_position_orientation(
            position=th.tensor([1.69918369, -3.63686664, 2.07894564]),
            orientation=th.tensor([0.39592411, 0.1348514, 0.29286304, 0.85982]),
        )

    og.sim.enable_viewer_camera_teleoperation()
    # Hold still briefly so viewer can see robot
    for _ in range(100):
        og.sim.step()

    for _ in range(1000):
        action_lo, action_hi = -0.1, 0.1
        action = th.cat([th.zeros(15),th.rand(22) * (action_hi - action_lo) + action_lo])
        # action = th.zeros(37)
        # if _ % 20 == 0:
        #     action[:15] = walker.generate_walk_command()
        #action[:15] = generate_zero_gravity_walk_command()
        #action=th.cat([th.zeros(10),th.tensor([-0.0]),th.zeros(26)])
        # time.sleep(0.1)
        # action[7]=0.1
        #action = th.rand(37) * (action_hi - action_lo) + action_lo
        for _ in range(5):
            env.step(action)
    

    # Stop the simulator and remove the robot
    og.sim.stop()


    # Always shut down the environment cleanly at the end
    og.clear()


if __name__ == "__main__":
    main()
