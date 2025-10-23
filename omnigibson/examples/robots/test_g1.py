import torch as th

import omnigibson as og
from omnigibson.robots import REGISTERED_ROBOTS


def main(random_selection=False, headless=False, short_exec=False):
    """
    Robot demo
    Loads all robots in an empty scene, generate random actions
    """
    og.log.info(f"Demo {__file__}\n    " + "*" * 80 + "\n    Description:\n" + main.__doc__ + "*" * 80)
    # Create empty scene with no robots in it initially
    cfg = {
        "scene": {
            "type": "Scene",
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

    og.sim.stop()
    og.sim.play()


    if not headless:
        # Set viewer in front facing robot
        og.sim.viewer_camera.set_position_orientation(
            position=th.tensor([2.69918369, -3.63686664, 4.57894564]),
            orientation=th.tensor([0.39592411, 0.1348514, 0.29286304, 0.85982]),
        )

    og.sim.enable_viewer_camera_teleoperation()
    # Hold still briefly so viewer can see robot
    for _ in range(100):
        og.sim.step()

    for _ in range(1000):
        action_lo, action_hi = -0.1, 0.1
        action = th.cat([th.zeros(15),th.rand(22) * (action_hi - action_lo) + action_lo])
        #action=th.cat([th.zeros(10),th.tensor([-0.0]),th.zeros(26)])
        
        for _ in range(5):
            env.step(action)
    

    # Stop the simulator and remove the robot
    og.sim.stop()


    # Always shut down the environment cleanly at the end
    og.clear()


if __name__ == "__main__":
    main()
