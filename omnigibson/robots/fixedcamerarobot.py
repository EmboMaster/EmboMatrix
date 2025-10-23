from abc import abstractmethod

import torch as th

from omnigibson.robots.robot_base import BaseRobot
from omnigibson.utils.python_utils import classproperty
from omnigibson.utils.usd_utils import ControllableObjectViewAPI
from omnigibson.robots.active_camera_robot import ActiveCameraRobot

class FixedCameraRobot(ActiveCameraRobot):
    """
    Robot with a fixed camera. The camera cannot move independently and remains static relative to the robot base.
    """

    def _validate_configuration(self):
        # Override validation to remove camera controller requirement
        # Since this robot has no controllable camera
        pass

    def _get_proprioception_dict(self):
        # Fixed camera, no dynamic proprioceptive information for the camera
        dic = super()._get_proprioception_dict()
        dic["camera_qpos"] = th.zeros(len(self.camera_control_idx))
        dic["camera_qpos_sin"] = th.zeros(len(self.camera_control_idx))
        dic["camera_qpos_cos"] = th.ones(len(self.camera_control_idx))
        dic["camera_qvel"] = th.zeros(len(self.camera_control_idx))
        return dic

    @property
    def controller_order(self):
        # No camera controller for a fixed camera
        return []

    @property
    def _default_controllers(self):
        # Disable all camera controllers
        controllers = super()._default_controllers
        if "camera" in controllers:
            del controllers["camera"]
        return controllers

    @property
    def _default_controller_config(self):
        # Remove camera-specific configuration
        cfg = super()._default_controller_config
        if "camera" in cfg:
            del cfg["camera"]
        return cfg

    @property
    def camera_joint_names(self):
        # No controllable camera joints for fixed camera
        return []

    @property
    def camera_control_idx(self):
        # No control indices for fixed camera
        return th.tensor([])

    def reset(self):
        """
        Override reset to ensure camera joints are fixed at initial positions.
        """
        super().reset()
        # Force camera joints to their initial positions
        if hasattr(self, "camera_joint_names") and self.camera_joint_names:
            joint_positions = ControllableObjectViewAPI.get_joint_positions(self.articulation_root_path)
            for idx in self.camera_control_idx:
                joint_positions[idx] = self.reset_joint_pos[idx]
            ControllableObjectViewAPI.set_joint_positions(self.articulation_root_path, joint_positions)
