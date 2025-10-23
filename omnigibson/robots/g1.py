import os
import torch as th
from omnigibson.macros import gm
from omnigibson.utils.python_utils import classproperty
from omnigibson.robots.leggedrobot import LeggedRobot
from omnigibson.robots.manipulation_robot import ManipulationRobot

class G1(LeggedRobot,ManipulationRobot):
    """
    G1 Robot
    """

    def __init__(
        self,
        # Shared kwargs in hierarchy
        name,
        relative_prim_path=None,
        scale=None,
        visible=True,
        visual_only=False,
        self_collisions=False,
        load_config=None,
        # Unique to USDObject hierarchy
        abilities=None,
        # Unique to ControllableObject hierarchy
        control_freq=None,
        controller_config=None,
        action_type="continuous",
        action_normalize=True,
        reset_joint_pos=None,
        # Unique to BaseRobot
        obs_modalities=("rgb", "proprio"),
        proprio_obs="default",
        sensor_config=None,
        # Unique to ManipulationRobot
        grasping_mode="physical",
        grasping_direction="lower",
        disable_grasp_handling=False,
        **kwargs,
    ):
        
        super().__init__(
            relative_prim_path=relative_prim_path,
            name=name,
            scale=scale,
            visible=visible,
            visual_only=visual_only,
            self_collisions=self_collisions,
            load_config=load_config,
            abilities=abilities,
            control_freq=control_freq,
            controller_config=controller_config,
            action_type=action_type,
            action_normalize=action_normalize,
            reset_joint_pos=reset_joint_pos,
            obs_modalities=obs_modalities,
            proprio_obs=proprio_obs,
            sensor_config=sensor_config,
        )
    

    
    @property
    def discrete_action_list(self):
        raise NotImplementedError()

    def _create_discrete_action_space(self):
        raise ValueError("G1 does not support discrete actions!")

    @property
    def controller_order(self):
        controllers = []
        for leg in self.leg_names:
            controllers += [ f"foot_{leg}"]
        controllers += ["leg"]
        for arm in self.arm_names:
            controllers += [f"arm_{arm}", f"gripper_{arm}"]
        return controllers
    
    @property
    def _default_joint_pos(self):
        return th.zeros(37)

    @property
    def _default_controllers(self):
        controllers = super()._default_controllers
        for leg in self.leg_names:
            controllers["foot_{}".format(leg)] = "JointController"
        controllers["leg"] = "JointController"
        for arm in self.arm_names:
            controllers["arm_{}".format(arm)] = "JointController"
            controllers["gripper_{}".format(arm)] = "JointController"
        return controllers
    
    @classproperty
    def n_arms(cls):
        return 2

    @classproperty
    def arm_names(cls):
        return ["left", "right"]

    @property
    def arm_link_names(self):
        return {arm: [
                f"{arm}_shoulder_pitch_link",
                f"{arm}_shoulder_roll_link",
                f"{arm}_shoulder_yaw_link",
                f"{arm}_elbow_pitch_link",
                f"{arm}_elbow_roll_link"                
                ] for arm in self.arm_names  
            }

    @property
    def arm_joint_names(self):
        return {arm: [
                f"{arm}_shoulder_pitch_joint",
                f"{arm}_shoulder_roll_joint",
                f"{arm}_shoulder_yaw_joint",
                f"{arm}_elbow_pitch_joint",
                f"{arm}_elbow_roll_joint"
                ] for arm in self.arm_names  
            }
    
    @property
    def eef_link_names(self):
        return {arm: f"{arm}_palm_link" for arm in self.arm_names}

    @property
    def finger_link_names(self):
        return {arm: [
                f"{arm}_five_link",
                f"{arm}_three_link",
                f"{arm}_six_link",
                f"{arm}_four_link",
                f"{arm}_zero_link",
                f"{arm}_one_link",
                f"{arm}_two_link"
                ] for arm in self.arm_names  
            }
    @property
    def finger_joint_names(self):
        return {arm: [
                f"{arm}_five_joint",
                f"{arm}_three_joint",
                f"{arm}_six_joint", 
                f"{arm}_four_joint",
                f"{arm}_zero_joint",
                f"{arm}_one_joint",
                f"{arm}_two_joint"
                ] for arm in self.arm_names  
            }

    @classproperty
    def n_legs(cls):
        return 2

    @classproperty
    def leg_names(cls):
        return ["left", "right"]
    
    @property
    def leg_link_names(self):
        return {leg: [
                f"{leg}_hip_yaw_link", 
                f"{leg}_hip_roll_link",
                f"{leg}_hip_pitch_link", 
                f"{leg}_knee_link"
                ] for leg in self.leg_names  
            }

    @property
    def leg_joint_names(self):
        return {leg: [
                f"{leg}_hip_yaw_joint", 
                f"{leg}_hip_roll_joint",
                f"{leg}_hip_pitch_joint", 
                f"{leg}_knee_joint"
                ] for leg in self.leg_names  
            }
    
    @property
    def foot_link_names(self):
        return {leg: [
                f"{leg}_ankle_roll_link",
                f"{leg}_ankle_pitch_link"
                ] for leg in self.leg_names  
            }
    
    @property
    def foot_joint_names(self):
        return {leg: [
                f"{leg}_ankle_roll_joint",
                f"{leg}_ankle_pitch_joint"
                ] for leg in self.leg_names  
            }

    @property
    def torso_joint_names(self):
        return ["torso_joint"]
    
    @property
    def usd_path(self):
        return os.path.join(gm.ASSET_PATH, "models/g1/g1.usd")
    
    @property
    def eef_usd_path(self):
        return {arm: os.path.join(gm.ASSET_PATH, "models/g1/g1_eef.usd") for arm in self.arm_names}
    
    @property
    def urdf_path(self):
        return os.path.join(gm.ASSET_PATH, f"models/g1/g1.urdf")
    
    @property
    def robot_arm_descriptor_yamls(self):
        return {arm: os.path.join(gm.ASSET_PATH, f"models/g1/g1_{arm}_descriptor.yaml") for arm in self.arm_names}
    
    @property
    def finger_lengths(self):
        return {self.default_arm: 0.087}
    
    @property
    def arm_workspace_range(self):
        return {arm: [th.deg2rad(-45), th.deg2rad(45)] for arm in self.arm_names}