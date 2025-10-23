import os
from abc import abstractmethod
import torch as th
from omnigibson.macros import gm
from omnigibson.robots.robot_base import BaseRobot
from omnigibson.utils.python_utils import classproperty

class LeggedRobot(BaseRobot):
    """
    LeggedRobot
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
    ):
        
        super().__init__(
            relative_prim_path=relative_prim_path,
            name=name,
            scale=scale,
            visible=visible,
            fixed_base=False,
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
    def controller_order(self):
        # Assumes we have leg(s) and corresponding foot(s)
        controllers = []
        for leg in self.leg_names:
            controllers += ["foot_{}".format(leg)]
        controllers += ['leg']

        return controllers

    @property
    def _default_controllers(self):
        # Always call super first
        controllers = super()._default_controllers

        # For best generalizability use, joint controller as default
        for leg in self.leg_names:
            controllers["foot_{}".format(leg)] = "JointController"
        controllers["leg"] = "JointController"

        return controllers
    
    @property
    def _default_controller_config(self):
        # Always run super method first
        cfg = super()._default_controller_config

        leg_joint_configs = self._default_leg_joint_controller_configs
        leg_null_joint_configs = self._default_leg_null_joint_controller_configs

        foot_joint_configs = self._default_foot_joint_controller_configs
        foot_null_configs = self._default_foot_null_controller_configs

        # Add leg and foot defaults, per leg
        cfg["leg"] = {
                leg_joint_configs["name"]: leg_joint_configs,
                leg_null_joint_configs["name"]: leg_null_joint_configs,
            }
        for leg in self.leg_names:
            
            cfg["foot_{}".format(leg)] = {
                foot_joint_configs[leg]["name"]: foot_joint_configs[leg],
                foot_null_configs[leg]["name"]: foot_null_configs[leg],
            }

        return cfg
    
    @property
    def _default_leg_joint_controller_configs(self):
        """
        Returns:
            dict: Dictionary mapping leg appendage name to default controller config to control that
                robot's leg. Uses velocity control by default.
        """
        idx=[self.torso_control_idx]
        for leg in self.leg_names:
            idx.append(self.leg_control_idx[leg])
        dic = {
            "name": "JointController",
            "control_freq": self._control_freq,
            "control_limits": self.control_limits,
            "dof_idx": th.cat(idx),
            "command_output_limits": None,
            "motor_type": "velocity",
            "use_delta_commands": True,
            "use_impedances": True,
        }
        return dic
    
    @property
    def _default_leg_null_joint_controller_configs(self):
        """
        Returns:
            dict: Dictionary mapping leg appendage name to default leg null controller config
                to control this robot's leg i.e. dummy controller
        """
        idx=[self.torso_control_idx]
        for leg in self.leg_names:
            idx.append(self.leg_control_idx[leg])
        dic = {
            "name": "NullJointController",
            "control_freq": self._control_freq,
            "control_limits": self.control_limits,
            "dof_idx": th.cat(idx),
            "command_output_limits": None,
            "motor_type": "velocity",
            "default_command": self.reset_joint_pos[th.cat(idx)],
            "use_impedances": False,
        }

        return dic
    
    @property
    def _default_foot_joint_controller_configs(self):
        """
        Returns:
            dict: Dictionary mapping leg appendage name to default foot joint controller config
                to control this robot's foot
        """
        dic = {}
        for leg in self.leg_names:
            dic[leg] = {
                "name": "JointController",
                "control_freq": self._control_freq,
                "motor_type": "velocity",
                "control_limits": self.control_limits,
                "dof_idx": self.foot_control_idx[leg],
                "command_output_limits": "default",
                "use_delta_commands": False,
                "use_impedances": False
            }
        return dic

    @property
    def _default_foot_null_controller_configs(self):
        """
        Returns:
            dict: Dictionary mapping leg appendage name to default foot null controller config
                to control this robot's (non-prehensile) foot i.e. dummy controller
        """
        dic = {}
        for leg in self.leg_names:
            dic[leg] = {
                "name": "NullJointController",
                "control_freq": self._control_freq,
                "motor_type": "velocity",
                "control_limits": self.control_limits,
                "dof_idx": self.foot_control_idx[leg],
                "default_command": th.zeros(len(self.foot_control_idx[leg])),
                "use_impedances": False,
            }
        return dic

    @classproperty
    def n_legs(cls):
        """
        Returns:
            int: Number of legs this robot has. Returns 2 by default
        """
        return 2

    @classproperty
    def leg_names(cls):
        """
        Returns:
            list of str: List of leg names for this robot. Should correspond to the keys used to index into
                leg- and foot-related dictionaries, e.g.: eef_link_names, foot_link_names, etc.
                Default is string enumeration based on @self.n_legs.
        """
        return [str(i) for i in range(cls.n_legs)]

    @property
    def default_leg(self):
        """
        Returns:
            str: Default leg name for this robot, corresponds to the first entry in @leg_names by default
        """
        return self.leg_names[0:1]
    
    @property
    @abstractmethod
    def leg_link_names(self):
        """
        Returns:
            dict: Dictionary mapping leg appendage name to corresponding leg link names,
                should correspond to specific link names in this robot's underlying model file

                Note: the ordering within the dictionary is assumed to be intentional, and is
                directly used to define the set of corresponding idxs.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def leg_joint_names(self):
        """
        Returns:
            dict: Dictionary mapping leg appendage name to corresponding leg joint names,
                should correspond to specific joint names in this robot's underlying model file

                Note: the ordering within the dictionary is assumed to be intentional, and is
                directly used to define the set of corresponding control idxs.
        """
        raise NotImplementedError
    
    @property
    @abstractmethod
    def foot_link_names(self):
        """
        Returns:
            dict: Dictionary mapping leg appendage name to array of link names corresponding to
                this robot's foots

                Note: the ordering within the dictionary is assumed to be intentional, and is
                directly used to define the set of corresponding idxs.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def foot_joint_names(self):
        """
        Returns:
            dict: Dictionary mapping leg appendage name to array of joint names corresponding to
                this robot's foots.

                Note: the ordering within the dictionary is assumed to be intentional, and is
                directly used to define the set of corresponding control idxs.
        """
        raise NotImplementedError

    @property
    def leg_control_idx(self):
        """
        Returns:
            dict: Dictionary mapping leg appendage name to indices in low-level control
                vector corresponding to leg joints.
        """
        return {
            leg: th.tensor([list(self.joints.keys()).index(name) for name in self.leg_joint_names[leg]])
            for leg in self.leg_names
        }

    @property
    def foot_control_idx(self):
        """
        Returns:
            dict: Dictionary mapping leg appendage name to indices in low-level control
                vector corresponding to foot joints.
        """
        return {
            leg: th.tensor([list(self.joints.keys()).index(name) for name in self.foot_joint_names[leg]])
            for leg in self.leg_names
        }

    @property
    def leg_links(self):
        """
        Returns:
            dict: Dictionary mapping leg appendage name to robot links corresponding to
                that leg's links
        """
        return {leg: [self._links[link] for link in self.leg_link_names[leg]] for leg in self.leg_names}


    @property
    def foot_links(self):
        """
        Returns:
            dict: Dictionary mapping leg appendage name to robot links corresponding to
                that leg's foot links
        """
        return {leg: [self._links[link] for link in self.foot_link_names[leg]] for leg in self.leg_names}

    @property
    def foot_joints(self):
        """
        Returns:
            dict: Dictionary mapping leg appendage name to robot joints corresponding to
                that leg's foot joints
        """
        return {leg: [self._joints[joint] for joint in self.foot_joint_names[leg]] for leg in self.leg_names}
    
    @property
    def torso_joint_names(self):
        raise NotImplementedError("trunk_joint_names must be implemented in subclass")

    @property
    def torso_control_idx(self):
        """
        Returns:
            n-array: Indices in low-level control vector corresponding to trunk joints.
        """
        return th.tensor([list(self.joints.keys()).index(name) for name in self.torso_joint_names])
    
    @classproperty
    def _do_not_register_classes(cls):
        # Don't register this class since it's an abstract template
        classes = super()._do_not_register_classes
        classes.add("LeggedRobot")
        return classes
