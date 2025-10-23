import os
import json
import os
import random
import torch as th
from bddl.activity import (
    Conditions,
    evaluate_goal_conditions,
    get_goal_conditions,
    get_ground_goal_state_options,
    get_initial_conditions,
    get_natural_goal_conditions,
    get_natural_initial_conditions,
    get_object_scope,
)

import omnigibson as og
import omnigibson.utils.transform_utils as T
from omnigibson.macros import gm
from omnigibson.object_states import Pose
from omnigibson.reward_functions.potential_reward import PotentialReward
from omnigibson.robots.robot_base import BaseRobot
from omnigibson.scenes.scene_base import Scene
from omnigibson.scenes.traversable_scene import TraversableScene
from omnigibson.tasks.task_base import BaseTask
from omnigibson.tasks.behavior_task import BehaviorTask
from omnigibson.termination_conditions.predicate_goal import PredicateGoal
from omnigibson.termination_conditions.timeout import Timeout
from omnigibson.utils.bddl_utils import BEHAVIOR_ACTIVITIES, BDDLEntity, BDDLSampler, OmniGibsonBDDLBackend
from omnigibson.utils.python_utils import assert_valid_key, classproperty
from omnigibson.utils.ui_utils import create_module_logger
from omnigibson.action_primitives.starter_semantic_action_primitives import (
    PlanningContext,
    StarterSemanticActionPrimitives,
)
from omnigibson.objects.object_base import REGISTERED_OBJECTS
from omnigibson.reward_functions.grasp_reward import GraspReward
from omnigibson.tasks.task_base import BaseTask
from omnigibson.termination_conditions.falling import Falling
from omnigibson.termination_conditions.grasp_goal import GraspGoal
from omnigibson.termination_conditions.timeout import Timeout
from omnigibson.utils.grasping_planning_utils import get_grasp_poses_for_object_sticky
from omnigibson.utils.motion_planning_utils import set_arm_and_detect_collision
from omnigibson.utils.python_utils import classproperty, create_class_from_registry_and_config
from omnigibson.utils.sim_utils import land_object
from omnigibson.utils.pose_utils import _get_target_object, _get_robot_object_relative_pose, _get_value_ObjectsInFOVOfRobot
import math
from omnigibson.utils.camera_utils import calculate_rotation_quaternion

MAX_JOINT_RANDOMIZATION_ATTEMPTS = 50
# Create module logger
log = create_module_logger(module_name=__name__)

class GraspBehaviorTask(BehaviorTask):
    """
    Task for BEHAVIOR

    Args:
        activity_name (None or str): Name of the Behavior Task to instantiate
        activity_definition_id (int): Specification to load for the desired task. For a given Behavior Task, multiple task
            specifications can be used (i.e.: differing goal conditions, or "ways" to complete a given task). This
            ID determines which specification to use
        activity_instance_id (int): Specific pre-configured instance of a scene to load for this BehaviorTask. This
            will be used only if @online_object_sampling is False.
        predefined_problem (None or str): If specified, specifies the raw string definition of the Behavior Task to
            load. This will automatically override @activity_name and @activity_definition_id.
        online_object_sampling (bool): whether to sample object locations online at runtime or not
        highlight_task_relevant_objects (bool): whether to overlay task-relevant objects in the scene with a colored mask
        termination_config (None or dict): Keyword-mapped configuration to use to generate termination conditions. This
            should be specific to the task class. Default is None, which corresponds to a default config being usd.
            Note that any keyword required by a specific task class but not specified in the config will automatically
            be filled in with the default config. See cls.default_termination_config for default values used
        reward_config (None or dict): Keyword-mapped configuration to use to generate reward functions. This should be
            specific to the task class. Default is None, which corresponds to a default config being usd. Note that
            any keyword required by a specific task class but not specified in the config will automatically be filled
            in with the default config. See cls.default_reward_config for default values used
    """

    def __init__(
        self,
        obj_name,
        activity_name=None,
        activity_definition_id=0,
        activity_instance_id=0,
        predefined_problem=None,
        online_object_sampling=False,
        highlight_task_relevant_objects=False,
        termination_config=None,
        reward_config=None,
        subgoals=False,
        subgoal_configs=[],
        precached_reset_pose_path=None        
    ):  
        ## behavior task init 
        # Make sure object states are enabled
        assert gm.ENABLE_OBJECT_STATES, "Must set gm.ENABLE_OBJECT_STATES=True in order to use BehaviorTask!"

        ## grasp task init 
        self.obj_name = obj_name
        self._primitive_controller = None
        self._reset_poses = None
        # self._objects_config = objects_config
        if precached_reset_pose_path is not None:
            with open(precached_reset_pose_path) as f:
                self._reset_poses = json.load(f)
        # Run super init
        super().__init__(
            activity_name=activity_name,
            activity_definition_id=activity_definition_id,
            activity_instance_id=activity_instance_id,
            predefined_problem=predefined_problem,
            online_object_sampling=online_object_sampling,
            highlight_task_relevant_objects=highlight_task_relevant_objects,
            termination_config=termination_config,
            reward_config=reward_config,
            subgoals=subgoals,
            subgoal_configs=subgoal_configs
        )

    def _create_termination_conditions(self):
        terminations = dict()
        terminations["graspgoal"] = GraspGoal(
            self.obj_name
        )
        # This helpes to prevent resets happening at different times
        terminations["timeout"] = Timeout(max_steps=self._termination_config["max_steps"])
        # terminations["falling"] = Falling()

        return terminations

    def _create_reward_functions(self):
        rewards = dict()
        rewards["grasp"] = GraspReward(self.obj_name, **self._reward_config)
        return rewards

    def _load(self, env):
        # Initialize the current activity
        success, self.feedback = self.initialize_activity(env=env)
        # assert success, f"Failed to initialize Behavior Activity. Feedback:\n{self.feedback}"

        # Store the scene name
        self.scene_name = env.scene.scene_model if isinstance(env.scene, TraversableScene) else None

        # Highlight any task relevant objects if requested
        if self.highlight_task_relevant_objs:
            for entity in self.object_scope.values():
                if entity.synset == "agent":
                    continue
                if not entity.is_system and entity.exists:
                    entity.highlighted = True

        # Add callbacks to handle internal processing when new systems / objects are added / removed to the scene
        callback_name = f"{self.activity_name}_refresh"
        og.sim.add_callback_on_add_obj(name=callback_name, callback=self._update_bddl_scope_from_added_obj)
        og.sim.add_callback_on_remove_obj(name=callback_name, callback=self._update_bddl_scope_from_removed_obj)

        og.sim.add_callback_on_system_init(name=callback_name, callback=self._update_bddl_scope_from_system_init)
        og.sim.add_callback_on_system_clear(name=callback_name, callback=self._update_bddl_scope_from_system_clear)

    def _load_non_low_dim_observation_space(self):
        # No non-low dim observations so we return an empty dict
        return dict()

    def get_agent(self, env):
        """
        Grab the 0th agent from @env

        Args:
            env (Environment): Current active environment instance

        Returns:
            BaseRobot: The 0th robot from the environment instance
        """
        # We assume the relevant agent is the first agent in the scene
        return env.robots[0]

    def _reset_agent(self, env):
        robot = env.robots[0]
        robot.release_grasp_immediately()
        # If available, reset the robot with cached reset poses.
        # This is significantly faster than randomizing using the primitives.
        if self._reset_poses is not None:
            joint_control_idx = th.cat([robot.trunk_control_idx, robot.arm_control_idx[robot.default_arm]])
            robot_pose = random.choice(self._reset_poses)
            robot.set_joint_positions(robot_pose["joint_pos"], joint_control_idx)
            robot_pos = th.tensor(robot_pose["base_pos"])
            robot_orn = th.tensor(robot_pose["base_ori"])
            robot.set_position_orientation(position=robot_pos, orientation=robot_orn, frame="scene")

        # Otherwise, reset using the primitive controller.
        else:
            if self._primitive_controller is None:
                self._primitive_controller = StarterSemanticActionPrimitives(env, enable_head_tracking=False)

            # Randomize the robots joint positions
            joint_control_idx = th.cat([robot.trunk_control_idx, robot.arm_control_idx[robot.default_arm]])
            dim = len(joint_control_idx)
            # For Tiago
            if "combined" in robot.robot_arm_descriptor_yamls:
                joint_combined_idx = th.cat([robot.trunk_control_idx, robot.arm_control_idx["combined"]])
                initial_joint_pos = th.tensor(robot.get_joint_positions()[joint_combined_idx])
                control_idx_in_joint_pos = th.where(th.isin(joint_combined_idx, joint_control_idx))[0]
            # For Fetch
            else:
                initial_joint_pos = th.tensor(robot.get_joint_positions()[joint_control_idx])
                control_idx_in_joint_pos = th.arange(dim)

            with PlanningContext(
                env, self._primitive_controller.robot, self._primitive_controller.robot_copy, "original"
            ) as context:
                for _ in range(MAX_JOINT_RANDOMIZATION_ATTEMPTS):
                    joint_pos, joint_control_idx = self._get_random_joint_position(robot)
                    initial_joint_pos[control_idx_in_joint_pos] = joint_pos
                    if not set_arm_and_detect_collision(context, initial_joint_pos):
                        robot.set_joint_positions(joint_pos, joint_control_idx)
                        og.sim.step()
                        break

            # Randomize the robot's 2d pose
            obj = env.scene.object_registry("name", self.obj_name)
            grasp_poses = get_grasp_poses_for_object_sticky(obj)
            grasp_pose, _ = random.choice(grasp_poses)
            world_robots_base_to_obj = obj.get_position() - grasp_pose[0]
            yaw_cos = world_robots_base_to_obj[0]/(th.norm(world_robots_base_to_obj[:2]))
            yaw_to_object = math.acos(yaw_cos)
            
            # sampled_pose_2d = self._primitive_controller._sample_pose_near_object(obj)
            # sampled_pose_2d = self._primitive_controller._sample_pose_near_object(obj, pose_on_obj=grasp_pose, distance_lo=0.3, distance_hi=0.5, yaw_lo=yaw_to_object-math.pi/3, yaw_hi=yaw_to_object+math.pi/3)
            sampled_pose_2d = self._primitive_controller._sample_pose_near_object(obj, pose_on_obj=grasp_pose, distance_lo=0.3, distance_hi=0.5, yaw_lo=math.pi, yaw_hi=math.pi)
            robot_pose = self._primitive_controller._get_robot_pose_from_2d_pose(sampled_pose_2d)
            robot.set_position_orientation(*robot_pose)
            
            # Settle robot
            for _ in range(10):
                og.sim.step()

            # Wait for the robot to fully stabilize.
            for _ in range(100):
                og.sim.step()
                if th.norm(robot.get_linear_velocity()) > 1e-2:
                    continue
                if th.norm(robot.get_angular_velocity()) > 1e-2:
                    continue
                break
            else:
                raise ValueError("Robot could not settle")

            # Check if the robot has toppled
            robot_up = T.quat_apply(robot.get_position_orientation()[1], th.tensor([0, 0, 1], dtype=th.float32))
            if robot_up[2] < 0.75:
                raise ValueError("Robot has toppled over")
            
            ## reset the virtual external camera
            object_position = obj.get_position()
            camera_pos = robot.sensors['robot0:eyes:Camera:0'].get_position()
            ori = calculate_rotation_quaternion(object_position - camera_pos)
            pos = camera_pos + 0.1 * (object_position - camera_pos)
            env._external_sensors['external_sensor1'].set_position_orientation(pos, ori)


    def _reset_scene(self, env):
        # Reset the scene
        super()._reset_scene(env)
        
        obs = env.scene.object_registry("name", self.obj_name)
        position = obs.get_position()
        position[2] = 2.0
        env._external_sensors['external_sensor0'].set_position_orientation(position, [0,0,0,1])
        
        # # Reset objects
        # for obj_config in self._objects_config:
        #     # Get object in the scene
        #     obj_name = obj_config["name"]
        #     obj = env.scene.object_registry("name", obj_name)
        #     if obj is None:
        #         raise ValueError("Object {} not found in scene".format(obj_name))

        #     # Set object pose
        #     obj_pos = [0.0, 0.0, 0.0] if "position" not in obj_config else obj_config["position"]
        #     obj_orn = [0.0, 0.0, 0.0, 1.0] if "orientation" not in obj_config else obj_config["orientation"]
        #     obj.set_position_orientation(position=obj_pos, orientation=obj_orn, frame="scene")

    # Overwrite reset by only removeing reset scene
    def reset(self, env):
        """
        Resets this task in the environment

        Args:
            env (Environment): environment instance to reset
        """
        # Reset the scene, agent, and variables

        # Try up to 20 times.
        for _ in range(20):
            try:
                self._reset_scene(env)
                self._reset_agent(env)
                break
            except Exception as e:
                print("Resetting error: ", e)
        else:
            raise ValueError("Could not reset task.")
        # self._reset_scene(env)
        # self._reset_agent(env)
        self._reset_variables(env)

        # Also reset all termination conditions and reward functions
        for termination_condition in self._termination_conditions.values():
            termination_condition.reset(self, env)
        for reward_function in self._reward_functions.values():
            reward_function.reset(self, env)

    def _get_random_joint_position(self, robot):
        joint_positions = []
        joint_control_idx = th.cat([robot.trunk_control_idx, robot.arm_control_idx[robot.default_arm]])
        joints = [joint for joint in robot.joints.values()]
        # joints = th.tensor([joint for joint in robot.joints.values()], dtype=th.float32)
        arm_joints = [joints[i] for i in joint_control_idx]
        for i, joint in enumerate(arm_joints):
            val = random.uniform(joint.lower_limit, joint.upper_limit)
            joint_positions.append(val)
        return th.tensor(joint_positions), joint_control_idx
    
    def _get_obs(self, env):
        low_dim_obs = dict()

        # Batch rpy calculations for much better efficiency
        objs_exist = {obj: obj.exists for obj in self.object_scope.values() if not obj.is_system}
        objs_rpy = T.quat2euler(
            th.stack(
                [
                    obj.states[Pose].get_value()[1] if obj_exist else th.tensor([0, 0, 0, 1.0])
                    for obj, obj_exist in objs_exist.items()
                ]
            )
        )
        objs_rpy_cos = th.cos(objs_rpy)
        objs_rpy_sin = th.sin(objs_rpy)

        # Always add agent info first
        agent = self.get_agent(env=env)
        low_dim_obs = self.add_robot_proprio(low_dim_obs, agent)
        low_dim_obs = self.pose_wrapper(low_dim_obs, env, self)

        return low_dim_obs, dict()

    def pose_wrapper(self, obs_tensor, env, policy):

        robot = env.robots[0]

        try:
            obj_env_name = self.obj_name
            target_object = _get_target_object(env, obj_env_name)
        except:
            obj_env_name = None
            target_object = None

        objects_in_fov = [
            # obj.name for obj in env.scene.get_objects_in_robot_fov(robot)
            obj.name for obj in _get_value_ObjectsInFOVOfRobot(robot, env)
        ]

        if "relative_pose" not in obs_tensor.keys():
            if target_object is not None:
                relative_pose = _get_robot_object_relative_pose(robot, target_object)
                obs_tensor["in_fov"] = th.tensor([1 if target_object.name in objects_in_fov else 0])
            else:
                relative_pose = th.tensor([0, 0, 0, 0, 0, 0, 0])
                obs_tensor["in_fov"] = th.tensor([0])
            obs_tensor["relative_pose"] = relative_pose
        
        return obs_tensor

    def _step_termination(self, env, action, info=None):
        # Get all dones and successes from individual termination conditions
        dones = []
        successes = []
        info = dict() if info is None else info
        if "termination_conditions" not in info:
            info["termination_conditions"] = dict()
        for name, termination_condition in self._termination_conditions.items():
            d, s = termination_condition.step(self, env, action)
            dones.append(d)
            successes.append(s)
            info["termination_conditions"][name] = {
                "done": d,
                "success": s,
            }
        # Any True found corresponds to a done / success
        done = sum(dones) > 0
        success = sum(successes) > 0

        # Populate info
        info["success"] = success


        # Add additional info
        # info["goal_status"] = self._termination_conditions["predicate"].goal_status

        return done, info


    @property
    def name(self):
        """
        Returns:
            str: Name of this task. Defaults to class name
        """
        name_base = super().name

        # Add activity name, def id, and inst id
        return f"{name_base}_{self.activity_name}_{self.activity_definition_id}_{self.activity_instance_id}"

    @classproperty
    def valid_scene_types(cls):
        # Any scene can be used
        return {Scene}

    @classproperty
    def default_termination_config(cls):
        return {
            "max_steps": 500,
        }

    @classproperty
    def default_reward_config(cls):
        return {
            "dist_coeff": 0.1,
            "grasp_reward": 1.0,
            "collision_penalty": 1.0,
            "eef_position_penalty_coef": 0.01,
            "eef_orientation_penalty_coef": 0.001,
            "regularization_coef": 0.01,
            "r_detect": 0.1,
            "r_approach_threshold": 2.0,
            "approach_distance_threshold": 1.0,
            "grasp_distance_threshold": 0.1,
        }