import json
import math
import os

import numpy as np
import torch as th

import sys
import os

# 定义你本地 bddl 包的父目录的绝对路径
# 例如 /home/user/my_project/bddl
# local_bddl_parent_path = "data/bddl" 

# # 使用 insert(0, ...) 来确保你的路径被优先搜索
# if local_bddl_parent_path not in sys.path:
#     sys.path.insert(0, local_bddl_parent_path)

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

from bddl.parsing import (
    gen_natural_language_condition,
    gen_natural_language_conditions,
    parse_domain,
    parse_problem,
)

import omnigibson as og
# from omnigibson.action_primitives.starter_semantic_action_primitives import StarterSemanticActionPrimitives
# from omnigibson.objects.dataset_object import DatasetObject
from omnigibson.utils.camera_utils import calculate_rotation_quaternion, quaternion_from_axis_angle, quaternion_multiply, rotate_vector_around_perpendicular_axis, rotate_vector_by_quaternion
# from omnigibson.utils.sim_utils import land_object, test_valid_pose
import omnigibson.utils.transform_utils as T
from omnigibson.macros import gm
from omnigibson.object_states import Pose
from omnigibson.reward_functions.potential_reward import PotentialReward
from omnigibson.robots.robot_base import BaseRobot
from omnigibson.scenes.scene_base import Scene
from omnigibson.scenes.traversable_scene import TraversableScene
from omnigibson.tasks.task_base import BaseTask
from omnigibson.termination_conditions.predicate_goal import PredicateGoal
from omnigibson.termination_conditions.timeout import Timeout
from omnigibson.utils.bddl_utils import BEHAVIOR_ACTIVITIES, BDDLEntity, BDDLSampler, OmniGibsonBDDLBackend
from omnigibson.utils.python_utils import assert_valid_key, classproperty
from omnigibson.utils.ui_utils import create_module_logger
# from omnigibson.reward_functions.new_point_goal_reward import NewPointGoalReward
# from omnigibson.reward_functions.new_grasp_reward import GraspReward
# from omnigibson.reward_functions.carry_reward import CarryReward
# from omnigibson.reward_functions.put_reward import PutReward
from omnigibson.utils.pose_utils import _get_target_object, _get_robot_object_relative_pose, _get_value_ObjectsInFOVOfRobot
# import random



# Create module logger
log = create_module_logger(module_name=__name__)

class BehaviorTask(BaseTask):
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
        relative_pose=True,
        subgoal_ongoing='subgoal1',
        randomize_initial_pos=False,
        problem_filename=None,
    ):
        # Make sure object states are enabled
        assert gm.ENABLE_OBJECT_STATES, "Must set gm.ENABLE_OBJECT_STATES=True in order to use BehaviorTask!"

        # Make sure task name is valid if not specifying a predefined problem
        # if predefined_problem is None:
        #     assert (
        #         activity_name is not None
        #     ), "Activity name must be specified if no predefined_problem is specified for BehaviorTask!"
        #     assert_valid_key(key=activity_name, valid_keys=BEHAVIOR_ACTIVITIES, name="Behavior Task")
        # else:
        #     # Infer activity name
        #     activity_name = predefined_problem.split("problem ")[-1].split("-")[0]

        # Initialize relevant variables

        # BDDL
        self.backend = OmniGibsonBDDLBackend()

        # Activity info
        self.activity_name = None
        self.activity_definition_id = activity_definition_id
        self.activity_instance_id = activity_instance_id
        self.activity_conditions = None
        self.activity_initial_conditions = None
        self.activity_goal_conditions = None
        self.ground_goal_state_options = None
        self.feedback = None  # None or str
        self.sampler = None  # BDDLSampler

        # Scene info
        self.scene_name = None

        # Object info
        self.online_object_sampling = online_object_sampling  # bool
        self.highlight_task_relevant_objs = highlight_task_relevant_objects  # bool
        self.object_scope = None  # Maps str to BDDLEntity
        self.object_instance_to_category = None  # Maps str to str
        self.future_obj_instances = None  # set of str

        # Info for demonstration collection
        self.instruction_order = None  # th.tensor of int
        self.currently_viewed_index = None  # int
        self.currently_viewed_instruction = None  # tuple of str
        self.activity_natural_language_goal_conditions = None  # str
        self.subgoals = subgoals
        self.relative_pose = relative_pose
        if self.subgoals:
            self.subgoal_configs = {}
            for subgoal_config in subgoal_configs:
                self.subgoal_configs[subgoal_config['type']] = subgoal_config
            
        self.problem_filename=problem_filename
        # Load the initial behavior configuration
        self.update_activity(
            activity_name=activity_name,
            activity_definition_id=activity_definition_id,
            predefined_problem=predefined_problem,
        )
        

        # init subgoal reward settings
        if self.subgoals:
            # self.subgoal_ongoing = list(self.subgoal_activity_goal_conditions.keys())[0]
            self.get_subgoals_reward()

        self.subgoal_ongoing = subgoal_ongoing
        self.randomize_initial_pos = randomize_initial_pos

        # Run super init
        super().__init__(termination_config=termination_config, reward_config=reward_config)
    
    def get_subgoals_reward(self):
        self.subgoals_reward_dict = {}
        for subgoalname, subgoal in self.subgoal_activity_goal_conditions.items():
            # find task type
            task_type = subgoal[0].instruction.split('task')[0].strip()
            task_cfg = self.subgoal_configs[task_type]
            reward_class_name = task_cfg['reward_class']
            reward_cls = globals()[reward_class_name]
            init_kwargs = {k: v for k, v in task_cfg.items() if k not in ['type', 'reward_class']}
            # Todo get object name
            target_object_name = subgoal[0].body[-1].strip("?")
            init_kwargs['obj_name'] = target_object_name
            try:
                self.subgoals_reward_dict[subgoalname] = reward_cls(**init_kwargs)
            except:
                pass
    @classmethod
    def get_cached_activity_scene_filename(
        cls, scene_model, activity_name, activity_definition_id, activity_instance_id
    ):
        """
        Helper method to programmatically construct the scene filename for a given pre-cached task configuration

        Args:
            scene_model (str): Name of the scene (e.g.: Rs_int)
            activity_name (str): Name of the task activity (e.g.: putting_away_halloween_decorations)
            activity_definition_id (int): ID of the task definition
            activity_instance_id (int): ID of the task instance

        Returns:
            str: Filename which, if exists, should include the cached activity scene
        """
        return f"{scene_model}_task_{activity_name}_{activity_definition_id}_{activity_instance_id}_template"

    @classmethod
    def verify_scene_and_task_config(cls, scene_cfg, task_cfg):
        # Run super first
        super().verify_scene_and_task_config(scene_cfg=scene_cfg, task_cfg=task_cfg)

        # Possibly modify the scene to load if we're using online_object_sampling
        scene_instance, scene_file = scene_cfg["scene_instance"], scene_cfg["scene_file"]
        activity_name = (
            task_cfg["predefined_problem"].split("problem ")[-1].split("-")[0]
            if task_cfg.get("predefined_problem", None) is not None
            else task_cfg["activity_name"]
        )
        if scene_file is None and scene_instance is None and not task_cfg["online_object_sampling"]:
            scene_instance = cls.get_cached_activity_scene_filename(
                scene_model=scene_cfg.get("scene_model", "Scene"),
                activity_name=activity_name,
                activity_definition_id=task_cfg.get("activity_definition_id", 0),
                activity_instance_id=task_cfg.get("activity_instance_id", 0),
            )
            # Update the value in the scene config
            scene_cfg["scene_instance"] = scene_instance

    @property
    def task_metadata(self):
        # Store mapping from entity name to its corresponding BDDL instance name
        return dict(
            inst_to_name={inst: entity.name for inst, entity in self.object_scope.items() if entity.exists},
        )

    def _create_termination_conditions(self):
        # Initialize termination conditions dict and fill in with Timeout and PredicateGoal
        terminations = dict()

        terminations["timeout"] = Timeout(max_steps=self._termination_config["max_steps"])
        terminations["predicate"] = PredicateGoal(goal_fcn=lambda: self.activity_goal_conditions)

        return terminations

    def _create_reward_functions(self):
        # Initialize reward functions dict and fill in with Potential reward
        rewards = dict()

        rewards["potential"] = PotentialReward(
            potential_fcn=self.get_potential,
            r_potential=self._reward_config["r_potential"],
        )

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

    def update_activity(self, activity_name, activity_definition_id, predefined_problem=None):
        """
        Update the active Behavior activity being deployed

        Args:
            activity_name (None or str): Name of the Behavior Task to instantiate
            activity_definition_id (int): Specification to load for the desired task. For a given Behavior Task, multiple task
                specifications can be used (i.e.: differing goal conditions, or "ways" to complete a given task). This
                ID determines which specification to use
            predefined_problem (None or str): If specified, specifies the raw string definition of the Behavior Task to
                load. This will automatically override @activity_name and @activity_definition_id.
        """
        # Update internal variables based on values

        # Activity info
        self.activity_name = activity_name
        self.activity_definition_id = activity_definition_id

        self.activity_conditions = Conditions(
            activity_name,
            activity_definition_id,
            simulator_name="omnigibson",
            predefined_problem=predefined_problem,
            problem_filename=self.problem_filename
        )

        # Get scope, making sure agent is the first entry
        self.object_scope = {"agent.n.01_1": None}
        self.object_scope.update(get_object_scope(self.activity_conditions))

        # Object info
        self.object_instance_to_category = {
            obj_inst: obj_cat
            for obj_cat in self.activity_conditions.parsed_objects
            for obj_inst in self.activity_conditions.parsed_objects[obj_cat]
        }

        # Generate initial and goal conditions
        self.activity_initial_conditions = get_initial_conditions(
            self.activity_conditions, self.backend, self.object_scope
        )
        self.activity_goal_conditions = get_goal_conditions(self.activity_conditions, self.backend, self.object_scope)
        self.ground_goal_state_options = get_ground_goal_state_options(
            self.activity_conditions, self.backend, self.object_scope, self.activity_goal_conditions
        )

        # if self.activity_conditions.parsed_goal_conditions == []:
        #     self.activity_conditions.parsed_goal_conditions = [['not', ['toggled_on', '?table_lamp.n.01_1']]]

        # Demo attributes
        if self.activity_conditions.parsed_goal_conditions:
            # 如果目標不為空，則正常建立指令順序
            self.instruction_order = th.arange(len(self.activity_conditions.parsed_goal_conditions))
            self.instruction_order = self.instruction_order[th.randperm(self.instruction_order.size(0))]
            self.currently_viewed_index = 0
            self.currently_viewed_instruction = self.instruction_order[self.currently_viewed_index]
        else:
            # 如果目標為空，設置為安全的空值
            self.instruction_order = th.tensor([])  # 空的張量
            self.currently_viewed_index = 0
            self.currently_viewed_instruction = None # 使用 None 表示沒有指令，比用數字索引更清晰

        self.activity_natural_language_initial_conditions = get_natural_initial_conditions(self.activity_conditions)
        self.activity_natural_language_goal_conditions = get_natural_goal_conditions(self.activity_conditions)
        
        if self.subgoals:
            # Add code about subgoals
            # 子目标都在同一个文件中，使用 SubgoalConditions 来解析
            self.subgoal_conditions = SubgoalConditions(activity_name, problem_filename=self.problem_filename)
            subgoals = self.subgoal_conditions.extract_subgoals()
            self.subgoal_activity_goal_conditions = {}
            # 对于每一个子目标，我们需要生成相应的goal conditions
            # 子目标中定义的条件(predictes)需要转换为activity_goal_conditions格式。
            # 因为 get_goal_conditions 是根据 Conditions 对象来获取的，这里我们可以
            # 临时构造一个类似 Conditions 的对象来给 get_goal_conditions 使用。
            
            # 为简单起见，这里假设每个子目标本身的parsed_goal_conditions就是我们要的条件列表（predicate列表），
            # 但实际中可能需要进一步解析。可以像主任务一样构建一个 "fake" Conditions 对象来调用 get_goal_conditions。

            # 注意：subgoals中的条件是单个子目标条件，需要构造一个临时Conditions-like对象来处理
            # 这里为演示目的简单处理：假设predicate本身可以直接evaluate_goal_conditions使用。
            
            for sg_name, sg_info in subgoals.items():
                # 构造一个临时 Conditions 对象只包含这个子目标的goal条件
                # 假设 parsed_goal_conditions 的格式与主任务一致，只是列表里只有这个子目标的条件
                try:
                    fake_conditions = self._build_fake_conditions_for_subgoal(sg_info["predicates"], self.subgoal_conditions)
                    sg_goal_conditions = get_goal_conditions(fake_conditions, self.backend, self.object_scope)
                    self.subgoal_activity_goal_conditions[sg_name] = sg_goal_conditions
                    self.subgoal_activity_goal_conditions[sg_name][0].instruction = self.subgoal_conditions.subgoals[sg_name]['instruction']
                except:
                    print(f"Failed to get goal conditions for subgoal {sg_name}")
                    continue
    def _build_fake_conditions_for_subgoal(self, predicate, subgoal_conditions):
        """
        构造一个假的 Conditions 对象，只包含当前子目标的条件，用于调用 get_goal_conditions。
        predicate: 单个子目标的谓词条件元组，比如("Inside", "obj1", "container")

        我们需要让这个fake conditions具备 parsed_goal_conditions 属性。
        假设子目标中就一个条件，如多条件可扩展为列表。
        """
        class FakeConditions:
            def __init__(self, parsed_objects, parsed_initial_conditions, single_predicate):
                self.parsed_objects = parsed_objects
                self.parsed_initial_conditions = parsed_initial_conditions
                # goal conditions需以列表形式存在
                self.parsed_goal_conditions = [single_predicate]

        # 由于子目标文件与主文件domain与objects一致，可以复用subgoal_conditions的parsed_objects
        return FakeConditions(subgoal_conditions.parsed_objects, subgoal_conditions.parsed_initial_conditions, predicate)



    # def get_potential(self, env):
    #     """
    #     Compute task-specific potential: distance to the goal

    #     Args:
    #         env (Environment): Current active environment instance

    #     Returns:
    #         float: Computed potential
    #     """
    #     # Evaluate the first ground goal state option as the potential
    #     _, satisfied_predicates = evaluate_goal_conditions(self.ground_goal_state_options[0])
    #     success_score = len(satisfied_predicates["satisfied"]) / (
    #         len(satisfied_predicates["satisfied"]) + len(satisfied_predicates["unsatisfied"])
    #     )
    #     # success_score = 0
    #     return -success_score

    def get_potential(self, env):
        """
        Compute task-specific potential: distance to the goal
        ...
        """
        # 如果没有任何 ground truth 目标选项，说明任务没有目标，视为已完成
        if not self.ground_goal_state_options:
            return -1.0  # 返回最大潜能（-1 * 100% 成功率）

        # Evaluate the first ground goal state option as the potential
        _, satisfied_predicates = evaluate_goal_conditions(self.ground_goal_state_options[0])
        
        satisfied_count = len(satisfied_predicates["satisfied"])
        unsatisfied_count = len(satisfied_predicates["unsatisfied"])
        total_count = satisfied_count + unsatisfied_count

        # 保護分母為零的情況 (如果目標列表為空，這裡也會是0)
        if total_count == 0:
            return -1.0  # 没有条件需要满足，视为100%成功

        success_score = satisfied_count / total_count
        return -success_score

    def initialize_activity(self, env):
        """
        Initializes the desired activity in the current environment @env

        Args:
            env (Environment): Current active environment instance

        Returns:
            2-tuple:
                - bool: Whether the generated scene activity should be accepted or not
                - dict: Any feedback from the sampling / initialization process
        """
        accept_scene = True
        feedback = None

        # Generate sampler
        self.sampler = BDDLSampler(
            env=env,
            activity_conditions=self.activity_conditions,
            object_scope=self.object_scope,
            backend=self.backend,
        )

        # Compose future objects
        self.future_obj_instances = {
            init_cond.body[1] for init_cond in self.activity_initial_conditions if init_cond.body[0] == "future"
        }

        if self.online_object_sampling:
            # Sample online
            accept_scene, feedback = self.sampler.sample()
            if not accept_scene:
                return accept_scene, feedback
        else:
            # Load existing scene cache and assign object scope accordingly
            self.assign_object_scope_with_cache(env)

        # Generate goal condition with the fully populated self.object_scope
        self.activity_goal_conditions = get_goal_conditions(self.activity_conditions, self.backend, self.object_scope)
        self.ground_goal_state_options = get_ground_goal_state_options(
            self.activity_conditions, self.backend, self.object_scope, self.activity_goal_conditions
        )
        return accept_scene, feedback

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

    def assign_object_scope_with_cache(self, env):
        """
        Assigns objects within the current object scope

        Args:
            env (Environment): Current active environment instance
        """
        # Load task metadata
        inst_to_name = self.load_task_metadata()["inst_to_name"]

        # Assign object_scope based on a cached scene
        for obj_inst in self.object_scope:
            if obj_inst in self.future_obj_instances:
                entity = None
            else:
                assert obj_inst in inst_to_name, (
                    f"BDDL object instance {obj_inst} should exist in cached metadata "
                    f"from loaded scene, but could not be found!"
                )
                name = inst_to_name[obj_inst]
                is_system = name in env.scene.available_systems.keys()
                entity = env.scene.get_system(name) if is_system else env.scene.object_registry("name", name)
            self.object_scope[obj_inst] = BDDLEntity(
                bddl_inst=obj_inst,
                entity=entity,
            )

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

        # set the nav & manip camera's position and orientation
        # nav camera: looking in the agent’s forward direction and points slightly down, with the horizon at a nominal 30 degree
        # manip camera: placed 90 degree in clockwise direction apart from the navigation camera around the vertical axis and also points slightly down, also with a nominal 30 degree horizon
        camera_pos = agent.sensors['robot0:eyes:Camera:0'].get_position()
        robot_ori = agent.get_orientation()
        robot_ori_vector = np.array(rotate_vector_by_quaternion([1,0,0], robot_ori)) # get the forward direction of the robot with camera's default orientation
        robot_ori_vector_xy =  np.array([robot_ori_vector[0], robot_ori_vector[1], 0])# ignore the z axis component since it comes from computational error
        robot_ori_vector_xy = robot_ori_vector_xy / np.linalg.norm(robot_ori_vector_xy) # normalize the vector

        nav_ori_vector = rotate_vector_around_perpendicular_axis(robot_ori_vector_xy, 30)
        nav_ori = calculate_rotation_quaternion(nav_ori_vector)
        nav_pos = camera_pos + robot_ori_vector_xy * 0.15
        env._external_sensors['external_sensor_nav'].set_position_orientation(nav_pos, nav_ori)

        manip_ori_vector_xy = np.array([robot_ori_vector[1], -robot_ori_vector[0], 0]) # rotate 90 degree in clockwise direction
        manip_ori_vector = rotate_vector_around_perpendicular_axis(manip_ori_vector_xy, 30)
        manip_ori = calculate_rotation_quaternion(manip_ori_vector)
        manip_pos = camera_pos + robot_ori_vector_xy * 0.15
        env._external_sensors['external_sensor_manip'].set_position_orientation(manip_pos, manip_ori)

        # for (obj, obj_exist), obj_rpy, obj_rpy_cos, obj_rpy_sin in zip(
        #     objs_exist.items(), objs_rpy, objs_rpy_cos, objs_rpy_sin
        # ):

        #     # TODO: May need to update checking here to USDObject? Or even baseobject?
        #     # TODO: How to handle systems as part of obs?
        #     if obj_exist:
        #         low_dim_obs[f"{obj.bddl_inst}_real"] = th.tensor([1.0])
        #         low_dim_obs[f"{obj.bddl_inst}_pos"] = obj.states[Pose].get_value()[0]
        #         low_dim_obs[f"{obj.bddl_inst}_ori_cos"] = obj_rpy_cos
        #         low_dim_obs[f"{obj.bddl_inst}_ori_sin"] = obj_rpy_sin
        #         if obj.name != agent.name:
        #             for arm in agent.arm_names:
        #                 grasping_object = agent.is_grasping(arm=arm, candidate_obj=obj.wrapped_obj)
        #                 low_dim_obs[f"{obj.bddl_inst}_in_gripper_{arm}"] = th.tensor([float(grasping_object)])
        #     else:
        #         low_dim_obs[f"{obj.bddl_inst}_real"] = th.zeros(1)
        #         low_dim_obs[f"{obj.bddl_inst}_pos"] = th.zeros(3)
        #         low_dim_obs[f"{obj.bddl_inst}_ori_cos"] = th.zeros(3)
        #         low_dim_obs[f"{obj.bddl_inst}_ori_sin"] = th.zeros(3)
        #         for arm in agent.arm_names:
        #             low_dim_obs[f"{obj.bddl_inst}_in_gripper_{arm}"] = th.zeros(1)


        return low_dim_obs, dict()

    # def _step_termination(self, env, action, info=None):
    #     # Run super first
    #     done, info = super()._step_termination(env=env, action=action, info=info)

    #     # Add additional info
    #     info["goal_status"] = self._termination_conditions["predicate"].goal_status

    #     return done, info

    def reset(self, env):
        """
        Resets this task in the environment

        Args:
            env (Environment): environment instance to reset
        """
        # Reset the scene, agent, and variables

        # Try up to 20 times.
        for _ in range(100):
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
    
    def _reset_agent(self, env):
        if self.randomize_initial_pos:
            robot = env.robots[0]
            # robot.reset()

            # # robot.set_position_orientation(*robot_pose)
            # scene_file = env.scene.scene_file
            # if scene_file is not None:
            #     # Grab objects info from the scene file
            #     if isinstance(scene_file, str):
            #         with open(scene_file, "r") as f:
            #             scene_info = json.load(f)
            #     else:
            #         scene_info = scene_file
                
            #     init_state = scene_info["state"]["object_registry"][robot.name]['root_link']
            #     robot.set_position_orientation(init_state["pos"], init_state["ori"])
            
            success, max_trials = False, 100

            pos, ori = None, None
            for i in range(max_trials):
                obj_env_name = env.scene._scene_info_meta_inst_to_name[env.task.subgoal_activity_goal_conditions[self.subgoal_ongoing][0].terms[-1]]
                target_object = env.scene.object_registry("name", obj_env_name)
                in_rooms = target_object.in_rooms[0]
                # room_instance = env.scene._seg_map.get_room_instance_by_point(obj_xy)
                _, pos = env.scene._seg_map.get_random_point_by_room_instance(in_rooms)
                yaw = th.rand(1) * 2 * math.pi - math.pi
                # pos = th.tensor([sample_pos[0], sample_pos[1], 0.01], dtype=th.float32)
                ori = T.euler2quat(th.tensor([0, 0, yaw], dtype=th.float32))
                if success:
                    break
            if not success:
                log.warning("Failed to reset robot without collision")
            # land_object(robot, pos, ori, env.initial_pos_z_offset)
            robot.set_position_orientation(pos, ori)
            
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

    def add_robot_proprio(self, low_dim_obs, robot):
        _proprioception_dict = robot._get_proprioception_dict()
        # transfer all the  0-d tensor into 1-d tensor
        for key in _proprioception_dict:
            if isinstance(_proprioception_dict[key], th.Tensor) and _proprioception_dict[key].dim() == 0:
                low_dim_obs[key] = _proprioception_dict[key].unsqueeze(0)
            else:
                low_dim_obs[key] = _proprioception_dict[key]

        return low_dim_obs

    def pose_wrapper(self, obs_tensor, env, policy):

        robot = env.robots[0]

        try:
            opreated_obj = policy.language_encoder.precomputed_lang_embedding[policy.ongoint_subgoal]['instruction'].split(' ')[-1]
            obj_env_name = env.scene._scene_info_meta_inst_to_name[opreated_obj]
            target_object = _get_target_object(env, obj_env_name)
        except:
            opreated_obj = None
            obj_env_name = None
            target_object = None

        objects_in_fov = [
            # obj.name for obj in env.scene.get_objects_in_robot_fov(robot)
            obj.name for obj in _get_value_ObjectsInFOVOfRobot(robot, env)
        ]

        if "relative_pose" not in obs_tensor.keys():
            if opreated_obj is not None:
                relative_pose = _get_robot_object_relative_pose(robot, target_object)
                obs_tensor["in_fov"] = th.tensor([1 if target_object.name in objects_in_fov else 0])
            else:
                relative_pose = th.tensor([0, 0, 0, 0, 0, 0, 0])
                obs_tensor["in_fov"] = th.tensor([0])
            obs_tensor["relative_pose"] = relative_pose

        return obs_tensor

    def _update_bddl_scope_from_added_obj(self, obj):
        """
        Internal callback function to be called when new objects are added to the simulator to potentially update internal
        bddl object scope

        Args:
            obj (BaseObject): Newly imported object
        """
        # Iterate over all entities, and if they don't exist, check if any category matches @obj's category, and set it
        # if it does, and immediately return
        for inst, entity in self.object_scope.items():
            if not entity.exists and not entity.is_system and obj.category in set(entity.og_categories):
                entity.set_entity(entity=obj)
                return

    def _update_bddl_scope_from_removed_obj(self, obj):
        """
        Internal callback function to be called when sim._pre_remove_object() is called to potentially update internal
        bddl object scope

        Args:
            obj (BaseObject): Newly removed object
        """
        # Iterate over all entities, and if they exist, check if any name matches @obj's name, and remove it
        # if it does, and immediately return
        for entity in self.object_scope.values():
            if entity.exists and not entity.is_system and obj.name == entity.name:
                entity.clear_entity()
                return

    def _update_bddl_scope_from_system_init(self, system):
        """
        Internal callback function to be called when system.initialize() is called to potentially update internal
        bddl object scope

        Args:
            system (BaseSystem): Newly initialized system
        """
        # Iterate over all entities, and potentially match the system to the scope
        for inst, entity in self.object_scope.items():
            if not entity.exists and entity.is_system and entity.og_categories[0] == system.name:
                entity.set_entity(entity=system)
                return

    def _update_bddl_scope_from_system_clear(self, system):
        """
        Internal callback function to be called when system.clear() is called to potentially update internal
        bddl object scope

        Args:
            system (BaseSystem): Newly cleared system
        """
        # Iterate over all entities, and potentially remove the matched system from the scope
        for inst, entity in self.object_scope.items():
            if entity.exists and entity.is_system and system.name == entity.name:
                entity.clear_entity()
                return

    # def show_instruction(self):
    #     """
    #     Get current instruction for user

    #     Returns:
    #         3-tuple:
    #             - str: Current goal condition in natural language
    #             - 3-tuple: (R,G,B) color to assign to text
    #             - list of BaseObject: Relevant objects for the current instruction
    #     """
    #     satisfied = (
    #         self.currently_viewed_instruction in self._termination_conditions["predicate"].goal_status["satisfied"]
    #     )
    #     natural_language_condition = self.activity_natural_language_goal_conditions[self.currently_viewed_instruction]
    #     objects = self.activity_goal_conditions[self.currently_viewed_instruction].get_relevant_objects()
    #     text_color = (
    #         [83.0 / 255.0, 176.0 / 255.0, 72.0 / 255.0] if satisfied else [255.0 / 255.0, 51.0 / 255.0, 51.0 / 255.0]
    #     )

    #     return natural_language_condition, text_color, objects

    def show_instruction(self):
        """
        Get current instruction for user
        ...
        """
        # 添加保護，如果沒有可查看的指令，返回預設信息
        if self.currently_viewed_instruction is None:
            # (顯示文字, 文字顏色, 相關物件列表)
            return "No goal specified.", (1.0, 1.0, 1.0), []

        satisfied = (
            self.currently_viewed_instruction in self._termination_conditions["predicate"].goal_status["satisfied"]
        )
        natural_language_condition = self.activity_natural_language_goal_conditions[self.currently_viewed_instruction]
        objects = self.activity_goal_conditions[self.currently_viewed_instruction].get_relevant_objects()
        text_color = (
            [83.0 / 255.0, 176.0 / 255.0, 72.0 / 255.0] if satisfied else [255.0 / 255.0, 51.0 / 255.0, 51.0 / 255.0]
        )

        return natural_language_condition, text_color, objects

    def iterate_instruction(self):
        """
        Increment the instruction
        """

        if not self.activity_conditions.parsed_goal_conditions:
            return

        self.currently_viewed_index = (self.currently_viewed_index + 1) % len(
            self.activity_conditions.parsed_goal_conditions
        )
        self.currently_viewed_instruction = self.instruction_order[self.currently_viewed_index]

    def save_task(self, path=None, override=False):
        """
        Writes the current scene configuration to a .json file

        Args:
            path (None or str): If specified, absolute fpath to the desired path to write the .json. Default is
                <gm.DATASET_PATH>/scenes/<SCENE_MODEL>/json/...>
            override (bool): Whether to override any files already found at the path to write the task .json
        """
        if path is None:
            assert self.scene_name is not None, "Scene name must be set in order to save task without specifying path"
            fname = self.get_cached_activity_scene_filename(
                scene_model=self.scene_name,
                activity_name=self.activity_name,
                activity_definition_id=self.activity_definition_id,
                activity_instance_id=self.activity_instance_id,
            )
            path = os.path.join(gm.DATASET_PATH, "scenes", self.scene_name, "json", f"{fname}.json")

        if os.path.exists(path) and not override:
            log.warning(f"Scene json already exists at {path}. Use override=True to force writing of new json.")
            return
        # Write metadata and then save
        self.write_task_metadata()
        og.sim.save(json_paths=[path])
        
        
    
    def _load_subgoals_from_file(self, activity_name):
        """
        根据activity_name从 
        bddl_subgoals/bddl_subgoals/activity_definitions/{activity_name}/problem0.bddl 文件中
        读取子目标定义。

        返回值:
            dict: {subgoal_name: {"instruction": str, "predicate": tuple(...)}}
        """
        subgoals_file = os.path.join("bddl_subgoals", "bddl_subgoals", "activity_definitions", activity_name, "problem0.bddl")
        if not os.path.exists(subgoals_file):
            log.warning(f"Subgoals file not found for activity {activity_name} at {subgoals_file}")
            return {}

        # 将文件内容读取为字符串传入parse_problem进行解析
        with open(subgoals_file, "r") as f:
            predefined_problem_str = f.read()

        domain_name, *__ = parse_domain("omnigibson")

        # 调用 parse_problem 来解析子目标定义文件
        # 这里 activity_definition 暂时写成0，依据实际需求可灵活调整
        # parse_problem返回: problem_name, objects, initial_state, goal_state
        # 其中goal_state即解析后的goal conditions列表
        _, _, _, goal_conditions = parse_problem(
            behavior_activity=activity_name,
            activity_definition=0,
            domain_name=domain_name,
            predefined_problem=predefined_problem_str
        )

        # 从goal_conditions中提取子目标
        subgoals = _parse_subgoals(goal_conditions)
        return subgoals
    
    def _parse_subgoals(self, conditions):
        """
        从conditions对象中解析出子目标。

        假设conditions.parsed_goal_conditions包含goal中定义的条件列表，
        其中子目标格式类似：
        (:subgoalX "描述信息" (Predicate ...))
        
        这里需要根据实际BDDL parser的结构来解析。
        下方代码仅作为示意，根据实际情况可能需要调整解析方式。
        """
        subgoals = {}
        for cond in conditions.parsed_goal_conditions:
            # 假设cond的结构类似于 (":subgoalX", "描述字符串", ("PredicateName", arg1, arg2...))
            if isinstance(cond, tuple) and len(cond) >= 3 and cond[0].startswith(':subgoal'):
                subgoal_name = cond[0][1:]  # 移除前缀':'得到subgoalX
                instruction = cond[1]
                predicate = cond[2]  # 实际的谓词条件元组
                subgoals[subgoal_name] = {
                    "instruction": instruction,
                    "predicate": predicate,
                }

        return subgoals

    def evaluate_subgoal(self, subgoal_predicate):
        """
        对单一子目标谓词进行评估。
        假设evaluate_goal_conditions可以接收列表作为参数。
        我们把subgoal_predicate包装成列表传入，返回该子目标是否已满足。
        """
        success, satisfied_predicates = evaluate_goal_conditions([subgoal_predicate])
        # 如果没有未满足的谓词，则表示完成
        return len(satisfied_predicates["unsatisfied"]) == 0

    def _step_termination(self, env, action, info=None):
        done, info = self._step_termination_base(env=env, action=action, info=info)

        info["goal_status"] = self._termination_conditions["predicate"].goal_status

        # 对每个子目标调用evaluate_goal_conditions
        if self.subgoals:
            subgoal_status = {}
            # for sg_name, sg_conditions in self.subgoal_activity_goal_conditions.items():
            #     done_sg, sg_status = evaluate_goal_conditions(sg_conditions)
            #     subgoal_status[sg_name] = sg_status
            sg_conditions = self.subgoal_activity_goal_conditions[self.subgoal_ongoing]
            try:
                done_sg, sg_status = evaluate_goal_conditions(sg_conditions)
            except:
                done_sg = False
                sg_status = {
                    "satisfied": [],
                    "unsatisfied": [0],
                }
            subgoal_status[self.subgoal_ongoing] = sg_status
            # 如果当前子目标完成，切换到下一个子目标
            if done_sg:
                # 找到下一个子目标
                subgoal_names = list(self.subgoal_activity_goal_conditions.keys())
                current_index = subgoal_names.index(self.subgoal_ongoing)
                next_index = (current_index + 1) % len(subgoal_names)
                self.subgoal_ongoing = subgoal_names[next_index]
                
            info["subgoal_status"] = subgoal_status

        return done, info
    
    def _step_termination_base(self, env, action, info=None):
        """
        Step and aggregate termination conditions

        Args:
            env (Environment): Environment instance
            action (n-array): 1D flattened array of actions executed by all agents in the environment
            info (None or dict): Any info to return

        Returns:
            2-tuple:
                - float: aggregated termination at the current timestep
                - dict: any information passed through this function or generated by this function
        """
        # Get all dones and successes from individual termination conditions
        dones = []
        successes = []
        info = dict() if info is None else info
        if "termination_conditions" not in info:
            info["termination_conditions"] = dict()
        for name, termination_condition in self._termination_conditions.items():
            try: 
                d, s = termination_condition.step(self, env, action)
            except:
                d, s = False, False
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
        return done, info
    
    def _step_reward(self, env, action, info=None):
        """
        Step and aggregate reward functions

        Args:
            env (Environment): Environment instance
            action (n-array): 1D flattened array of actions executed by all agents in the environment
            info (None or dict): Any info to return

        Returns:
            2-tuple:
                - float: aggregated reward at the current timestep
                - dict: any information passed through this function or generated by this function
        """
        # Make sure info is a dict
        total_info = dict() if info is None else info
        # We'll also store individual reward split as well
        breakdown_dict = dict()
        # Aggregate rewards over all reward functions
        total_reward = 0.0
        for reward_name, reward_function in self._reward_functions.items():
            reward, reward_info = reward_function.step(self, env, action)
            total_reward += reward
            breakdown_dict[reward_name] = reward
            total_info[reward_name] = reward_info

        if self.subgoals:
            # for reward_name, reward_function in self.subgoals_reward_dict.items():
            subgoal_reward_function = self.subgoals_reward_dict[self.subgoal_ongoing]
            subgoal_reward, subgoal_reward_info = subgoal_reward_function.step(self, env, action)
            total_reward += subgoal_reward
            breakdown_dict[self.subgoal_ongoing] = subgoal_reward
            total_info[self.subgoal_ongoing] = subgoal_reward_info
            if 'done' in subgoal_reward_info.keys():
                if subgoal_reward_info['done']:
                    # 如果子目标完成，切换到下一个子目标
                    # subgoal1->subgoal2->subgoal3->...->subgoal1->subgoal2->...
                    subgoal_names = list(self.subgoal_activity_goal_conditions.keys())
                    current_index = subgoal_names.index(self.subgoal_ongoing)
                    next_index = (current_index + 1) % len(subgoal_names)
                    self.subgoal_ongoing = subgoal_names[next_index]
                return total_reward, total_info
             
        # Store breakdown dict
        total_info["reward_breakdown"] = breakdown_dict

        return total_reward, total_info
    
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
            "r_potential": 1.0,
        }

def _parse_subgoals(goal_conditions):
    """
    从goal_conditions中提取子目标定义。
    假设子目标定义类似：
    (:subgoal1 "描述" (Predicate arg1 arg2 ...))
    """
    subgoals = {}
    for cond in goal_conditions:
        # cond 应该是一个tuple或list, 形式类似:
        # (':subgoal1', '描述字符串', ('PredicateName', 'arg1', 'arg2', ...))
        if isinstance(cond, (tuple, list)) and len(cond) >= 3:
            subgoal_tag = cond[0]
            if isinstance(subgoal_tag, str) and subgoal_tag.startswith(':subgoal'):
                subgoal_name = subgoal_tag[1:]  # 去掉冒号得到 "subgoal1"
                instruction = cond[1]
                predicate = cond[2]
                subgoals[subgoal_name] = {
                    "instruction": instruction,
                    "predicate": predicate,
                }
    return subgoals

class SubgoalConditions(object):
    """
    用于解析并存储单个文件中定义的多个子目标(subgoals)。
    文件中有一个(:goal (and (:subgoal1 ...) (:subgoal2 ...) ...))结构。
    """
    def __init__(self, behavior_activity, simulator_name="omnigibson", problem_filename=None):
        self.behavior_activity = behavior_activity
        self.activity_definition = 0
        domain_name, *__ = parse_domain(simulator_name)

        # 调用 parse_subgoal_problem 解析子目标定义文件
        self.problem_name, self.parsed_objects, self.parsed_initial_conditions, self.subgoals = parse_subgoal_problem(
            behavior_activity=self.behavior_activity,
            activity_definition=self.activity_definition,
            domain_name=domain_name,
            predefined_problem=None,
            problem_filename=problem_filename
        )

    def extract_subgoals(self):
        """
        从parsed_goal_conditions中提取所有子目标定义。
        格式假设为:
        (:subgoal1 "描述" (Predicate arg1 arg2 ...))
        (:subgoal2 "描述" (Predicate arg1 arg2 ...))
        ...
        返回一个字典:
        { "subgoal1": {"instruction": "描述", "predicate": (Predicate, arg1, arg2, ...) },
          "subgoal2": {...}, ...}
        """
        return self.subgoals

def get_definition_filename(behavior_activity, activity_definition):
    """
    根据活动名与定义ID返回subgoal bddl文件路径：
    bddl_subgoals/bddl_subgoals/activity_definitions/{activity_name}/problem{activity_definition}.bddl
    """
    base_dir = "bddl_subgoals/bddl_subgoals/activity_definitions"
    filename = f"problem{activity_definition}.bddl"
    return os.path.join(base_dir, behavior_activity, filename)

def scan_tokens(filename=None, string=None):
    """
    将BDDL文件或字符串解析为嵌套列表结构的S表达式。
    简化实现，不考虑注释和特殊字符。
    """
    if filename is not None:
        with open(filename, 'r') as f:
            text = f.read().lower()
    else:
        text = string

    tokens = tokenize(text)
    expr, _ = parse_expression(tokens, 0)
    return expr

def tokenize(text):
    text = text.replace('(', ' ( ').replace(')', ' ) ')
    tokens = text.split()
    return tokens

def parse_expression(tokens, start_idx):
    if tokens[start_idx] != '(':
        # 原子token
        return tokens[start_idx], start_idx + 1

    # '(' 开始新列表
    expr = []
    idx = start_idx + 1
    while idx < len(tokens) and tokens[idx] != ')':
        subexpr, idx = parse_expression(tokens, idx)
        expr.append(subexpr)
    if idx == len(tokens):
        raise Exception("Unmatched parentheses")
    # 跳过 ')'
    idx += 1
    return expr, idx

def package_predicates(expr):
    """
    从解析后的S表达式中提取子目标信息，返回一个字典：
    {
      "subgoalX": {
        "instruction": <str>,
        "predicates": [(PredicateName, arg1, arg2, ...), ...]
      },
      ...
    }

    expr的示例结构大致如下：
    ['define',
      ['problem', 'recycling_office_papers-0'],
      [':domain', 'omnigibson'],
      [':objects', ...],
      [':init', ...],
      [':subgoal1', 'Find', 'legal_document.n.01_1',
        [':and',
          ['NextTo', '?legal_document.n.01_1', '?agent.n.01_1']
        ]
      ],
      [':subgoal2', 'Grasp', 'legal_document.n.01_1',
        [':and',
          ['AttachedTo', '?legal_document.n.01_1', '?agent.n.01_1']
        ]
      ],
      ...
    ]

    注：描述字符串会被分成多个token，这里假设描述只需要简单拼接。
    """
    subgoals = {}
    # expr[0] = 'define', 其后是各种块
    for block in expr[1:]:
        if isinstance(block, list) and len(block) > 0:
            head = block[0]
            if isinstance(head, str) and head.startswith(':subgoal'):
                # subgoal名称，如 ':subgoal1'
                subgoal_name = head[1:]  # 移除前缀':'

                # 格式：[:subgoal1, "Find", "legal_document.n.01_1", [':and', [Predicate...]]]
                # 第二个元素开始到最后第二个元素为描述token，最后一个元素为(:and ...)结构
                if len(block) < 3:
                    continue  # 没有and块的情况，跳过

                # 描述可能有多个token，需要拼接
                # 最后一个block是[:and ...]谓词列表，其余的在描述中
                logic_block = None
                instruction_tokens = []
                for elem in block[1:]:
                    if isinstance(elem, list) and len(elem) > 0 and elem[0] == ':and':
                        logic_block = elem
                        break
                    elif isinstance(elem, list) and "?" in elem[1]:
                        break
                    else:
                        instruction_tokens.append(elem)

                instruction = " ".join(instruction_tokens)

                # 解析and块中的谓词
                predicates = []
                # if logic_block and len(logic_block) > 1:
                # if 1:
                #     # and_block结构 [:and, [PredicateName, arg1, arg2...], ...]
                #     # for pred_expr in logic_block[1:]:
                #     for pred_expr in block[1:][-1]:
                #         if isinstance(pred_expr, list) and len(pred_expr) > 0:
                #             predicates.append(list(pred_expr).insert(0, logic_block[0].strip(":")))

                subgoals[subgoal_name] = {
                    "instruction": instruction,
                    "predicates": block[-1]
                }

    return subgoals


def parse_subgoal_problem(behavior_activity, activity_definition, domain_name, predefined_problem=None, problem_filename=None):
    """
    解析子目标定义的BDDL文件。

    返回:
        tuple: (problem_name, objects, initial_state, subgoals_dict)
        其中:
        - problem_name (str): 问题名称，例如 "recycling_office_papers-0"
        - objects (dict): 物体类型到物体实例列表的映射，例如 {"recycling_bin.n.01": ["recycling_bin.n.01_1"], ...}
        - initial_state (list): 初始条件谓词的列表表示
        - subgoals_dict (dict): 所有子目标定义的字典，键为子目标名（"subgoal1", "subgoal2", ...），值为
          {"description": <描述字符串>, "predicates": [(PredicateName, arg1, arg2, ...), ...]}
    """

    # 如果给定predefined_problem字符串，则直接解析，否则从文件加载
    if predefined_problem is not None:
        expr = scan_tokens(string=predefined_problem)
    else:
        if problem_filename is None:
            problem_filename = get_definition_filename(behavior_activity, activity_definition)
        expr = scan_tokens(filename=problem_filename)



    if not (isinstance(expr, list) and len(expr) > 0 and expr[0] == 'define'):
        raise Exception(f"Problem {behavior_activity} {activity_definition} does not match problem pattern")

    problem_name = 'unknown'
    objects = {}
    initial_state = []
    # 不再有单独的:goal段落，而是多个:subgoalX块，统一在package_predicates中提取
    # 因此这里不解析goal_state，而是等会用package_predicates解析subgoals

    # 遍历顶层块，获取problem_name, domain校验, objects, init等
    for block in expr[1:]:
        if not (isinstance(block, list) and len(block) > 0):
            continue
        t = block[0]
        if t == 'problem':
            # block如 ['problem', 'recycling_office_papers-0']
            if len(block) > 1:
                problem_name = block[1]
        elif t == ':domain':
            # 检查domain一致性
            if len(block) > 1 and domain_name != block[1]:
                raise Exception('Different domain specified in problem file')
        elif t == ':objects':
            # 解析 objects
            # 格式: [':objects', 'recycling_bin.n.01_1', '-', 'recycling_bin.n.01', 'floor.n.01_1', '-', 'floor.n.01', ...]
            # 解析逻辑：遇 '-' 表示前面累积的实例属于下一类型
            # 注意：给定的示例中，objects定义类似：
            # (:objects
            #   recycling_bin.n.01_1 - recycling_bin.n.01
            #   floor.n.01_1 - floor.n.01
            #   legal_document.n.01_1 legal_document.n.01_2 legal_document.n.01_3 - legal_document.n.01
            #   ...
            # )
            # 即先列出实例，然后 '-' 后面是类型。
            # 我们需要将上一个'-'前收集到的实例归到'-'后面的类型下。
            temp_objs = []
            current_type = None
            # block[1:]是objects定义的令牌
            tokens = block[1:]
            while tokens:
                token = tokens.pop(0)
                if token == '-':
                    # 下一个token应为type
                    if tokens:
                        obj_type = tokens.pop(0)
                        if obj_type not in objects:
                            objects[obj_type] = []
                        # temp_objs中的对象归属于obj_type
                        objects[obj_type].extend(temp_objs)
                        temp_objs = []
                        current_type = None
                    else:
                        # 没有下一个token，无效定义
                        break
                else:
                    # 对象实例
                    temp_objs.append(token)
            # 若最后还有temp_objs未归属到类型中则忽略(定义一般完整)
        elif t == ':init':
            # init区域 block[1:]全是初始条件谓词
            # 例如:
            # (ontop recycling_bin.n.01_1 floor.n.01_1)
            # (inroom floor.n.01_1 private_office) ...
            initial_state = block[1:]
        # :subgoalX块和:description块不用在这里解析
        # package_predicates会统一处理:subgoalX

    # 使用package_predicates解析子目标
    subgoals_dict = package_predicates(expr)

    return problem_name, objects, initial_state, subgoals_dict