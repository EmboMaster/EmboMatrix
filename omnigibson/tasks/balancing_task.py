import math

import torch as th

import omnigibson as og
import omnigibson.utils.transform_utils as T
from omnigibson.object_states import Pose
from omnigibson.objects.primitive_object import PrimitiveObject
from omnigibson.reward_functions.collision_reward import CollisionReward
from omnigibson.reward_functions.point_goal_reward import PointGoalReward
from omnigibson.reward_functions.potential_reward import PotentialReward
from omnigibson.scenes.traversable_scene import TraversableScene
from omnigibson.tasks.task_base import BaseTask
from omnigibson.termination_conditions.falling import Falling
from omnigibson.termination_conditions.max_collision import MaxCollision
from omnigibson.termination_conditions.point_goal import PointGoal
from omnigibson.termination_conditions.timeout import Timeout
from omnigibson.utils.python_utils import assert_valid_key, classproperty
from omnigibson.utils.sim_utils import land_object, test_valid_pose
from omnigibson.utils.ui_utils import create_module_logger
# import gymnasium as gym
from omnigibson.utils.numpy_utils import NumpyTypes

# Create module logger
log = create_module_logger(module_name=__name__)


# Valid point navigation reward types
POINT_NAVIGATION_REWARD_TYPES = {"l2", "geodesic"}


class BalancingTask(BaseTask):
    """
    Balancing Task
    Args:
        robot_idn (int): Which robot that this task corresponds to
        floor (int): Which floor to navigate on
        initial_pos (None or 3-array): If specified, should be (x,y,z) global initial position to place the robot
            at the start of each task episode. If None, a collision-free value will be randomly sampled
        reward_type (str): Type of reward to use. Valid options are: {"l2", "geodesic"}
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
        robot_idn=0,
        floor=0,
        initial_pos=None,
        initial_quat=None,
        goal_pos=None,
        goal_tolerance=0.5,
        goal_in_polar=False,
        path_range=None,
        visualize_goal=False,
        visualize_path=False,
        goal_height=0.06,
        waypoint_height=0.05,
        waypoint_width=0.1,
        n_vis_waypoints=10,
        reward_type="l2",
        termination_config=None,
        reward_config=None,
    ):
        # Store inputs
        self._robot_idn = robot_idn
        self._floor = floor
        self._initial_pos = initial_pos if initial_pos is None else th.tensor(initial_pos)
        self._initial_quat = initial_quat if initial_quat is None else th.tensor(initial_quat)
        self._goal_pos = goal_pos if goal_pos is None else th.tensor(goal_pos)
        self._goal_tolerance = goal_tolerance
        self._goal_in_polar = goal_in_polar
        self._path_range = path_range
        self._randomize_initial_pos = initial_pos is None
        self._randomize_initial_quat = initial_quat is None
        self._randomize_goal_pos = goal_pos is None
        self._visualize_goal = visualize_goal
        self._visualize_path = visualize_path
        self._goal_height = goal_height
        self._waypoint_height = waypoint_height
        self._waypoint_width = waypoint_width
        self._n_vis_waypoints = n_vis_waypoints
        assert_valid_key(key=reward_type, valid_keys=POINT_NAVIGATION_REWARD_TYPES, name="reward type")
        self._reward_type = reward_type

        # Create other attributes that will be filled in at runtime
        self._initial_pos_marker = None
        self._goal_pos_marker = None
        self._waypoint_markers = None
        self._path_length = None
        self._current_robot_pos = None
        self._geodesic_dist = None

        # Run super
        super().__init__(termination_config=termination_config, reward_config=reward_config)

    def _create_termination_conditions(self):
        # Initialize termination conditions dict and fill in with MaxCollision, Timeout, Falling, and PointGoal
        terminations = dict()
        terminations["max_collision"] = MaxCollision(max_collisions=self._termination_config["max_collisions"])
        terminations["timeout"] = Timeout(max_steps=self._termination_config["max_steps"])
        terminations["falling"] = Falling(
            robot_idn=self._robot_idn, fall_height=self._termination_config["fall_height"]
        )
        terminations["pointgoal"] = PointGoal(
            robot_idn=self._robot_idn,
            distance_tol=self._goal_tolerance,
            distance_axes="xy",
        )

        return terminations

    def _create_reward_functions(self):
        # Initialize reward functions dict and fill in with Potential, Collision, and PointGoal rewards
        rewards = dict()

        rewards["potential"] = PotentialReward(
            potential_fcn=self.get_potential,
            r_potential=self._reward_config["r_potential"],
        )
        rewards["collision"] = CollisionReward(r_collision=self._reward_config["r_collision"])
        rewards["pointgoal"] = PointGoalReward(
            pointgoal=self._termination_conditions["pointgoal"],
            r_pointgoal=self._reward_config["r_pointgoal"],
        )

        return rewards