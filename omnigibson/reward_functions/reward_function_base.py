from abc import ABCMeta, abstractmethod
from copy import deepcopy

from omnigibson.utils.python_utils import Registerable, classproperty

REGISTERED_REWARD_FUNCTIONS = dict()


class BaseRewardFunction(Registerable, metaclass=ABCMeta):
    """
    Base RewardFunction class
    Reward-specific reset and get_reward methods are implemented in subclasses
    """

    def __init__(self):
        # Store internal vars that will be filled in at runtime
        self._reward = None
        self._info = None

    @abstractmethod
    def _step(self, task, env, action):
        """
        Step the reward function and compute the reward at the current timestep. Overwritten by subclasses.

        Args:
            task (BaseTask): Task instance
            env (Environment): Environment instance
            action (n-array): 1D flattened array of actions executed by all agents in the environment

        Returns:
            2-tuple:
                - bool: computed reward
                - dict: any reward-related information for this specific reward
        """
        raise NotImplementedError()

    def step(self, task, env, action):
        """
        Step the reward function and compute the reward at the current timestep.

        Args:
            task (BaseTask): Task instance
            env (Environment): Environment instance
            action (n-array): 1D flattened array of actions executed by all agents in the environment

        Returns:
            2-tuple:
                - bool: computed reward
                - dict: any reward-related information for this specific reward
        """
        # Step internally and store output
        self._reward, self._info = self._step(task=task, env=env, action=action)

        # Return reward and a copy of the info
        return self._reward, deepcopy(self._info)

    def reset(self, task, env):
        """
        Reward function-specific reset

        Args:
            task (BaseTask): Task instance
            env (Environment): Environment instance
        """
        # Reset internal vars
        self._reward = None
        self._info = None

    @property
    def reward(self):
        """
        Returns:
            float: Current reward for this reward function
        """
        assert self._reward is not None, "At least one step() must occur before reward can be calculated!"
        return self._reward

    @property
    def info(self):
        """
        Returns:
            dict: Current info for this reward function
        """
        assert self._info is not None, "At least one step() must occur before info can be calculated!"
        return self._info

    @classproperty
    def _do_not_register_classes(cls):
        # Don't register this class since it's an abstract template
        classes = super()._do_not_register_classes
        classes.add("BaseRewardFunction")
        return classes

    @classproperty
    def _cls_registry(cls):
        # Global registry
        global REGISTERED_REWARD_FUNCTIONS
        return REGISTERED_REWARD_FUNCTIONS

    def _get_target_object(self, env, obj_env_name):
        if not hasattr(self, 'target_object') or self.target_object is None:
            self.subgoalname = list(env.task.subgoal_activity_goal_conditions.keys())[0]
            # self.target_object = env.scene.object_registry("name", obj_env_name)
            target_id = env.task.subgoal_activity_goal_conditions[self.subgoalname][0].terms[-1] if 'agent' not in env.task.subgoal_activity_goal_conditions[self.subgoalname][0].terms[-1] else env.task.subgoal_activity_goal_conditions[self.subgoalname][0].terms[1]
            target_object = env.scene.object_registry("name", env.scene._scene_info_meta_inst_to_name[target_id])
        return target_object