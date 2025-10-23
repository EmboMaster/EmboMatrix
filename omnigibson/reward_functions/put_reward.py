from omnigibson.reward_functions.reward_function_base import BaseRewardFunction
from omnigibson.utils.transform_utils import l2_distance
import math

class PutReward(BaseRewardFunction):
    """
    Reward function for placing an object at the specified target object's position.

    The reward is calculated based on:
    - Distance-based reward: Encourages the robot to move the object close to the target object's position.
    - Precision reward: If the object is placed within a small threshold of the target object's position, the robot gets a larger reward.
    - Penalty for incorrect placement: Penalizes the robot if the object is far from the target position.

    Attributes:
        obj_name (str): Name of the target object to place the object near.
        alpha (float): Coefficient for the distance-based reward.
        r_precision (float): Reward for placing the object within a small distance from the target object's position.
        r_penalty (float): Penalty for incorrect placement or when the object is too far from the target position.
        precision_threshold (float): The distance threshold for giving the precision reward.
    """

    def __init__(
        self,
        obj_name,
        alpha=1.0,
        r_precision=10.0,
        r_penalty=-5.0,
        precision_threshold=0.05,
    ):
        self.obj_name = obj_name
        self.target_object = None
        self.alpha = alpha
        self.r_precision = r_precision
        self.r_penalty = r_penalty
        self.precision_threshold = precision_threshold

        super().__init__()

    def _get_target_position(self, env):
        target_scene_name = env.scene._scene_info_meta_inst_to_name[self.obj_name]
        target_object = self._get_target_object(env, target_scene_name)
        return target_object.get_position()

    def _step(self, task, env, action):
        reward = 0.0
        info = {}

        robot = env.robots[0]
        target_position = self._get_target_position(env)

        # Get the position of the object being placed
        obj = robot._ag_obj_in_hand[robot.default_arm]  # Assuming object is in the robot's hand
        obj_position = obj.get_position()

        # Calculate the distance between the object and the target position
        distance_to_target = l2_distance(obj_position, target_position)

        # Distance-based reward
        distance_reward = self.alpha * math.exp(-distance_to_target)
        reward += distance_reward
        info["distance_reward"] = distance_reward

        # Precision reward: If the object is placed within a small threshold, reward the robot
        if distance_to_target <= self.precision_threshold:
            reward += self.r_precision
            info["precision_reward"] = self.r_precision

        # Penalty for incorrect placement: If the object is too far, penalize the robot
        if distance_to_target > self.precision_threshold:
            reward += self.r_penalty
            info["placement_penalty"] = self.r_penalty

        return reward, info

    def reset(self, task, env):
        """
        Reward function-specific reset

        Args:
            task (BaseTask): Task instance
            env (Environment): Environment instance
        """
        super().reset(task, env)
        self.target_object = None
