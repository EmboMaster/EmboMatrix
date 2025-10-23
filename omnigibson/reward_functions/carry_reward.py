import math
from omnigibson.reward_functions.reward_function_base import BaseRewardFunction
from omnigibson.reward_functions.new_point_goal_reward import NewPointGoalReward
from omnigibson.utils.transform_utils import l2_distance

class CarryReward(NewPointGoalReward):
    """
    Reward function for robot navigation and carrying an object.

    This class extends NewPointGoalReward and adds carry-specific logic:
    - Reward for keeping the object close to the end-effector (EEF) while moving.
    - Penalty for separating the object from the EEF.
    - Can still reward the robot for approaching the goal object.

    Attributes:
        r_carry (float): Reward for keeping the object and the EEF together.
        r_penalty_separation (float): Penalty for separating the object from the EEF.
        carry_threshold (float): Threshold for how far the object can move away from the EEF before penalizing.
        eef_distance_threshold (float): Threshold distance for how close the object should be to the EEF for the carry reward.
    """

    def __init__(
        self,
        obj_name,
        r_carry=5.0,
        r_penalty_separation=10.0,
        carry_threshold=0.2,
        eef_distance_threshold=0.1,
        alpha=1.0,
        r_detect=5.0,
        r_approach_threshold=10.0,
        approach_distance_threshold=1.0,
        grasp_distance_threshold=0.1,
    ):
        super().__init__(
            obj_name,
            alpha,
            r_detect,
            r_approach_threshold,
            approach_distance_threshold,
            grasp_distance_threshold,
        )
        self.r_carry = r_carry
        self.r_penalty_separation = r_penalty_separation
        self.carry_threshold = carry_threshold
        self.eef_distance_threshold = eef_distance_threshold

    def _step(self, task, env, action):
        reward, info = super()._step(task, env, action)

        robot = env.robots[0]

        # Check if the robot is carrying an object
        if robot._ag_obj_in_hand[robot.default_arm]:
            carried_object = robot._ag_obj_in_hand[robot.default_arm]
            object_position = carried_object.get_position()
            eef_position = robot.get_end_effector_position()

            # Reward for keeping the object close to the EEF
            if l2_distance(eef_position, object_position) <= self.eef_distance_threshold:
                reward += self.r_carry
                info["carry_reward"] = self.r_carry
            # Penalty for separating the object from the EEF
            elif l2_distance(eef_position, object_position) > self.carry_threshold:
                reward -= self.r_penalty_separation
                info["separation_penalty"] = self.r_penalty_separation

        return reward, info

    def reset(self, task, env):
        """
        Carry reward-specific reset

        Args:
            task (BaseTask): Task instance
            env (Environment): Environment instance
        """
        super().reset(task, env)