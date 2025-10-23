import omnigibson.utils.transform_utils as T
from omnigibson.reward_functions.reward_function_base import BaseRewardFunction
from omnigibson.sensors import VisionSensor




_IN_REACH_DISTANCE_THRESHOLD = 2.0

class FindTargetReward(BaseRewardFunction):
    """
    Finding target reward
    Guide the robot to find the target object and approach it.

    Args:
        robot_idn (int): robot identifier to evaluate distance to target. Default is 0.
        r_find (float): reward for getting closer to the target.
        alpha (float): scaling factor for the exponential reward.
        distance_tol (float): Distance (m) tolerance between robot and target object to be considered as 'close enough'.
        vision_reward (float): additional reward if the robot detects the object (through vision or similar).
    """

    def __init__(self, robot_idn=0, r_find=1.0, alpha=1.0, distance_tol=0.1, vision_reward=5.0):
        # Store internal vars
        self._robot_idn = robot_idn
        self._r_find = r_find
        self._alpha = alpha
        self._distance_tol = distance_tol
        self._vision_reward = vision_reward

        # Run super
        super().__init__()

    def _step(self, task, env, action):
        # Get the position of the robot and the target object
        robot_pos = env.robots[self._robot_idn].get_base_position()
        target_pos = task.target_pos  # target position (assuming task has a target_pos attribute)

        # Calculate the distance between the robot and the target object
        dist = T.l2_distance(robot_pos, target_pos)

        # Reward is an exponentially decaying function of distance
        reward = self._r_find * self._alpha * (2.71828 ** (-dist))

        # If the robot is close enough to the target, add vision detection reward
        if dist < self._distance_tol:
            reward += self._vision_reward

        return reward, {}

    def _get_fov_value(self, env):
        """
        Gets all objects in the robot's field of view.

        Returns:
            list: List of objects in the robot's field of view
        """
        if not any(isinstance(sensor, VisionSensor) for sensor in env.robot.sensors.values()):
            raise ValueError("No vision sensors found on robot.")
        obj_names = []
        names_to_exclude = set(["background", "unlabelled"])
        for sensor in env.robot.sensors.values():
            if isinstance(sensor, VisionSensor):
                _, info = sensor.get_obs()
                obj_names.extend([name for name in info["seg_instance"].values() if name not in names_to_exclude])
        return [x for x in [env.obj.scene.object_registry("name", x) for x in obj_names] if x is not None]

class ApproachTargetReward(BaseRewardFunction):
    """
    Approaching target reward
    Guide the robot's end-effector to approach the target object for grasping.

    Args:
        robot_idn (int): robot identifier to evaluate distance to target. Default is 0.
        r_approach (float): reward for approaching the target.
        threshold (float): Threshold distance for end-effector to be considered 'close enough' to the target.
    """

    def __init__(self, robot_idn=0, r_approach=2.0, threshold=0.1):
        # Store internal vars
        self._robot_idn = robot_idn
        self._r_approach = r_approach
        self._threshold = threshold

        # Run super
        super().__init__()

    def _step(self, task, env, action):
        # Get the position of the end-effector and the target object
        eef_pos = env.robots[self._robot_idn].get_eef_position()
        target_pos = task.target_pos  # target position (assuming task has a target_pos attribute)

        # Calculate the distance between the end-effector and the target object
        dist = T.l2_distance(eef_pos, target_pos)

        # Reward is based on how close the end-effector is to the target object
        reward = self._r_approach * (2.71828 ** (-dist))

        # If the end-effector is within the threshold, give a fixed reward
        if dist < self._threshold:
            reward += 5.0  # Fixed reward for being in a pre-grasp position

        return reward, {}