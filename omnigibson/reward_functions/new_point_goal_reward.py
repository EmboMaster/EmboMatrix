import math

from omnigibson.reward_functions.reward_function_base import BaseRewardFunction
from omnigibson.utils.transform_utils import l2_distance
from omnigibson.sensors import VisionSensor
from omnigibson.utils.motion_planning_utils import astar

class NewPointGoalReward(BaseRewardFunction):
    """
    Reward function for finding and approaching a target object.

    1. Finding the object:
        - Encourage the robot to move closer to the target object.
        - Provide a distance-based reward.
        - Give a bonus reward if the target object is detected in the robot's field of view.

    2. Approaching the object:
        - Activate when the object is in the robot's field of view and close enough.
        - Encourage the robot's end-effector (EEF) to move to a pre-grasp position.
        - Provide a distance-based reward for the EEF and a fixed reward for reaching the threshold distance.

    Attributes:
        obj_name (str): Name of the target object.
        alpha (float): Coefficient for the distance-based reward.
        r_detect (float): Bonus reward for detecting the object in the field of view.
        r_approach_threshold (float): Fixed reward for reaching the pre-grasp position.
        approach_distance_threshold (float): Distance threshold for enabling the approach phase.
        grasp_distance_threshold (float): Distance threshold for detecting a pre-grasp position.
    """

    def __init__(
        self,
        obj_name,
        alpha=10.0,
        r_detect=2.0,
        r_carry=2.0,
        r_approach_threshold=10.0,
        approach_distance_threshold=1.0,
        grasp_distance_threshold=0.1,
        r_penalty_separation=1.1,
        carry_threshold=0.1,
        eef_distance_threshold=0.1,
    ):
        self.obj_name = obj_name
        self.target_object = None
        self.alpha = alpha
        self.r_detect = r_detect
        self.r_approach_threshold = r_approach_threshold
        self.approach_distance_threshold = approach_distance_threshold
        self.grasp_distance_threshold = grasp_distance_threshold

        super().__init__()

    def _step(self, task, env, action):
        reward = 0.0
        info = {}
        
        robot = env.robots[0]
        obj_env_name = env.scene._scene_info_meta_inst_to_name[self.obj_name]
        target_object = self._get_target_object(env, obj_env_name)

        robot_position = robot.get_position()
        object_position = target_object.get_position()
        robot_object_distance = l2_distance(robot_position, object_position)
        # short_path, geo_dis = env.scene.trav_map.get_shortest_path(0, robot_position[:2], object_position[:2], entire_path=True)
        # print(f"Robot-Object distance: {robot_object_distance}")
        # Finding phase: Distance-based reward
        distance_reward = self.alpha * math.exp(-robot_object_distance)
        reward += distance_reward
        info["distance_reward"] = distance_reward
        info['done'] = False

        # Check if the object is in the field of view
        objects_in_fov = [
            # obj.name for obj in env.scene.get_objects_in_robot_fov(robot)
            obj.name for obj in self._get_value_ObjectsInFOVOfRobot(robot, env)
        ]
        if target_object.name in objects_in_fov:
            reward += self.r_detect
            info["vision_reward"] = self.r_detect

            if robot_object_distance < 0.1:
                info['done'] = True
                

            # Approaching phase: Activate if within threshold
            # if robot_object_distance <= self.approach_distance_threshold:
            #     eef_position = robot.get_end_effector_position()
            #     eef_object_distance = l2_distance(eef_position, object_position)

            #     # Distance-based reward for EEF
            #     eef_reward = self.alpha * math.exp(-eef_object_distance)
            #     reward += eef_reward
            #     info["eef_distance_reward"] = eef_reward

            #     # Fixed reward for reaching grasping threshold
            #     if eef_object_distance <= self.grasp_distance_threshold:
            #         reward += self.r_approach_threshold
            #         info["grasp_reward"] = self.r_approach_threshold
        
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

    def _get_value_ObjectsInFOVOfRobot(self, robot, env):
        """
        Gets all objects in the robot's field of view.

        Returns:
            list: List of objects in the robot's field of view
        """
        if not any(isinstance(sensor, VisionSensor) for sensor in robot.sensors.values()):
            raise ValueError("No vision sensors found on robot.")
        obj_names = []
        names_to_exclude = set(["background", "unlabelled"])
        for sensor in robot.sensors.values():
            if isinstance(sensor, VisionSensor):
                _, info = sensor.get_obs()
                obj_names.extend([name for name in info["seg_instance"].values() if name not in names_to_exclude])
        return [x for x in [env.scene.object_registry("name", x) for x in obj_names] if x is not None]