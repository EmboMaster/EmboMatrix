import math

import torch as th

import omnigibson.utils.transform_utils as T
from omnigibson.reward_functions.reward_function_base import BaseRewardFunction
from omnigibson.utils.motion_planning_utils import detect_robot_collision_in_sim
from omnigibson.utils.transform_utils import l2_distance
from omnigibson.sensors import VisionSensor
from pytorch3d.transforms import quaternion_to_matrix, matrix_to_quaternion


class GraspReward(BaseRewardFunction):
    """
    A composite reward function for grasping tasks. This reward function not only evaluates the success of object grasping
    but also considers various penalties and efficiencies.

    The reward is calculated based on several factors:
    - Grasping reward: A positive reward is given if the robot is currently grasping the specified object.
    - Distance reward: A reward based on the inverse exponential distance between the end-effector and the object.
    - Regularization penalty: Penalizes large magnitude actions to encourage smoother and more energy-efficient movements.
    - Position and orientation penalties: Discourages excessive movement of the end-effector.
    - Collision penalty: Penalizes collisions with the environment or other objects.

    Attributes:
        obj_name (str): Name of the object to grasp.
        dist_coeff (float): Coefficient for the distance reward calculation.
        grasp_reward (float): Reward given for successfully grasping the object.
        collision_penalty (float): Penalty incurred for any collision.
        eef_position_penalty_coef (float): Coefficient for the penalty based on end-effector's position change.
        eef_orientation_penalty_coef (float): Coefficient for the penalty based on end-effector's orientation change.
        regularization_coef (float): Coefficient for penalizing large actions.
    """

    def __init__(
        self,
        obj_name,
        dist_coeff,
        grasp_reward,
        collision_penalty,
        eef_position_penalty_coef,
        eef_orientation_penalty_coef,
        regularization_coef,
        alpha=1.0,
        r_detect=2.0,
        r_approach_threshold=10.0,
        approach_distance_threshold=1.0,
        grasp_distance_threshold=0.1,
    ):
        # Store internal vars
        self.prev_grasping = False
        self.prev_eef_pos = None
        self.prev_eef_rot = None
        self.obj_name = obj_name
        self.obj = None
        self.dist_coeff = dist_coeff
        self.grasp_reward = grasp_reward
        self.collision_penalty = collision_penalty
        self.eef_position_penalty_coef = eef_position_penalty_coef
        self.eef_orientation_penalty_coef = eef_orientation_penalty_coef
        self.regularization_coef = regularization_coef
        self.alpha = alpha
        self.r_detect = r_detect
        self.r_approach_threshold = r_approach_threshold
        self.approach_distance_threshold = approach_distance_threshold
        self.grasp_distance_threshold = grasp_distance_threshold
        # Run super
        super().__init__()

    # def _step(self, task, env, action):
    #     self.obj = env.scene.object_registry("name", self.obj_name) if self.obj is None else self.obj

    #     robot = env.robots[0]
    #     obj_in_hand = robot._ag_obj_in_hand[robot.default_arm]
    #     current_grasping = obj_in_hand == self.obj

    #     info = {"grasp_success": current_grasping}

    #     # Reward varying based on combination of whether the robot was previously grasping the desired object
    #     # and is currently grasping the desired object
    #     reward = 0.0

    #     # Penalize large actions
    #     action_mag = th.sum(th.abs(action))
    #     regularization_penalty = -(action_mag * self.regularization_coef)
    #     reward += regularization_penalty
    #     info["regularization_penalty_factor"] = action_mag
    #     info["regularization_penalty"] = regularization_penalty

    #     # Penalize based on the magnitude of the action
    #     eef_pos = robot.get_eef_position(robot.default_arm)
    #     info["position_penalty_factor"] = 0.0
    #     info["position_penalty"] = 0.0
    #     if self.prev_eef_pos is not None:
    #         eef_pos_delta = T.l2_distance(self.prev_eef_pos, eef_pos)
    #         position_penalty = -eef_pos_delta * self.eef_position_penalty_coef
    #         reward += position_penalty
    #         info["position_penalty_factor"] = eef_pos_delta
    #         info["position_penalty"] = position_penalty
    #     self.prev_eef_pos = eef_pos

    #     eef_quat = robot.get_eef_orientation(robot.default_arm)
    #     info["rotation_penalty_factor"] = 0.0
    #     info["rotation_penalty"] = 0.0
    #     if self.prev_eef_rot is not None:
    #         delta_rot = T.get_orientation_diff_in_radian(self.prev_eef_rot, eef_quat)
    #         rotation_penalty = -delta_rot * self.eef_orientation_penalty_coef
    #         reward += rotation_penalty
    #         info["rotation_penalty_factor"] = delta_rot.item()
    #         info["rotation_penalty"] = rotation_penalty.item()
    #     self.prev_eef_rot = eef_quat

    #     # Penalize robot for colliding with an object
    #     info["collision_penalty_factor"] = 0.0
    #     info["collision_penalty"] = 0.0
    #     if detect_robot_collision_in_sim(robot, filter_objs=[self.obj]):
    #         reward += -self.collision_penalty
    #         info["collision_penalty_factor"] = 1.0
    #         info["collision_penalty"] = -self.collision_penalty

    #     # obj_env_name = env.scene._scene_info_meta_inst_to_name[self.obj_name]
    #     target_object = self._get_target_object(env, self.obj_name)

    #     robot_position = robot.get_position()
    #     object_position = target_object.get_position()
    #     robot_object_distance = l2_distance(robot_position, object_position)

    #     # Finding phase: Distance-based reward
    #     distance_reward = self.alpha * math.exp(-robot_object_distance)
    #     reward += distance_reward
    #     info["distance_reward"] = distance_reward

    #     # Check if the object is in the field of view
    #     objects_in_fov = [
    #         # obj.name for obj in env.scene.get_objects_in_robot_fov(robot)
    #         obj.name for obj in self._get_value_ObjectsInFOVOfRobot(robot, env)
    #     ]
    #     if target_object.name in objects_in_fov:
    #         reward += self.r_detect
    #         info["vision_reward"] = self.r_detect

    #         # Approaching phase: Activate if within threshold
    #         if robot_object_distance <= self.approach_distance_threshold:
    #             # eef_position = robot.get_end_effector_position()
    #             eef_object_distance = l2_distance(eef_pos, object_position)

    #             # Distance-based reward for EEF
    #             eef_reward = self.alpha * math.exp(-eef_object_distance)
    #             reward += eef_reward
    #             info["eef_distance_reward"] = eef_reward

    #             # Fixed reward for reaching grasping threshold
    #             if eef_object_distance <= self.grasp_distance_threshold:
    #                 reward += self.r_approach_threshold
    #                 info["grasp_reward"] = self.r_approach_threshold

    #     # If we're not currently grasping
    #     info["grasp_reward_factor"] = 0.0
    #     info["grasp_reward"] = 0.0
    #     info["pregrasp_dist"] = 0.0
    #     info["pregrasp_dist_reward_factor"] = 0.0
    #     info["pregrasp_dist_reward"] = 0.0
    #     info["postgrasp_dist"] = 0.0
    #     info["postgrasp_dist_reward_factor"] = 0.0
    #     info["postgrasp_dist_reward"] = 0.0
    #     if not current_grasping:
    #         # TODO: If we dropped the object recently, penalize for that
    #         obj_center = self.obj.get_position_orientation()[0]
    #         dist = T.l2_distance(eef_pos, obj_center)
    #         dist_reward = math.exp(-dist) * self.dist_coeff
    #         reward += dist_reward
    #         info["pregrasp_dist"] = dist
    #         info["pregrasp_dist_reward_factor"] = math.exp(-dist)
    #         info["pregrasp_dist_reward"] = dist_reward
    #     else:
    #         # We are currently grasping - first apply a grasp reward
    #         reward += self.grasp_reward
    #         info["grasp_reward_factor"] = 1.0
    #         info["grasp_reward"] = self.grasp_reward

    #         # Then apply a distance reward to take us to a tucked position
    #         robot_center = robot.links["torso_lift_link"].get_position_orientation()[0]
    #         obj_center = self.obj.get_position_orientation()[0]
    #         dist = T.l2_distance(robot_center, obj_center)
    #         dist_reward = math.exp(-dist) * self.dist_coeff
    #         reward += dist_reward
    #         info["postgrasp_dist"] = dist
    #         info["postgrasp_dist_reward_factor"] = math.exp(-dist)
    #         info["postgrasp_dist_reward"] = dist_reward

    #     self.prev_grasping = current_grasping

    #     return reward, info
    def _step(self, task, env, action):
        self.obj = env.scene.object_registry("name", self.obj_name) if self.obj is None else self.obj

        robot = env.robots[0]
        obj_in_hand = robot._ag_obj_in_hand[robot.default_arm]
        current_grasping = obj_in_hand == self.obj

        info = {"grasp_success": current_grasping}

        # Reward varying based on combination of whether the robot was previously grasping the desired object
        # and is currently grasping the desired object
        reward = 0.0

        # Penalize large actions
        # action_mag = th.sum(th.abs(action))
        action_mag = th.sum(th.abs(action[2:])) 
        regularization_penalty = -(action_mag * self.regularization_coef)
        reward += regularization_penalty
        info["regularization_penalty_factor"] = action_mag
        info["regularization_penalty"] = regularization_penalty

        # Penalize based on the magnitude of the action
        eef_pos = robot.get_eef_position(robot.default_arm)
        info["position_penalty_factor"] = 0.0
        info["position_penalty"] = 0.0
        if self.prev_eef_pos is not None:
            eef_pos_delta = T.l2_distance(self.prev_eef_pos, eef_pos)
            position_penalty = -eef_pos_delta * self.eef_position_penalty_coef
            reward += position_penalty
            info["position_penalty_factor"] = eef_pos_delta
            info["position_penalty"] = position_penalty
        self.prev_eef_pos = eef_pos

        eef_quat = robot.get_eef_orientation(robot.default_arm)
        info["rotation_penalty_factor"] = 0.0
        info["rotation_penalty"] = 0.0
        if self.prev_eef_rot is not None:
            delta_rot = T.get_orientation_diff_in_radian(self.prev_eef_rot, eef_quat)
            rotation_penalty = -delta_rot * self.eef_orientation_penalty_coef
            reward += rotation_penalty
            info["rotation_penalty_factor"] = delta_rot.item()
            info["rotation_penalty"] = rotation_penalty.item()
        self.prev_eef_rot = eef_quat

        # Penalize robot for colliding with an object
        info["collision_penalty_factor"] = 0.0
        info["collision_penalty"] = 0.0
        if detect_robot_collision_in_sim(robot, filter_objs=[self.obj]):
        # if detect_robot_collision_in_sim(robot):
            reward += -self.collision_penalty
            info["collision_penalty_factor"] = 1.0
            info["collision_penalty"] = -self.collision_penalty

        # obj_env_name = env.scene._scene_info_meta_inst_to_name[self.obj_name]
        target_object = self._get_target_object(env, self.obj_name)

        robot_position = robot.get_position()
        object_position = target_object.get_position()
        robot_object_distance = l2_distance(robot_position, object_position)

        # Finding phase: Distance-based reward

        # # # calculate_angle_reward
        # camera_pos = robot.sensors['robot0:eyes:Camera:0'].get_position()
        # camera_quat = robot.sensors['robot0:eyes:Camera:0'].get_orientation()
        # camera_quat_matrix = quaternion_to_matrix(camera_quat)
        # camera_quat_matrix_inv = th.linalg.inv(camera_quat_matrix)
     
        # object_vector = camera_quat_matrix_inv @ (camera_pos - object_position)
        # x,y,z = object_vector
    
        # object_norm = th.norm(object_vector)
        # cos_theta = z / object_norm
        
        # cos_theta = th.clamp(cos_theta, -1.0, 1.0)
        # theta = th.arccos(cos_theta)
        # theta_degrees = theta * (180.0 / th.pi)
        # info['theta'] = theta_degrees
        # info["vision_angle_penalty"] = -theta * 0.5
        # reward -= theta * 0.5

        # # Check if the object is in the field of view
        # objects_in_fov = [
        #     # obj.name for obj in env.scene.get_objects_in_robot_fov(robot)
        #     obj.name for obj in self._get_value_ObjectsInFOVOfRobot(robot, env)
        # ]
        # info["vision_reward"] = 0
        # info["vision_center_penalty"] = 0
        # if target_object.name in objects_in_fov:
        #     reward += self.r_detect
        #     info["vision_reward"] = self.r_detect

        #     view_center_dis = self._get_distance_to_view_center(robot, env, self.obj_name).to(reward.device)
        #     reward -= view_center_dis * 0.5
        #     info["vision_center_penalty"] = -view_center_dis * 0.5
        # else:
        #     reward -= 0.5
        #     info["vision_center_penalty"] = -0.5

        # If we're not currently grasping
        info["grasp_reward_factor"] = 0.0
        info["grasp_reward"] = 0.0
        info["pregrasp_dist"] = 0.0
        info["pregrasp_dist_reward_factor"] = 0.0
        info["pregrasp_dist_reward"] = 0.0
        info["postgrasp_dist"] = 0.0
        info["postgrasp_dist_reward_factor"] = 0.0
        info["postgrasp_dist_reward"] = 0.0
        if not current_grasping:
            # TODO: If we dropped the object recently, penalize for that
            obj_center = self.obj.get_position_orientation()[0]
            dist = T.l2_distance(eef_pos, obj_center)
            dist_reward = math.exp(-dist) * self.dist_coeff
            reward += dist_reward
            info["pregrasp_dist"] = dist
            info["pregrasp_dist_reward_factor"] = math.exp(-dist)
            info["pregrasp_dist_reward"] = dist_reward

           

            # # Approaching phase: Activate if within threshold
            # if robot_object_distance <= self.approach_distance_threshold:
            #     distance_reward = 0
            #     # eef_position = robot.get_end_effector_position()
            #     eef_object_distance = l2_distance(eef_pos, object_position)

            #     # Distance-based reward for EEF
            #     eef_reward = self.dist_coeff * math.exp(-eef_object_distance)
            #     reward += eef_reward
            #     info["eef_distance_reward"] = eef_reward

            #     # Fixed reward for reaching grasping threshold
            #     if eef_object_distance <= self.grasp_distance_threshold:
            #         reward += self.r_approach_threshold
            #         info["grasp_reward"] = self.r_approach_threshold
            # else:
            #     distance_reward = self.dist_coeff * math.exp(-robot_object_distance)
            # reward += distance_reward
            # info["distance_reward"] = distance_reward
        else:
            # We are currently grasping - first apply a grasp reward
            reward += self.grasp_reward
            info["grasp_reward_factor"] = 1.0
            info["grasp_reward"] = self.grasp_reward

            # Then apply a distance reward to take us to a tucked position
            robot_center = robot.links["torso_lift_link"].get_position_orientation()[0]
            obj_center = self.obj.get_position_orientation()[0]
            dist = T.l2_distance(robot_center, obj_center)
            dist_reward = math.exp(-dist) * self.dist_coeff
            reward += dist_reward
            info["postgrasp_dist"] = dist
            info["postgrasp_dist_reward_factor"] = math.exp(-dist)
            info["postgrasp_dist_reward"] = dist_reward

        self.prev_grasping = current_grasping
        # reward = max(-1.0, min(reward, 1.0)) 
        return reward, info
    
    def reset(self, task, env):
        """
        Reward function-specific reset

        Args:
            task (BaseTask): Task instance
            env (Environment): Environment instance
        """
        super().reset(task, env)
        self.prev_grasping = False
        self.prev_eef_pos = None
        self.prev_eef_rot = None

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

    def _get_distance_to_view_center(self, robot, env, obj_name):
        """
        Gets the center of object.

        Returns:
            list: List of objects in the robot's field of view
        """
        if not any(isinstance(sensor, VisionSensor) for sensor in robot.sensors.values()):
            raise ValueError("No vision sensors found on robot.")
        distance_ratio = 1.0
        for sensor in robot.sensors.values():
            if isinstance(sensor, VisionSensor):
                obs, info = sensor.get_obs()
                obs_instance_seg = obs['seg_instance']

                cls_id = -1
                for k,v in info['seg_instance'].items():
                    if v == obj_name:
                        cls_id = k

                height, width = obs_instance_seg.shape
                view_center_x, view_center_y = width / 2, height / 2
                y_coords, x_coords = th.where(obs_instance_seg == cls_id)

                if len(x_coords) == 0:
                    continue

                center_x = x_coords.float().mean()
                center_y = y_coords.float().mean()
                distance = th.sqrt((center_x - view_center_x) ** 2 + (center_y - view_center_y) ** 2)
                distance_ratio = distance / (height / 2)
                distance_ratio = th.clamp(distance_ratio, 0.0, 1.0)
        return distance_ratio
