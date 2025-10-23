from pytorch3d.transforms import quaternion_to_matrix, matrix_to_quaternion
from omnigibson.sensors import VisionSensor
from pytorch3d.transforms import quaternion_to_matrix, matrix_to_quaternion, so3_exponential_map
import torch

def _get_value_ObjectsInFOVOfRobot(robot, env):
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
            if info == {}:
                continue
            # import pdb; pdb.set_trace()
            obj_names.extend([name for name in info["seg_instance"].values() if name not in names_to_exclude])
    return [x for x in [env.scene.object_registry("name", x) for x in obj_names] if x is not None]

def _get_target_object(env, obj_env_name):
    # if self.target_object is None:
    # self 是否包含 target_object 属性
    # if not hasattr("target_object") or self.target_object is None:
    #     self.target_object = env.scene.object_registry("name", obj_env_name)
    # return self.target_object
    return env.scene.object_registry("name", obj_env_name)

def _get_robot_object_relative_pose(robot, target_object):
    # Get robot's eye position and orientation (torch.tensor)
    robot_eye_position = robot.sensors['robot0:eyes:Camera:0'].get_position()  # torch.tensor([x, y, z])
    robot_eye_orientation = robot.sensors['robot0:eyes:Camera:0'].get_orientation()  # torch.tensor([x, y, z, w])
    
    # Get target object's position and orientation (torch.tensor)
    target_object_position = target_object.get_position()  # torch.tensor([x, y, z])
    target_object_orientation = target_object.get_orientation()  # torch.tensor([x, y, z, w])
    
    # Convert robot's eye orientation (quaternion) to rotation matrix
    robot_eye_rotation_matrix = quaternion_to_matrix(robot_eye_orientation)
    
    # Compute the inverse of robot's eye rotation matrix
    robot_eye_rotation_matrix_inv = torch.linalg.inv(robot_eye_rotation_matrix)
    
    # Compute relative position
    relative_position = robot_eye_rotation_matrix_inv @ (target_object_position - robot_eye_position)
    
    # Convert target object's orientation (quaternion) to rotation matrix
    target_object_rotation_matrix = quaternion_to_matrix(target_object_orientation)
    
    # Compute relative orientation in rotation matrix form
    relative_rotation_matrix = robot_eye_rotation_matrix_inv @ target_object_rotation_matrix
    
    # Convert relative rotation matrix back to quaternion
    relative_orientation = matrix_to_quaternion(relative_rotation_matrix)
    
    return torch.cat([relative_position, relative_orientation], 0)