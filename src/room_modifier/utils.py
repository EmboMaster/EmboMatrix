import json
import math
import torch as th

def read_position_config(input_file_path,object_list):
    
    with open(input_file_path, 'r') as file:
        data = json.load(file)
    data_registry = data.get("state", {}).get("object_registry", {})

    obj_position_config = {
        obj_id: details for obj_id, details in data_registry.items() if obj_id in object_list
    }
    
    return obj_position_config,data
 

def update_position_config(output_file_path,object_list,data,modified_position_dict):
    updated_obj_position_config = {}
    data_registry = data.get("state", {}).get("object_registry", {})
    for obj_id, details in data_registry.items():
        if obj_id in object_list:
            # 如果在 object_list 中，使用函数 f 更新
            updated_obj_position_config[obj_id] = modified_position_dict.get(obj_id, details)
        else:
            # 如果不在 object_list 中，保留原始内容
            updated_obj_position_config[obj_id] = details
    if "state" in data and "object_registry" in data["state"]:
        data["state"]["object_registry"] = updated_obj_position_config

    with open(output_file_path, 'w') as file:
        json.dump(data, file, indent=4)

    print(f"文件 {output_file_path} 已更新。")
ori = th.tensor([0.7875, -0.2173, 0.3633, 0.4479])
# w, x, y, z = -0.7875 ,-0.2173,0.3633,0.4479 # 例如
# X, Y, Z = quaternion_to_euler(w, x, y, z)
# print("X (roll):", X, "Y (pitch):", Y, "Z (yaw):", Z)
def euler2quat(euler: th.Tensor) -> th.Tensor:
    """
    Converts euler angles into quaternion form

    Args:
        euler (th.Tensor): (..., 3) (r,p,y) angles

    Returns:
        th.Tensor: (..., 4) (x,y,z,w) float quaternion angles

    Raises:
        AssertionError: [Invalid input shape]
    """
    assert euler.shape[-1] == 3, "Invalid input shape"
    #角度转弧度
    #euler = th.deg2rad(euler)
    
    # Unpack roll, pitch, yaw
    roll, pitch, yaw = euler.unbind(-1)

    # Compute sines and cosines of half angles
    cy = th.cos(yaw * 0.5)
    sy = th.sin(yaw * 0.5)
    cr = th.cos(roll * 0.5)
    sr = th.sin(roll * 0.5)
    cp = th.cos(pitch * 0.5)
    sp = th.sin(pitch * 0.5)

    # Compute quaternion components
    qw = cy * cr * cp + sy * sr * sp
    qx = cy * sr * cp - sy * cr * sp
    qy = cy * cr * sp + sy * sr * cp
    qz = sy * cr * cp - cy * sr * sp

    # Stack and return
    return th.stack([qx, qy, qz, qw], dim=-1)
def copysign(a, b):
    # type: (float, th.Tensor) -> th.Tensor
    a = th.tensor(a, device=b.device, dtype=th.float).repeat(b.shape[0])
    return th.abs(a) * th.sign(b)
def quat2euler(q):

    single_dim = q.dim() == 1

    if single_dim:
        q = q.unsqueeze(0)

    qx, qy, qz, qw = 0, 1, 2, 3
    # roll (x-axis rotation)
    sinr_cosp = 2.0 * (q[:, qw] * q[:, qx] + q[:, qy] * q[:, qz])
    cosr_cosp = q[:, qw] * q[:, qw] - q[:, qx] * q[:, qx] - q[:, qy] * q[:, qy] + q[:, qz] * q[:, qz]
    roll = th.atan2(sinr_cosp, cosr_cosp)
    # pitch (y-axis rotation)
    sinp = 2.0 * (q[:, qw] * q[:, qy] - q[:, qz] * q[:, qx])
    pitch = th.where(th.abs(sinp) >= 1, copysign(math.pi / 2.0, sinp), th.asin(sinp))
    # yaw (z-axis rotation)
    siny_cosp = 2.0 * (q[:, qw] * q[:, qz] + q[:, qx] * q[:, qy])
    cosy_cosp = q[:, qw] * q[:, qw] + q[:, qx] * q[:, qx] - q[:, qy] * q[:, qy] - q[:, qz] * q[:, qz]
    yaw = th.atan2(siny_cosp, cosy_cosp)

    euler = th.stack([roll, pitch, yaw], dim=-1) % (2 * math.pi)
    euler[euler > math.pi] -= 2 * math.pi

    if single_dim:
        euler = euler.squeeze(0)
    #弧度转角度
    #euler = th.rad2deg(euler)
    return euler

def change_ori_by_euler(object_config):
    object_config_euler = {}

    # 遍历 object_config 并修改 ori
    for object_name, object_data in object_config.items():
        # 深拷贝 object_data 到新字典
        object_data_euler = object_data.copy()
        
        # 提取 ori 并转为 Tensor
        ori_quat = th.tensor(object_data["root_link"]["ori"])
        
        # 使用已有 quat2euler 函数将四元数转换为欧拉角
        ori_euler = quat2euler(ori_quat)
        
        # 将欧拉角覆盖到 ori
        object_data_euler["root_link"]["ori"] = ori_euler.tolist()
        
        # 添加到新字典中
        object_config_euler[object_name] = object_data_euler
    return object_config_euler
def change_ori_by_quat(object_config_euler):
    object_config = {}

    # 遍历 object_config_euler 并修改 ori
    for object_name, object_data_euler in object_config_euler.items():
        # 深拷贝 object_data_euler 到新字典
        object_data = object_data_euler.copy()

        # 提取 ori 并转为 Tensor
        ori_euler = th.tensor(object_data["root_link"]["ori"])

        # 使用已有 euler2quat 函数将欧拉角转换回四元数
        ori_quat = euler2quat(ori_euler)

        # 将四元数覆盖到 ori
        object_data["root_link"]["ori"] = ori_quat.tolist()

        # 添加到新字典中
        object_config[object_name] = object_data

    return object_config

