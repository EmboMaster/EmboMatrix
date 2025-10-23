import omnigibson as og
from omnigibson.macros import gm
from omnigibson.utils.asset_utils import get_available_g_scenes, get_available_og_scenes
from omnigibson.utils.ui_utils import choose_from_options

import json
import os
import matplotlib.pyplot as plt
import matplotlib
import collections
import math
import torch as th
import numpy as np
from scipy.spatial.transform import Rotation as R
from PIL import Image, ImageDraw
import numpy as np
import sys
from collections import Counter
from omnigibson.utils.vision_utils import colorize_bboxes_3d

# calculate the orientation
def quaternion_from_axis_angle(axis, angle):
    """
    根据旋转轴和角度生成四元数 (x, y, z, w)。
    """
    axis = axis / np.linalg.norm(axis)
    sin_half_angle = np.sin(angle / 2)
    cos_half_angle = np.cos(angle / 2)
    return (
        sin_half_angle * axis[0],
        sin_half_angle * axis[1],
        sin_half_angle * axis[2],
        cos_half_angle
    )

def calculate_rotation_quaternion(v_target):
    """
    计算满足要求的旋转四元数 (x, y, z, w)。
    """
    # 初始向量
    v_z = np.array([0, 0, -1])
    v_x = np.array([1, 0, 0])
    
    # 目标向量归一化
    v_target = v_target / np.linalg.norm(v_target)
    
    # 第一部分：旋转 v_z 到 v_target
    axis1 = np.cross(v_z, v_target)
    if np.linalg.norm(axis1) < 1e-6:
        # 如果 v_z 和 v_target 平行，则无需旋转
        q1 = (0, 0, 0, 1) if np.dot(v_z, v_target) > 0 else (1, 0, 0, 0)
    else:
        axis1 = axis1 / np.linalg.norm(axis1)
        angle1 = np.arccos(np.dot(v_z, v_target))
        q1 = quaternion_from_axis_angle(axis1, angle1)
    
    # 旋转后的 v_x
    v_x_rotated = rotate_vector_by_quaternion(v_x, q1)
    
    # 第二部分：调整 v_x 的方向，使其与目标平面垂直
    z_axis = np.array([0, 0, 1])
    normal_to_plane = np.cross(v_target, z_axis)  # 平面法向量
    normal_to_plane = normal_to_plane / np.linalg.norm(normal_to_plane)
    
    # 计算旋转角度和轴
    angle2 = np.arccos(np.dot(v_x_rotated, normal_to_plane))
    if np.dot(np.cross(v_x_rotated, normal_to_plane), v_target) < 0:
        angle2 = -angle2  # 确保旋转方向正确
    q2 = quaternion_from_axis_angle(v_target, angle2)
    
    # 最终四元数
    q_final = quaternion_multiply(q2, q1)
    return q_final

def rotate_vector_by_quaternion(v, q):
    """
    使用四元数旋转向量 v。
    """
    q_v = np.array([v[0], v[1], v[2], 0])
    q_conj = np.array([-q[0], -q[1], -q[2], q[3]])
    q_result = quaternion_multiply(quaternion_multiply(q, q_v), q_conj)
    return q_result[:3]

def quaternion_multiply(q1, q2):
    """
    计算两个四元数的乘积。
    """
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2
    return (
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    )

def camera_for_scene_room(scene_model, room_name, type, caption, uninclude_list, env, image_height=2160, image_width=3840, focal_length=14, if_all_room = False):
    """
    type: "top_view", "front_view";
    caption: "rgb", "bbox_2d_tight";
    uninclude_list: ["walls", "floors", "window", "door"] etc;

    """
    json_name = "omnigibson/data/og_dataset/scenes/" + scene_model + "/json/" + scene_model + "_best.json"
    json_origin_name = "omnigibson/data/og_dataset/scenes/" + scene_model + "/json/" + scene_model + "_best_origin.json"

    if not os.path.exists(json_origin_name):  
        with open(json_name, 'r') as file:
            data = json.load(file)
    else:
        with open(json_origin_name, 'r') as file:
            data = json.load(file)
    
    objects = data["state"]["object_registry"]
    room_info = data["objects_info"]["init_info"]
    floors_pos_list = []
    ceiling_list = []
    obj_list = []

    for key in objects.keys():
        if "ceilings" in key:
            pos = objects[key]["root_link"]["pos"]
            ceiling_list.append((key, pos))

    for key in objects.keys():
        if "floors" in key:
            pos = objects[key]["root_link"]["pos"]
            room = room_info[key]["args"]["in_rooms"]
            if room != "":
                floors_pos_list.append((pos, room[0], key))
    
    for key in objects.keys():
        if "root_link" not in objects[key]:
            continue
        pos = objects[key]["root_link"]["pos"]
        obj_list.append((key, pos))

    # Load the environment
    objs = [obj for obj in env.scene.objects if obj.category != 'light']
    obj_names = [obj.name for obj in objs]
    obj_bbox = [obj.native_bbox for obj in objs]
    bdict = dict(zip(obj_names, obj_bbox))
    obj_dict = {obj.name: obj for obj in objs}

    if not gm.HEADLESS:
        og.sim.enable_viewer_camera_teleoperation()
    
    # 12/2: using double height result in Out of GPU memory.
    og.sim.viewer_camera.image_height = image_height
    og.sim.viewer_camera.image_width = image_width
    for s in ["rgb", "bbox_2d_tight", "bbox_2d_loose", "bbox_3d", "seg_semantic", "seg_instance", "seg_instance_id"]:
        og.sim.viewer_camera.add_modality(s)
    og.sim.viewer_camera.focal_length = focal_length

    from omnigibson.utils.constants import semantic_class_id_to_name
    semantic_dict = semantic_class_id_to_name()

    unincluded_objs = ['walls', 'floors', 'window', 'door']
    unincluded_id = [k for k, v in semantic_dict.items() if v in unincluded_objs]

    import collections
    room_cnt = collections.defaultdict(int)
    og.log.info("Resetting environment")
    four_list = []

    for (name, pos) in ceiling_list:
        # ceiling is invisible
        obj = obj_dict[name]
        obj.visible = False
        z = pos[2]

        # other objects are invisible
        for (m, p) in obj_list:
            d = abs(z - p[2])
            if m not in obj_dict:
                continue
            if d <= 0.4:
                obj_dict[m].visible = False

    for tup in floors_pos_list:

        pos = tup[0]
        room = tup[1]
        model = tup[2]
        if not if_all_room and room != room_name:
            continue

        for i in range(10):
            action = env.action_space.sample()
            state, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                break
        
        bbox = bdict[model].numpy().tolist()
        area = bbox[0]*bbox[1]
        pos[2] = math.sqrt(area)*1.61
        height = 1.4
        ratio = 0.369
        room_list = [room] # center position of the room
        room_list.append((pos[0], pos[1], pos[2]))
        room_list.append((pos[0], pos[1], height))
        d_list = [(ratio, ratio), (0, ratio), (-1*ratio, ratio), (-1*ratio, 0), (-1*ratio, -1*ratio), (0, -1*ratio), (ratio, -1*ratio), (ratio, 0)]
        for (dx, dy) in d_list:
            room_list.append((pos[0] + dx*bbox[0], pos[1] + dy*bbox[1], height*1.4))
        four_list.append(room_list)

    all_room_imgs = {}
    all_room_obj_num = {}
    import collections
    from collections import Counter
    room_cnt = collections.defaultdict(int)

    for l in four_list:
        room_name = l[0]
        top_view_center = np.array(list(l[1]))
        center = np.array(list(l[2]))

        if type == 'top_view':
            og.sim.viewer_camera.set_position_orientation(top_view_center, [0,0,0,1])
            img_file, _ = get_camera(caption,uninclude_list, env, room_name, unincluded_id, semantic_dict)
            if not if_all_room:
                return [img_file], [], math.sqrt(area)
            else:
                room_cnt[room_name] += 1
                final_room_name = room_name + f"_{room_cnt[room_name]}"
                all_room_imgs[final_room_name] = [img_file]
                all_room_obj_num[final_room_name] = []
        else:
            outcome_list = []
            objs_list = []
            for pos in l[3:]:
                start_p = np.array(list(pos))
                target_v = center - start_p
                ori = calculate_rotation_quaternion(target_v)
                og.sim.viewer_camera.set_position_orientation(pos, ori)

                tmp_outcome, tmp_objs_list = get_camera(caption,uninclude_list, env, room_name, unincluded_id, semantic_dict)

                outcome_list.append(tmp_outcome)
                objs_list.append(tmp_objs_list)

            if not if_all_room:
                return outcome_list, objs_list
            else:
                room_cnt[room_name] += 1
                final_room_name = room_name + f"_{room_cnt[room_name]}"
                all_room_imgs[final_room_name] = outcome_list
                all_room_obj_num[final_room_name] = objs_list


def get_camera(caption, uninclude_list, env, room_name, unincluded_id, semantic_dict):

    image_height = og.sim.viewer_camera.image_height
    font_size = (50 / 2160) * image_height
    # print(font_size)

    for i in range(10):
        action = env.action_space.sample()
        state, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            break
    obs, info = og.sim.viewer_camera.get_obs()

    if caption == 'rgb':
        return obs['rgb'], []
    else:

        bbox_modality = "bbox_2d_tight"
        seg_modality = "seg_instance"
        bboxes_2d_data = obs[bbox_modality]
        seg_data = obs[seg_modality]
        seg_info = info[seg_modality]

        from omnigibson.utils.deprecated_utils import colorize_bboxes_with_labels, get_image
        from omnigibson.utils.vision_utils import colorize_bboxes_3d
        clear_bboxes_2d_data = [tmp for tmp in bboxes_2d_data if tmp[0] not in unincluded_id]
        
        final_bboxes_2d_data = check_objs_in_room(room_name, clear_bboxes_2d_data, semantic_dict, env)
        bboxes_2d_data_with_name = append_name_to_bbox(final_bboxes_2d_data, semantic_dict, seg_data, seg_info)
        final_bboxes_2d_data_with_name = []
        for tmp in bboxes_2d_data_with_name:
            obj_name_tmp = tmp[0]
            if env.scene.object_registry("name", obj_name_tmp).in_rooms[0] == room_name and obj_name_tmp not in uninclude_list:
                final_bboxes_2d_data_with_name.append(tmp)
        colorized_img = get_image(final_bboxes_2d_data_with_name, obs["rgb"].cpu().numpy(), font_size=font_size)
        return colorized_img, [tmp_obj[0] for tmp_obj in bboxes_2d_data_with_name]


def compute_bbox_3d(semantic_data):
    bbox_3d_list = {}
    bbox_3d_list["x_min"] = []
    bbox_3d_list["y_min"] = []
    bbox_3d_list["z_min"] = []
    bbox_3d_list["x_max"] = []
    bbox_3d_list["y_max"] = []
    bbox_3d_list["z_max"] = []
    bbox_3d_list["transform"] = []

    for item in semantic_data:
        semanticID = item[0]
        x_min, y_min, z_min = item[1], item[2], item[3]
        x_max, y_max, z_max = item[4], item[5], item[6]
        transform = item[7]
        occlusion_ratio = item[8]

        bbox_3d_list["x_min"].append(x_min)
        bbox_3d_list["y_min"].append(y_min)
        bbox_3d_list["z_min"].append(z_min)
        bbox_3d_list["x_max"].append(x_max)
        bbox_3d_list["y_max"].append(y_max)
        bbox_3d_list["z_max"].append(z_max)
        bbox_3d_list["transform"].append(transform)

    bbox_3d_list["x_min"] = np.array(bbox_3d_list["x_min"])
    bbox_3d_list["y_min"] = np.array(bbox_3d_list["y_min"])
    bbox_3d_list["z_min"] = np.array(bbox_3d_list["z_min"])
    bbox_3d_list["x_max"] = np.array(bbox_3d_list["x_max"])
    bbox_3d_list["y_max"] = np.array(bbox_3d_list["y_max"])
    bbox_3d_list["z_max"] = np.array(bbox_3d_list["z_max"])
    bbox_3d_list["transform"] = np.array(bbox_3d_list["transform"])
    return bbox_3d_list

def check_objs_in_room(room_name, bbox_2d_data, semantic_dict_id_class, env):

    already_class = []
    final_bbox2d_data = []

    for tmp in bbox_2d_data:
        semantic_id = tmp[0]
        tmp_class = semantic_dict_id_class.get(semantic_id)
        if tmp_class in already_class:
            final_bbox2d_data.append(tmp)
            continue
        tmp_set = env.scene.object_registry("category", tmp_class)
        if tmp_set == None:
            continue
        try:
            tmp_room = [obj.in_rooms[0] for obj in tmp_set if hasattr(obj, 'in_rooms') and obj.in_rooms!=None]
        except:
            import pdb; pdb.set_trace()
        if room_name in tmp_room:
            final_bbox2d_data.append(tmp)

    return final_bbox2d_data

def append_name_to_bbox(final_bbox2d_data, semantic_dict_id_class, seg_data,seg_info):
    final_bbox2d_data_with_name = []
    for tmp in final_bbox2d_data:
        semantic_id = tmp[0]
        tmp_class = semantic_dict_id_class.get(semantic_id)
        
        name = most_frequent_name_in_range_tensor(seg_data, seg_info, tmp_class, tmp[1], tmp[2], tmp[3], tmp[4])
        if name is not None:
            tmp = (name,) + tmp[1:]
            final_bbox2d_data_with_name.append(tmp)
    return final_bbox2d_data_with_name
  

def most_frequent_name_in_range_tensor(seg_data, seg_info, objectname, xmin, ymin, xmax, ymax):
    sub_tensor = seg_data[ymin:ymax+1, xmin:xmax+1]
    unique_vals, counts = th.unique(sub_tensor, return_counts=True)
    name_counts = Counter()
    for val, count in zip(unique_vals, counts):
        name = seg_info.get(val.item(), None)
        if name and isinstance(name, str) and name.startswith(objectname):
            name_counts[name] += count.item()
    if name_counts:
        return name_counts.most_common(1)[0][0]
    else:
        return None
    

    
    

if __name__ == "__main__":
    main()