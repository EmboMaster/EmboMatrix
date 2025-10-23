import matplotlib.pyplot as plt
import numpy as np
import torch as th
from tqdm import tqdm
import sys
import omnigibson as og
from omnigibson.macros import gm
from omnigibson.utils.camera_utils import camera_for_scene_room 
from PIL import Image, ImageDraw
from omnigibson.object_states import OnTop, NextTo, ContactBodies
from omnigibson.objects.dataset_object import DatasetObject
from omnigibson.utils.constants import PrimType
from omnigibson.action_primitives.starter_semantic_action_primitives import StarterSemanticActionPrimitives

from scipy.spatial.transform import Rotation as R
from shapely.geometry import Polygon
def quat2euler(quat):
    """Convert quaternion to Euler angles."""
    r = R.from_quat(quat)
    return r.as_euler('xyz', degrees=False)

def is_point_in_bbox(px, py, bx, by, bw, bh, angle):
    """Check if a point (px, py) is inside a rotated bounding box centered at (bx, by) with width bw, height bh, and rotation angle."""
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    dx, dy = px - bx, py - by
    x_rot = cos_a * dx + sin_a * dy
    y_rot = -sin_a * dx + cos_a * dy
    return abs(x_rot) <= bw / 2 and abs(y_rot) <= bh / 2


def find_unoccupied_grids(positions, extents, orientations, objects, base_obj_name, grid_size_number=50, self_object_name=None, visible_objects=[]):
    """Find unoccupied grid positions within the boundary of a specified object."""
    base_index = -1
    for idx in range(len(objects)):
        if base_obj_name == objects[idx]:
            base_index = idx
            break
    if base_index == -1:
        raise ValueError(f"Object {base_obj_name} not found in objects list.")
    
    base_size = extents[base_index]
    base_center = positions[base_index]
    
    # 设定 grid 的大小，使得在合理范围内
    grid_size = min(base_size) / grid_size_number  # 设定为对象最小边长的1/grid_size_number
    
    # 计算对象边界
    base_min_x = base_center[0] - base_size[0] / 2
    base_max_x = base_center[0] + base_size[0] / 2
    base_min_y = base_center[1] - base_size[1] / 2
    base_max_y = base_center[1] + base_size[1] / 2
    
    # 计算 grid 的边界
    x_min, y_min = base_min_x, base_min_y
    x_max, y_max = base_max_x, base_max_y
    
    # 计算网格的数量
    x_num = int(np.ceil((x_max - x_min) / grid_size))
    y_num = int(np.ceil((y_max - y_min) / grid_size))
    
    # 创建 occupancy grid
    occupancy_grid = np.zeros((x_num, y_num), dtype=bool)
    
    # 计算每个物体的旋转角度
    angles = np.array([quat2euler(q)[2] for q in orientations])
    
    # 遍历所有物体，占据网格
    for pos, ext, angle, obj in zip(positions, extents, angles, objects):
        if "window" in obj or "ceiling" in obj or "electric_switch" in obj or obj == self_object_name or base_obj_name == obj:  # 忽略窗口和自身
            continue

        if "floors_" not in base_obj_name and obj not in visible_objects:
            continue

        for i in range(x_num):
            for j in range(y_num):
                x_coord = x_min + i * grid_size + grid_size / 2
                y_coord = y_min + j * grid_size + grid_size / 2
                if is_point_in_bbox(x_coord, y_coord, pos[0], pos[1], ext[0], ext[1], angle):
                    occupancy_grid[i, j] = True
    
    # 获取未占据的网格
    unoccupied_grids = []
    for i in range(x_num):
        for j in range(y_num):
            if not occupancy_grid[i, j]:
                x_coord = x_min + i * grid_size + grid_size / 2
                y_coord = y_min + j * grid_size + grid_size / 2
                unoccupied_grids.append((x_coord, y_coord))
    
    return unoccupied_grids, grid_size

def _sample_pose_near_object_directly(
    positions, extents, orientations, object_names, obj_pos, obj_extent, obj_ori,
    robot_extent, unoccupied_grids, distance_lo=0.1, distance_hi=0.7, base_obj_name="floor"
):
    """
    Sample a valid robot pose near a given object while satisfying all constraints.
    """
    from shapely.geometry import Point

    # 获取 base_obj 的边界
    base_index = object_names.index(base_obj_name)
    base_bbox = bbox_to_polygon(
        positions[base_index], extents[base_index], quat2euler(orientations[base_index])[2]
    )

    # 计算 obj_bbox 以获取边界
    obj_bbox = bbox_to_polygon(obj_pos, obj_extent, obj_ori)
    obj_poly = obj_bbox.exterior

    max_length = 1/2 * max(obj_extent) + 1/2 * max(robot_extent)

    valid_positions = []

    # 先过滤掉不可能的 `unoccupied_grids`
    candidate_grids = [
        (gx, gy) for (gx, gy) in unoccupied_grids if distance_lo + max_length <= Point(gx, gy).distance(Point(obj_pos)) <= distance_hi + max_length
    ]
    for (gx, gy) in tqdm(candidate_grids):
        for yaw in [0, np.pi / 2, np.pi, 3 * np.pi / 2]:
            robot_bbox = bbox_to_polygon((gx, gy), robot_extent, yaw)

            # 1. 机器人 bbox 必须完全在 base_obj bbox 内
            if not base_bbox.contains(robot_bbox):
                continue

            # 2. 机器人 bbox 不能与房间中的任何物体相交
            if any(
                robot_bbox.intersects(bbox_to_polygon(pos, extent, quat2euler(ori)[2]))
                for pos, extent, ori, obj in zip(positions, extents, orientations, object_names)
                if obj != base_obj_name and "window" not in obj and "ceilings" not in obj and "floors" not in obj and "electric_switch" not in obj
            ):
                continue

            # 3. 机器人 bbox 与物体边缘的距离要在指定范围内
            if not (distance_lo <= Point(gx, gy).distance(obj_poly) <= distance_hi):
                continue

            valid_positions.append((gx, gy, yaw))

            if len(valid_positions) >= 1:
                return valid_positions  # 直接返回，避免不必要的计算

    return None

def find_best_placement(scores, final_grids, positions, extents, orientations, object_names,
                         obj_extent, robot_extent, unoccupied_grids,
                         distance_lo=0.1, distance_hi=0.7, base_obj_name="floors"):
    """
    Sorts final_grids based on scores and finds the first valid robot position using _sample_pose_near_object.
    """
    # Sort grids by score from high to low
    sorted_grids = [grid for _, grid in sorted(zip(scores, final_grids), reverse=True)]
    final_grid, final_score, final_robot = [], [], []

    iteration_number = 0
    
    for grid, score in zip(sorted_grids, sorted(scores, reverse=True)):

        obj_pos, obj_ori = grid

        robot_position = _sample_pose_near_object_directly(
            positions, extents, orientations, object_names, obj_pos, obj_extent, obj_ori,
            robot_extent, unoccupied_grids, distance_lo, distance_hi, base_obj_name
        )

        print("robot_position", robot_position)
        iteration_number += 1
        
        if robot_position:
            final_grid.append(grid)
            final_robot.append(robot_position)
            final_score.append(score)
            
            if len(final_grid) >= 8:
                return final_grid, final_robot, final_score

        if iteration_number > 200:
            if len(final_grid) >= 1:
                return final_grid, final_robot, final_score
            else:
                print("Can not find robot position!")
                sys.exit(1)
    
    if len(final_grid) > 0:
        return final_grid, final_robot, final_score
    return None, None, None  # Return None if no valid placement is found


def bbox_to_polygon(center, extent, angle):
    """Convert a bounding box with center, extent, and angle to a polygon."""
    w, h = extent[0] / 2, extent[1] / 2
    corners = np.array([[-w, -h], [w, -h], [w, h], [-w, h]])
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    rotation_matrix = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
    rotated_corners = np.dot(corners, rotation_matrix.T) + center
    return Polygon(rotated_corners)

def find_valid_placement_grids(unoccupied_grids, new_extent, positions, extents, orientations, objects, base_obj_name, self_object_name, grid_size, visible_objects=[]):
    """Find all valid grid positions where a new object can be placed without overlapping existing objects and within base object boundaries."""
    valid_grids = []
    angles = [0, np.pi/2, np.pi, -np.pi/2]  # 0, 90, 180, 270 degrees in radians
    if "floors_" in base_obj_name:
        existing_polygons = [
            bbox_to_polygon(pos, ext, quat2euler(q)[2])
            for pos, ext, q, obj in zip(positions, extents, orientations, objects)
            if not ("window" in obj or "ceilings" in obj or obj == base_obj_name or obj == self_object_name or "electric_switch" in obj)
        ]
    else:
        existing_polygons = [
            bbox_to_polygon(pos, ext, quat2euler(q)[2])
            for pos, ext, q, obj in zip(positions, extents, orientations, objects)
            if not ("window" in obj or "ceilings" in obj or obj == base_obj_name or obj == self_object_name or "electric_switch" in obj) and obj in visible_objects
        ]
    
    # 获取 base_obj 的 AABB 作为新的边界
    base_index = -1
    for idx in range(len(objects)):
        if base_obj_name == objects[idx]:
            base_index = idx
            break
    if base_index == -1:
        raise ValueError(f"Object {base_obj_name} not found in objects list.")
    
    base_center = positions[base_index]
    base_size = extents[base_index]
    room_min_x = base_center[0] - base_size[0] / 2
    room_max_x = base_center[0] + base_size[0] / 2
    room_min_y = base_center[1] - base_size[1] / 2
    room_max_y = base_center[1] + base_size[1] / 2

    new_extent = np.array([new_extent[0]+2*grid_size, new_extent[1]+2*grid_size])
    
    for grid_x, grid_y in unoccupied_grids:
        for angle in angles:
            new_bbox = bbox_to_polygon((grid_x, grid_y), new_extent, angle)
            bbox_min_x, bbox_min_y, bbox_max_x, bbox_max_y = new_bbox.bounds
            
            # 确保新的物体不超出 base_obj 的边界
            if (bbox_min_x < room_min_x or bbox_max_x > room_max_x or
                bbox_min_y < room_min_y or bbox_max_y > room_max_y):
                continue
            
            collision = any(new_bbox.intersects(existing) for existing in existing_polygons)
            if not collision:
                valid_grids.append(((grid_x, grid_y), angle))
    
    return valid_grids

def evaluate_holodeck_rule_based(cons_list, valid_grids, new_extent, positions, extents, orientations, objects, base_obj_name, position_tolerance=0.1, angle_tolerance=np.deg2rad(5)):
    """Evaluate each valid grid position against constraints and return scores."""
    scores = []
    unsatisfied_cons = []
    
    # 获取房间边界
    base_index = -1
    for idx in range(len(objects)):
        if base_obj_name == objects[idx]:
            base_index = idx
            break
    if base_index == -1:
        raise ValueError(f"Object {base_obj_name} not found in objects list.")
    base_center = positions[base_index]
    base_size = extents[base_index]
    room_min_x = base_center[0] - base_size[0] / 2
    room_max_x = base_center[0] + base_size[0] / 2
    room_min_y = base_center[1] - base_size[1] / 2
    room_max_y = base_center[1] + base_size[1] / 2
    
    for idx, ((grid_x, grid_y), angle) in enumerate(valid_grids):
        score = 0
        unsatisfied = []
        new_bbox = bbox_to_polygon((grid_x, grid_y), new_extent, angle)
        bbox_min_x, bbox_min_y, bbox_max_x, bbox_max_y = new_bbox.bounds
        
        for cons in cons_list:
            if cons == "edge":
                if (abs(bbox_min_x - room_min_x) < position_tolerance or abs(bbox_max_x - room_max_x) < position_tolerance or
                    abs(bbox_min_y - room_min_y) < position_tolerance or abs(bbox_max_y - room_max_y) < position_tolerance):
                    score += 2
                else:
                    unsatisfied.append(cons)
            else:
                cons_type, cons_obj = cons.split(", ")
                obj_idx = objects.index(cons_obj)
                obj_pos = positions[obj_idx]
                obj_angle = quat2euler(orientations[obj_idx])[2] # [-pi, pi]
                
                dx, dy = grid_x - obj_pos[0], grid_y - obj_pos[1]
                alignment_angle = np.arctan2(dy, dx)  # [-pi, pi]
                relative_angle = abs((alignment_angle - obj_angle + np.pi) % (2 * np.pi) - np.pi)  # Normalize to [-pi, pi]

                if cons_type == "near" and 0 < np.linalg.norm([dx, dy]) < 1:
                    score += 0.5
                elif cons_type == "far" and np.linalg.norm([dx, dy]) > 1:
                    score += 0.5
                elif cons_type == "center aligned":
                    obj_orientation = obj_angle
                    if any(abs(alignment_angle - a) < angle_tolerance for a in [obj_orientation, (obj_orientation + np.pi/2 + np.pi) % (2*np.pi) - np.pi, (obj_orientation + np.pi + np.pi) % (2*np.pi) - np.pi, (obj_orientation + 3*np.pi/2 + np.pi) % (2*np.pi) - np.pi]):
                        score += 1.5
                    else:
                        unsatisfied.append(cons)
                elif cons_type == "face to":
                    target_angle =  - alignment_angle
                    if abs(angle - target_angle) < angle_tolerance:
                        score += 1.5
                    else:
                        unsatisfied.append(cons)
                elif cons_type == "in front of":
                    if relative_angle < np.deg2rad(45):
                        score += 0.5
                    else:
                        unsatisfied.append(cons)
                elif cons_type == "side of":
                    if np.deg2rad(45) <= relative_angle < np.deg2rad(135):
                        score += 0.5
                    else:
                        unsatisfied.append(cons)
        
        scores.append(score)
        unsatisfied_cons.append(unsatisfied)
    
    return scores, unsatisfied_cons

def avail_grids_for_new_added_object(positions, extents, orientations, objects, new_extent, cons_list, grid_size_number=50, new_object_name = None, base_obj_name=None, robot_extent = [],floor_object_name="floors", visible_objects = []):

    unoccupied_grids, grid_size = find_unoccupied_grids(positions, extents, orientations, objects, grid_size_number=grid_size_number, self_object_name=new_object_name, base_obj_name=base_obj_name, visible_objects=visible_objects)

    if "floors_" not in base_obj_name:
        unoccupied_grids_floor, grid_size = find_unoccupied_grids(positions, extents, orientations, objects, grid_size_number=grid_size_number, self_object_name=new_object_name, base_obj_name=floor_object_name)
    else:
        unoccupied_grids_floor = unoccupied_grids

    # print(unoccupied_grids)

    valid_grids = find_valid_placement_grids(unoccupied_grids, new_extent, positions, extents, orientations, objects, self_object_name=new_object_name, base_obj_name=base_obj_name, grid_size=grid_size,visible_objects=visible_objects)

    # import pdb; pdb.set_trace()

    if valid_grids == []:
        return [], 0, [], [], [], unoccupied_grids

    scores, unsatisfied_cons = evaluate_holodeck_rule_based(cons_list, valid_grids, new_extent, positions, extents, orientations, objects, position_tolerance=0.1, angle_tolerance=np.deg2rad(5), base_obj_name=base_obj_name)

    # max_score = max(scores)
    # final_grids = [tmp_grid for tmp_grid, tmp_score in zip(valid_grids, scores) if tmp_score == max_score]
    # unsatisfied_cons = [tmp_cons for tmp_cons, tmp_score in zip(unsatisfied_cons, scores) if tmp_score == max_score]
    final_grids, final_robots, max_score = find_best_placement(scores, valid_grids, positions, extents, orientations, objects, new_extent, robot_extent, unoccupied_grids_floor, distance_lo=0.1, distance_hi= 1.5, base_obj_name=floor_object_name)

    return final_grids, max_score, scores, unsatisfied_cons, final_robots, unoccupied_grids


def get_image_now(scene_model, room_name, env, full_path, top_view_obj_name = None, uninclude_list = ['floor'], include_list = None):

    if include_list is not None:
        for obj in env.scene.objects:
            if obj.name not in include_list:
                uninclude_list.append(obj.name)

    bbox = camera_for_scene_room(scene_model, room_name, "top_view", "bbox_2d_tight", uninclude_list=uninclude_list, env=env, image_height=1080, image_width=1440, focal_length=14, top_view_obj_name=top_view_obj_name)

    if bbox is None:
        print("can not get image!")
        return None
    else:
        bbox, obj_name_list = bbox
    # print(bbox)
    # import pdb; pdb.set_trace()
    image = Image.fromarray(bbox[0].astype('uint8'))

    # 保存图像
    image.save(full_path)
    return obj_name_list[0]

def OnTopHolodeck(leaf_obj, base_obj=None, env=None, scene_model=None, room_name=None, use_official_api = False, debug_path_image = None,new_added_objects_list=[], just_for_robot_pose = False):

    from src.room_modifier.utils import quat2euler, euler2quat

    robot_extent = [0.5739, 0.5802]

    bbox_positions, bbox_orientations, bbox_extents, obj_names = get_room_bbox_information(env, room_name)
    new_extent = get_relative_aabb_information(leaf_obj)[2]
    room_objects_list = [obj.name for obj in env.scene.objects if "in_rooms" in dir(obj) and obj.in_rooms[0] == room_name]
    floor_object_name = [tmp for tmp in room_objects_list if "floors_" in tmp][0]
    
    if just_for_robot_pose:
        base_obj = env.scene.object_registry("name",floor_object_name)

    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if debug_path_image is not None:
        img_path = f"{debug_path_image}/{scene_model}_{room_name}_{base_obj.name}_{timestamp}.png"
    else:
        img_path = f"src/room_creater/ontop_check_images/{scene_model}_{room_name}_{base_obj.name}_{timestamp}.png"
    visible_objects = draw_top_view_only_for_bbox(bbox_positions,bbox_extents,bbox_orientations, obj_names, base_obj.name, img_path)
    visible_objects = [tmp for tmp in visible_objects if tmp != base_obj.name]
    visible_objects = [tmp for tmp in visible_objects if tmp in obj_names]

    if "floors_" not in base_obj.name:
        ontop_objects = []
        for obj in env.scene.objects:
            try:
                flag = obj.states[OnTop].get_value(base_obj)
                if flag:
                    ontop_objects.append(obj.name)
            except:
                continue
        visible_objects = [tmp for tmp in visible_objects if tmp in ontop_objects]
    

    if debug_path_image is not None:
        img_path = f"{debug_path_image}/{scene_model}_{room_name}_{base_obj.name}_{timestamp}_origin.png"
    else:
        img_path = f"src/room_creater/ontop_check_images/{scene_model}_{room_name}_{base_obj.name}_{timestamp}_origin.png"
    obj_name_list = get_image_now(scene_model,room_name,env,img_path)

    # step1. get the constraints
    if "floor" in base_obj.name:
        constraints_global = "1) edge: at the edge of the floor, close to the wall. 2) middle: not close to the edge of the room. 3) random: randomly placed on the floor."
    else:
        constraints_global = f"1) middle: not close to the edge of the {base_obj.name}. 2) random: randomly placed on the {base_obj.name}."

    error_hint = ""
    
    prompt = f"""
    You are an experienced room designer and now you need to help to put {leaf_obj.name} on top of {base_obj.name}. Currently, there are {len(visible_objects)} objects on top of {base_obj.name}, which are: {visible_objects}. Please help me arrange {leaf_obj.name} in the room by assigning constraints. The top-view image of the {base_obj.name} is attached. Here are the constraints and their definitions:

    1. global constraint: {constraints_global}. 
    2. distance constraint: 
        1) near, object: near to the other object, but with some distance, 0 < distance < 2.    
        2) far, object: far away from the other object, distance > 2. 
    3. position constraint: 
        1) in front of, object: in front of another object. 
        2) side of, object: on the side (left or right) of another object. 
    4. alignment constraint: 
        1) center aligned, object: align the center of the object with the center of another object. 
    5. Rotation constraint: 
        1) face to, object: face to the center of another object.

    You must have one global constraint and you can select various numbers of constraints and any combinations of them and the output format must be like below: 
        object | global constraint | constraint 1 | constraint 2 | ... 
    If you choose 'random' as the global constraint, you can ignore other constraints.
    For example: 
        bottle-0 | random
        sofa-0 | edge 
        coffee table-0 | middle | in front of, sofa-0 | center aligned, sofa-0
        tv stand-0 | edge | far, coffee table-0 | side of, coffee table-0 | face to, coffee table-0

    Here are some guidelines for you to follow:
    1. If there are no objects on top of {base_obj.name}, you must choose 'random' as the global constraint. Currently, there are {len(visible_objects)} objects on top of {base_obj.name}.
    2. 'edge' can only be chosen for the case when you put objects on floor. If you put objects on other objects, you can not choose 'edge'.
    3. each type of constraints can only be used not more than once. For example, you can not make 'near, object1' and 'near, object2' in the same output. 
    4. you can only use objects in the constraints that are in the below list: {visible_objects}. 
    5. you can never ever make {base_obj.name} in the constraints because it is the base object that you put {leaf_obj.name} on top of.

    {error_hint}
    
    Please first use natural language to explain your high-level design strategy, and then follow the desired format *strictly* (do not add any additional text at the beginning or end) to provide the constraints for each object.
    """

    from src.llm_selection import get_gpt_response_by_request, convert_images_to_base64
    messages = [{
            "role": "user",
            "content": [{
                "type": "text",
                "text": prompt
            }]
        }]

    image_path = [img_path]
    if obj_name_list is None:
        image_path = []
    image_paths = convert_images_to_base64(image_path, target_width=800)

    if not just_for_robot_pose:
        cons_check_flag = True
    else:
        cons_check_flag = False
        global_cons = "random"
        local_cons = []

    while cons_check_flag:
        result = get_gpt_response_by_request(messages=messages, max_try=3, use_official_api=use_official_api, image_paths=image_paths)

        print(result)

        try:
            global_cons, local_cons = extract_constraints(result, leaf_obj.name)
        except Exception as e:
            error_hint = f"Attention! There is an error when extracting constraints: {e} Please check the format and try again."
            continue
        
        real_local_cons = {}
        for con in local_cons:
            cons_type, cons_obj = con.split(", ")
            if cons_type not in ["near", "far", "in front of", "side of", "center aligned", "face to"]:
                error_hint = f"Attention! The constraint type '{cons_type}' is not supported."
                cons_check_flag = True
                continue
            if cons_obj not in visible_objects:
                error_hint = f"Attention! The object '{cons_obj}' is not on top of {base_obj.name}. You can not add it to the final output."
                cons_check_flag = True
                continue
            real_local_cons[cons_type] = cons_obj
        cons_check_flag = False

    falg, sample_falg = True, False
    if global_cons == "random" or (global_cons == "middle" and len(local_cons) == 0):
        
        sample_robots = None
        for i in range(10):
            if not just_for_robot_pose:
                falg = leaf_obj.states[OnTop].set_value(base_obj, True)
            if falg:

                new_pos, new_ori, new_extent = get_relative_aabb_information(leaf_obj)
                bbox_positions, bbox_orientations, bbox_extents, obj_names = get_room_bbox_information(env, room_name)
                unoccupied_grids, grid_size = find_unoccupied_grids(bbox_positions, bbox_extents, bbox_orientations, obj_names, self_object_name=leaf_obj.name, base_obj_name=floor_object_name)

                sample_robots = _sample_pose_near_object_directly(positions=bbox_positions, extents=bbox_extents, orientations=bbox_orientations, object_names=obj_names, obj_pos=new_pos, obj_extent=new_extent, obj_ori=quat2euler(new_ori)[2], robot_extent=robot_extent, unoccupied_grids=unoccupied_grids, distance_lo=0.1, distance_hi=1.5, base_obj_name=floor_object_name)

                print("random sample_robots", sample_robots)

                if sample_robots != None:
                    sample_falg = True
                    break
        
        if falg:
            return falg, ["nonsense",0,[]], sample_robots
    
    
    if "edge" in global_cons:
        cons_list = ["edge"]
    else:
        cons_list = []
    cons_list.extend(local_cons)


    valid_grids, max_score, scores, unsatisfied_cons_list, final_robots, unoccupied_grids = avail_grids_for_new_added_object(bbox_positions, bbox_extents, bbox_orientations, obj_names, new_extent, cons_list, grid_size_number=50, new_object_name=leaf_obj.name,base_obj_name=base_obj.name, robot_extent=robot_extent,floor_object_name = floor_object_name, visible_objects=visible_objects)
    print("max_score: ",max_score)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if debug_path_image is not None:
        img_path = f"{debug_path_image}/{scene_model}_{room_name}_{base_obj.name}_{timestamp}_after.png"
    else:
        img_path = f"src/room_creater/ontop_check_images/{scene_model}_{room_name}_{base_obj.name}_{timestamp}_after.png"
    _ = draw_top_view_only_for_bbox(bbox_positions, bbox_extents, bbox_orientations, obj_names, base_obj.name, img_path, unoccupied_grids=unoccupied_grids, valid_grids=valid_grids, new_extent=new_extent, robot_extent = robot_extent, final_robots = final_robots)

    if valid_grids == [] or valid_grids is None:
        return False, [cons_list, max_score, unsatisfied_cons_list], []

    if falg:
        if not leaf_obj.states[OnTop].get_value(base_obj):
            for i in range(10):
                falg = leaf_obj.states[OnTop].set_value(base_obj, True)
                if falg:
                    break
        for obj in env.scene.objects:
            obj.set_angular_velocity(th.tensor([0., 0., 0.]))
            obj.set_linear_velocity(th.tensor([0., 0., 0.]))
        leaf_obj_z_axis = leaf_obj.get_base_aligned_bbox()[0][2]
    else:
        leaf_obj_z_axis = base_obj.get_base_aligned_bbox()[0][2] + base_obj.get_base_aligned_bbox()[2][2]/2 + leaf_obj.get_base_aligned_bbox()[2][2]/2
    
    contact_objects = leaf_obj.states[ContactBodies].get_value()
    contact_objects = [tmp_obj.name.split(":")[0] if ":" in tmp_obj.name else tmp_obj.name for tmp_obj in contact_objects]
    contact_objects = [env.scene.object_registry("name",tmp_obj) for tmp_obj in contact_objects]

    state = og.sim.dump_state(serialized=False)
    collide_iteration_dict = {}
    flag, robot_idx = False, -1
    for i in range(min(10, len(valid_grids))):
        print("unsatisfied_cons_list:", unsatisfied_cons_list[i])
        new_pos = th.tensor([valid_grids[i][0][0],valid_grids[i][0][1],leaf_obj_z_axis])
        new_ori = euler2quat(th.tensor([0,0,valid_grids[i][1]]))
        leaf_obj.set_bbox_center_position_orientation(new_pos, new_ori)
        leaf_obj.set_angular_velocity(th.tensor([0., 0., 0.]))
        leaf_obj.set_linear_velocity(th.tensor([0., 0., 0.]))
        for tmp_obj in contact_objects:
            tmp_obj.set_angular_velocity(th.tensor([0., 0., 0.]))
            tmp_obj.set_linear_velocity(th.tensor([0., 0., 0.]))
        if debug_path_image is not None:
            img_path = f"{debug_path_image}/{scene_model}_{room_name}_{base_obj.name}_{timestamp}_exact_position{i}.png"
        else:
            img_path = f"src/room_creater/ontop_check_images/{scene_model}_{room_name}_{base_obj.name}_{timestamp}_exact_position{i}.png"
        get_image_now(scene_model,room_name,env,img_path)
        collide_object_list = velocity_collide(env,[leaf_obj.name]+[tmp_obj.name for tmp_obj in contact_objects]+new_added_objects_list, norm_threshold=0.1, room_name=room_name)
        if collide_object_list!=[]:
            for tmp_obj in collide_object_list:
                if tmp_obj not in collide_iteration_dict:
                    collide_iteration_dict[tmp_obj] = 1
                else:
                    collide_iteration_dict[tmp_obj] += 1
            if max(collide_iteration_dict.values()) > 5:
                try:
                    if debug_path_image is not None:
                        img_path = f"{debug_path_image}/{scene_model}_{room_name}_{base_obj.name}_{timestamp}_collide_position{i}.png"
                    else:
                        img_path = f"src/room_creater/ontop_check_images/{scene_model}_{room_name}_{base_obj.name}_{timestamp}_collide_position{i}.png"
                    flag_collide = get_image_now(scene_model,room_name,env,img_path,include_list=collide_object_list+[leaf_obj.name])
                    if flag_collide is not None:
                        img_collide = convert_images_to_base64([img_path], target_width=800)
                        prompt_collide = f"""You are an excellent observer! You need to check if the object {leaf_obj.name} is colliding with these objects {collide_object_list}. Please check the image and return the answer in the format [[Collide]] or [[Not Collide]]. """
                        messages = [{
                            "role": "user",
                            "content": [{
                                "type": "text",
                                "text": prompt_collide
                            }]
                        }]
                        result = get_gpt_response_by_request(messages=messages, max_try=3, use_official_api=use_official_api, image_paths=img_collide)
                        if "[[Collide]]" in result or "[[collide]]" in result:
                            flag_final = False
                        elif "[[Not Collide]]" in result or "[[not collide]]" in result or "[[Not collide]]" in result or "[[not Collide]]" in result:
                            flag_final = True
                        else:
                            flag_final = False
                    else:
                        flag_final = False

                    if flag_final:
                        flag = True
                    else:
                        print(collide_object_list)
                        print("This position causes collide: ",new_pos,new_ori)
                        og.sim.load_state(state, serialized=False)
                except:
                    print(collide_object_list)
                    print("This position causes collide: ",new_pos,new_ori)
                    og.sim.load_state(state, serialized=False)
            else:
                print(collide_object_list)
                print("This position causes collide: ",new_pos,new_ori)
                og.sim.load_state(state, serialized=False)
        else:
            flag = True
            robot_idx = i
        if flag:
            break
    
    if flag:
        return True, [cons_list, max_score, unsatisfied_cons_list[i]], final_robots[robot_idx]
    else:
        return False, ["nonsense",0,[]], []
    

def get_relative_aabb_information(objA):

    if isinstance(objA, DatasetObject) and objA.prim_type == PrimType.RIGID:
        # Retrieve base CoM frame-aligned bounding box parallel to the XY plane
        parallel_bbox_center, parallel_bbox_orn, parallel_bbox_extents, _ = objA.get_base_aligned_bbox(
            xy_aligned=True
        )
    else:
        try:
            aabb_lower, aabb_upper = objA.states[AABB].get_value()
            parallel_bbox_center = (aabb_lower + aabb_upper) / 2.0
            parallel_bbox_orn = th.tensor([0.0, 0.0, 0.0, 1.0])
            parallel_bbox_extents = aabb_upper - aabb_lower
        except:
            print(f"{objA.name} has no AABB states!")
            return None, None, None
    
    return parallel_bbox_center.numpy()[:2], parallel_bbox_orn, parallel_bbox_extents.numpy()[:2]
def velocity_collide(env,target_name, norm_threshold=0.1, room_name = None):
    collide_object_list = []
    for obj in env.scene.objects:
        if obj.name in target_name:
            continue

        if obj.get_linear_velocity().norm() > norm_threshold or obj.get_angular_velocity().norm() > norm_threshold:
            if "in_rooms" in dir(obj) and obj.in_rooms[0] != room_name:
                obj.set_angular_velocity(th.tensor([0., 0., 0.]))
                obj.set_linear_velocity(th.tensor([0., 0., 0.]))
                continue
            collide_object_list.append(obj.name)
    return collide_object_list

def get_room_bbox_information(env, room_name):

    bbox_center, bbox_orn, bbox_extent, bbox_name = [], [], [], []
    for obj in env.scene.objects:
        if "in_rooms" in dir(obj) and room_name in obj.in_rooms:
            parallel_bbox_center, parallel_bbox_orn, parallel_bbox_extents = get_relative_aabb_information(obj)
            if parallel_bbox_center is not None:
                bbox_center.append(parallel_bbox_center)
                bbox_orn.append(parallel_bbox_orn)
                bbox_extent.append(parallel_bbox_extents)
                bbox_name.append(obj.name)
    return bbox_center, bbox_orn, bbox_extent, bbox_name

def wrap_text(text, max_chars):
    """Wrap text into multiple lines if it exceeds max_chars."""
    words = text.split()
    lines = []
    line = ""
    for word in words:
        if len(line) + len(word) <= max_chars:
            line += (" " if line else "") + word
        else:
            lines.append(line)
            line = word
    lines.append(line)
    return "\n".join(lines)

# def draw_top_view_only_for_bbox(positions, extents, orientations, objects, base_obj_name, save_path,unoccupied_grids=None, valid_grids=None, new_extent=None):
#     """Draw a top-down view of the room with a specific object as the reference boundary and return visible objects."""
#     visible_objects = []
    
#     # 获取参考物体的索引
#     base_index = objects.index(base_obj_name)
#     base_center = positions[base_index]
#     base_size = extents[base_index]
    
#     # 计算参考物体的边界
#     base_min_x = base_center[0] - base_size[0] / 2
#     base_max_x = base_center[0] + base_size[0] / 2
#     base_min_y = base_center[1] - base_size[1] / 2
#     base_max_y = base_center[1] + base_size[1] / 2
    
#     # 创建图像
#     fig, ax = plt.subplots(figsize=(10, 10))
#     ax.set_xlim([base_min_x - 0.3, base_max_x + 0.3])
#     ax.set_ylim([base_min_y - 0.3, base_max_y + 0.3])
#     ax.set_aspect('equal')
#     ax.axis('off')  # 移除坐标轴

#     ax.add_patch(plt.Rectangle(
#         (base_min_x, base_min_y), base_size[0], base_size[1],
#         fill=False, edgecolor='black', linewidth=2, alpha=0.3
#     ))
    
#     # 画所有范围内的物体
#     for i, (pos, extent, obj, quat) in enumerate(zip(positions, extents, objects, orientations)):
#         if "window" in obj or "ceilings" in obj or "floors" in obj:
#             continue
        
#         obj_bbox = bbox_to_polygon(pos, extent, quat2euler(quat)[2])
#         base_bbox = bbox_to_polygon(base_center, base_size, 0)
        
#         if not obj_bbox.intersects(base_bbox):
#             continue  # 忽略范围外的物体
        
#         visible_objects.append(obj)
        
#         x, y = pos
#         w, h = extent
#         e1, e2, e3 = quat2euler(quat)
#         angle = np.rad2deg(e3)
        
#         if abs(angle - 0) < 10 or abs(angle - 180) < 10 or abs(angle - 360) < 10:
#             real_xy = (x - w / 2, y - h / 2)
#             real_w, real_h = w, h
#             text_rotation = 0
#             font_size = max(8, min(16, real_w * 10))
#             max_chars = int(real_w / 0.15)
#         else:
#             real_xy = (x - h / 2, y - w / 2)
#             real_w, real_h = h, w
#             text_rotation = 90
#             font_size = max(8, min(16, real_h * 10))
#             max_chars = int(real_h / 0.15)
        
#         ax.add_patch(plt.Rectangle(
#             real_xy, real_w, real_h,
#             fill=True, edgecolor='blue', facecolor='lightblue', alpha=0.5
#         ))
#         ax.text(x, y, wrap_text(obj, max_chars), fontsize=font_size, ha='center', va='center', fontweight='bold', rotation=text_rotation)

#     if unoccupied_grids:
#         ux, uy = zip(*unoccupied_grids)
#         ax.scatter(ux, uy, color='g', marker='o', label='Unoccupied Grids')

#     if valid_grids and new_extent is not None:
#         for (vx, vy), angle in valid_grids:
#             new_bbox = bbox_to_polygon((vx, vy), new_extent, angle)
#             x, y = new_bbox.exterior.xy
#             ax.fill(x, y, edgecolor='red', facecolor='pink', alpha=0.5)
    
#     plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
#     return visible_objects

def draw_top_view_only_for_bbox_old(positions, extents, orientations, objects, base_obj_name, save_path, unoccupied_grids=None, valid_grids=None, new_extent=None):
    """Draw a top-down view of the room with a specific object as the reference boundary and return visible objects."""
    visible_objects = []
    
    # 获取参考物体的索引
    base_index = objects.index(base_obj_name)
    base_center = positions[base_index]
    base_size = extents[base_index]
    base_ori = quat2euler(orientations[base_index])[2]
    
    # 计算参考物体的边界
    base_bbox = bbox_to_polygon(base_center, base_size, base_ori)
    
    # 创建图像
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_xlim([base_bbox.bounds[0] - (base_bbox.bounds[2]-base_bbox.bounds[0])/12, base_bbox.bounds[2] + (base_bbox.bounds[2]-base_bbox.bounds[0])/12])
    ax.set_ylim([base_bbox.bounds[1] - (base_bbox.bounds[3]-base_bbox.bounds[1])/12, base_bbox.bounds[3] + (base_bbox.bounds[3]-base_bbox.bounds[1])/12])
    ax.set_aspect('equal')
    ax.axis('off')  # 移除坐标轴

    ax.add_patch(plt.Rectangle(
        (base_bbox.bounds[0], base_bbox.bounds[1]), base_size[0], base_size[1],
        fill=False, edgecolor='black', linewidth=2, alpha=0.3
    ))
    
    # 画所有与 base_obj 相交的物体
    for i, (pos, extent, obj, quat) in enumerate(zip(positions, extents, objects, orientations)):
        if "window" in obj or "ceilings" in obj or "floors" in obj or "electric_switch" in obj:
            continue
        
        obj_bbox = bbox_to_polygon(pos, extent, quat2euler(quat)[2])
        if not obj_bbox.intersects(base_bbox):
            continue  # 忽略不与 base_obj 相交的物体
        
        visible_objects.append(obj)
        
        x, y = pos
        w, h = extent
        angle = np.rad2deg(quat2euler(quat)[2])
        
        if abs(angle - 0) < 10 or abs(angle - 180) < 10 or abs(angle - 360) < 10:
            real_xy = (x - w / 2, y - h / 2)
            real_w, real_h = w, h
            text_rotation = 0
            font_size = max(8, min(16, real_w * 10))
            max_chars = int(real_w / 0.15)
        else:
            real_xy = (x - h / 2, y - w / 2)
            real_w, real_h = h, w
            text_rotation = 90
            font_size = max(8, min(16, real_h * 10))
            max_chars = int(real_h / 0.15)
        
        ax.add_patch(plt.Rectangle(
            real_xy, real_w, real_h,
            fill=True, edgecolor='blue', facecolor='lightblue', alpha=0.5
        ))
        ax.text(x, y, obj, fontsize=font_size, ha='center', va='center', fontweight='bold', rotation=text_rotation)

    if unoccupied_grids:
        ux, uy = zip(*unoccupied_grids)
        ax.scatter(ux, uy, color='g', marker='o', label='Unoccupied Grids')

    if valid_grids and new_extent is not None:
        for (vx, vy), angle in valid_grids:
            new_bbox = bbox_to_polygon((vx, vy), new_extent, angle)
            x, y = new_bbox.exterior.xy
            ax.fill(x, y, edgecolor='red', facecolor='pink', alpha=0.5)
    
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    return visible_objects

def draw_top_view_only_for_bbox(positions, extents, orientations, objects, base_obj_name, save_path, 
                                unoccupied_grids=None, valid_grids=None, new_extent=None, 
                                robot_extent=None, final_robots=None):
    """Draw a top-down view of the room with a specific object as the reference boundary and return visible objects."""
    visible_objects = []

    # 获取参考物体的索引
    base_index = objects.index(base_obj_name)
    base_center = positions[base_index]
    base_size = extents[base_index]
    base_ori = quat2euler(orientations[base_index])[2]

    # 计算参考物体的边界
    base_bbox = bbox_to_polygon(base_center, base_size, base_ori)

    # 创建图像
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_xlim([base_bbox.bounds[0] - (base_bbox.bounds[2] - base_bbox.bounds[0]) / 12,
                 base_bbox.bounds[2] + (base_bbox.bounds[2] - base_bbox.bounds[0]) / 12])
    ax.set_ylim([base_bbox.bounds[1] - (base_bbox.bounds[3] - base_bbox.bounds[1]) / 12,
                 base_bbox.bounds[3] + (base_bbox.bounds[3] - base_bbox.bounds[1]) / 12])
    ax.set_aspect('equal')
    ax.axis('off')  # 移除坐标轴

    # 绘制参考物体
    ax.add_patch(plt.Rectangle(
        (base_bbox.bounds[0], base_bbox.bounds[1]), base_size[0], base_size[1],
        fill=False, edgecolor='black', linewidth=2, alpha=0.3
    ))

    # 绘制所有与 base_obj 相交的物体
    for i, (pos, extent, obj, quat) in enumerate(zip(positions, extents, objects, orientations)):
        if "window" in obj or "ceilings" in obj or "floors" in obj or "electric_switch" in obj:
            continue
        
        obj_bbox = bbox_to_polygon(pos, extent, quat2euler(quat)[2])
        if not obj_bbox.intersects(base_bbox):
            continue  # 忽略不与 base_obj 相交的物体
        
        visible_objects.append(obj)
        
        x, y = pos
        w, h = extent
        angle = np.rad2deg(quat2euler(quat)[2])

        if abs(angle - 0) < 10 or abs(angle - 180) < 10 or abs(angle - 360) < 10:
            real_xy = (x - w / 2, y - h / 2)
            real_w, real_h = w, h
            text_rotation = 0
        else:
            real_xy = (x - h / 2, y - w / 2)
            real_w, real_h = h, w
            text_rotation = 90

        ax.add_patch(plt.Rectangle(
            real_xy, real_w, real_h,
            fill=True, edgecolor='blue', facecolor='lightblue', alpha=0.5
        ))
        ax.text(x, y, obj, fontsize=10, ha='center', va='center', fontweight='bold', rotation=text_rotation)

    # 绘制未占据的网格点
    if unoccupied_grids:
        ux, uy = zip(*unoccupied_grids)
        ax.scatter(ux, uy, color='g', marker='o', label='Unoccupied Grids')

    # 绘制 valid_grids 的新物体放置点
    if valid_grids and new_extent is not None:
        idx = 0
        for (vx, vy), angle in valid_grids:
            new_bbox = bbox_to_polygon((vx, vy), new_extent, angle)
            x, y = new_bbox.exterior.xy
            ax.fill(x, y, edgecolor='red', facecolor='pink', alpha=0.5, label=f"Obj {idx}")
            idx += 1

    # 绘制 final_robots 机器人位置
    if final_robots and robot_extent is not None:
        idx = 0
        for tmp_tuple in final_robots:
            for rx, ry, yaw in tmp_tuple:
                robot_bbox = bbox_to_polygon((rx, ry), robot_extent, yaw)
                x, y = robot_bbox.exterior.xy
                ax.fill(x, y, edgecolor='purple', facecolor='violet', alpha=0.6, label=f"Robot {idx}")
            idx += 1

    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    return visible_objects


def extract_constraints(text, obj_name):
    """
    从输入字符串中提取约束信息。
    
    输入：
      text: 包含描述和约束信息的字符串，
            其中最后部分（由 '---' 分隔）包含类似：
            "straight_chair_dmcixv_2 | middle | near, straight_chair_dmcixv_1 | side of, 
             straight_chair_dmcixv_1 | center aligned, straight_chair_dmcixv_1 | face to, straight_chair_dmcixv_1"
      obj_name: 需要提取约束的物体名称，如 "straight_chair_dmcixv_2"
    
    处理流程：
      1. 提取字符串中 '---' 之后的部分。
      2. 在这一部分中查找以 obj_name 开头的那一行（如果存在多行，则取匹配行）。
      3. 将该行按 " | " 分割成多个部分。
         - 第一个部分应为物体名称（验证一致）。
         - 第二个部分为 global_cons。
         - 后续部分为 local_cons 列表（可能为空）。
    
    返回：
      (global_cons, local_cons)
      其中 global_cons 为字符串，
            local_cons 为列表（每个元素为一个约束短语）。
    """
    # 取 '---' 之后的部分（如果没有 '---' 则取整个文本）
    if '---' in text:
        tail = text.split('---')[-1].strip()
    else:
        tail = text.strip()
    
    # 如果有多行，尝试找出以 obj_name 开头的行
    candidate_line = None
    for line in tail.splitlines():
        line = line.strip()
        if line.startswith(obj_name):
            candidate_line = line
            break
    # 如果没找到，则直接使用整个 tail 作为候选
    if candidate_line is None:
        candidate_line = tail
    
    # 按 " | " 分割
    parts = [part.strip() for part in candidate_line.split("|")]
    
    # 检查第一个部分是否与 obj_name 相符
    if parts[0] != obj_name:
        raise ValueError(f"提取的行不以指定物体名称 '{obj_name}' 开头: {parts[0]}")
    
    # 第二部分为 global_cons
    if len(parts) >= 2:
        global_cons = parts[1]
    else:
        global_cons = ""
    
    # 从第三部分开始为 local_cons 列表（如果存在）
    local_cons = parts[2:] if len(parts) > 2 else []

    real_local_cons=[]
    for con in local_cons:
        if con == "" or con == " ":
            continue
        real_local_cons.append(con)
    local_cons = real_local_cons
    
    return global_cons, local_cons