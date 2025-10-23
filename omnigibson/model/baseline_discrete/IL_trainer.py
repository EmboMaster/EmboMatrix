import warnings
import datetime
warnings.filterwarnings("ignore")
import os
import cv2
import h5py
import numpy as np
import torch
from omnigibson.model.baseline_discrete.baseline_model_IL import MultiModalRobotPolicy_Discrete
from omnigibson.action_primitives.starter_semantic_action_primitives import StarterSemanticActionPrimitives
from omnigibson.macros import create_module_macros
import time
import omnigibson.utils.transform_utils as T
m = create_module_macros(module_path=__file__)
import math
from scipy.spatial.transform import Rotation as R
from PIL import Image
import omnigibson as og
from omnigibson.objects.primitive_object import PrimitiveObject
import json

def save_video_with_opencv(image_list, output_path, fps=30):
    """
    使用 OpenCV 将一系列 PIL 图像保存为视频文件

    Args:
        image_list (list of PIL.Image): 图像列表
        output_path (str): 输出视频路径
        fps (int): 视频帧率
    """
    output_dir = os.path.dirname(output_path)
    if output_dir:  # 确保路径不为空（防止直接为文件名时无目录部分）
        os.makedirs(output_dir, exist_ok=True)
    # 获取图像宽度和高度
    if not image_list:
        raise ValueError("Image list is empty")
    width, height = image_list[0].size

    # 初始化视频编码器
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    for img in image_list:
        # 将 PIL.Image 转为 numpy 格式，并转换为 BGR (OpenCV 使用 BGR 顺序)
        frame = np.array(img.convert("RGB"))
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        video_writer.write(frame)

    video_writer.release()


def tensor_serializer(obj):
    if isinstance(obj, torch.Tensor):
        return obj.tolist()
    raise TypeError(f"无法序列化类型 {type(obj)}")

def save_hdf5(data, sample_num, file_path):   
    # 确保目录存在
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with h5py.File(file_path, "w") as f:
        for i in range(len(data)):
            # 创建外层 key，命名为 sample_num
            sample_group = f.create_group(str(i))
            
            for key, value in data[i].items():
                # 转换 torch.Tensor 为 NumPy 数组
                if isinstance(value, torch.Tensor):
                    value = value.numpy()
                # 转换 list 为 NumPy 数组
                elif isinstance(value, list):
                    value = np.array(value)
                
                # 保存数据到 HDF5 文件
                if isinstance(value, np.ndarray):
                    sample_group.create_dataset(key, data=value)
                elif isinstance(value, dict):
                    # 转换字典为 JSON 字符串并存储为字节
                    json_str = json.dumps(value)
                    sample_group.create_dataset(key, data=np.string_(json_str))
                elif isinstance(value, str):
                    # 直接存储字符串为字节
                    sample_group.create_dataset(key, data=np.string_(value))
                else:
                    # 不支持的类型抛出异常
                    raise ValueError(f"Unsupported data type for key '{key}': {type(value)}")

class imitation_learning_trainner():
    def __init__(self, args):
        self.subgoalname = "subgoal1"
        self.large_angle_threshold=30
        self.small_angle_threshold=20
        self.forward_angle_threshold=0.5
        self.chtar_threshold=0.3
        self.obj_threshold=0.4
        self.save_path='IL_data/'
        self.sample_num=0

    def train(self, env, tensorboard_log_dir, device, args):
        self.model = MultiModalRobotPolicy_Discrete(transformer_num_layers=args.num_transformer_block)

        self.taskname = env.task.scene_name + '_' + env.task.name
        nav_subgoals = {
            key: subgoal for key, subgoal in env.task.subgoal_conditions.subgoals.items() 
            if 'navigation' in subgoal.get('instruction', '') or 'carry' in subgoal.get('instruction', '')
        }


        env.scene.trav_map.floor_map[0] = self.adjust_trav_map(env)
        for subgoalname, subgoal in nav_subgoals.items():
            # collect_data
            datas = {}
            # try:
            for i in range(3):
                self.subgoalname = subgoalname
                self.subgoal_inst = subgoal['instruction']
                self.subgoal_pred = subgoal['predicates']
                
                # env.task.randomize_initial_pos = True
                # env.task.subgoal_ongoing = self.subgoalname
                # env.reset()
                
                _primitive_controller = StarterSemanticActionPrimitives(env, enable_head_tracking=False)
                # sampled_pose_2d = _primitive_controller._sample_pose_near_object(first_target_object, pose_on_obj=None, distance_lo=3.0, distance_hi=5, yaw_lo=-math.pi, yaw_hi=math.pi)
                # sampled_2d_position = _primitive_controller._sample_pose_in_room(room_name)
                # sampled_pose_2d = _primitive_controller._sample_pose_in_room('0')
                # robot_pose = _primitive_controller._get_robot_pose_from_2d_pose(sampled_pose_2d)
                np.random.seed(int(time.time())) 
                # index = np.random.randint(1, 100)
                index = 1000
                target_id = env.task.subgoal_activity_goal_conditions[self.subgoalname][0].terms[-1] if 'agent' not in env.task.subgoal_activity_goal_conditions[self.subgoalname][0].terms[-1] else env.task.subgoal_activity_goal_conditions[self.subgoalname][0].terms[1]
                target_object = env.scene.object_registry("name", env.scene._scene_info_meta_inst_to_name[target_id])
                break_flag = False
                sample_time = 0
                while True:
                    print(f"Sample time: {sample_time}")
                    try:
                        target_object_2d = _primitive_controller._sample_pose_near_object(target_object, pose_on_obj=None, distance_lo=0.3, distance_hi=0.8, yaw_lo=-math.pi, yaw_hi=math.pi)
                    except:
                        success = False
                        break_flag = True
                        break
                    target_object_2dxy = target_object_2d[:2]
                    # target_object_2dxy = torch.tensor([-0.9277,  3.3871])
                    sample_clear_point = env.scene.get_random_clear_point(0, target_object_2dxy, env.robots[0])
                    if sample_clear_point is None:
                        print("No feasible path")
                    if sample_clear_point is not None:
                        break_flag = False
                        break
                    sample_time += 1
                    if sample_time == 300:
                        break_flag = True
                    if break_flag:
                        break
                if break_flag:
                    continue

                self.target_object_2dxy = target_object_2dxy
                for i in range(index):
                    sampled_pose_2d = sample_clear_point[1][:2]
                    # sampled_pose_2d = torch.tensor([-1.4000, -3.2000])
                    short_path, geo_dis = env.scene.trav_map.get_wide_shortest_path(0, sampled_pose_2d, target_object_2dxy, entire_path=True, extra=2, robot=env.robots[0])
                    if short_path is not None:
                        break
                if short_path is None:
                    continue
                robot_pose = _primitive_controller._get_robot_pose_from_2d_pose(torch.cat([sampled_pose_2d, torch.tensor([0])]))
                env.robots[0].set_position_orientation(robot_pose[0])

                roll_outs_data = self.collect_data(env, device, args)
                nav_images, manip_images, robot_proprio, actions, action_labels, success = roll_outs_data
                if not success:
                    continue
                now = datetime.datetime.now()
                save_minute = now.strftime("%Y%m%d%H%M")
                save_video_with_opencv(nav_images, self.save_path + '/{}/{}/{}/raw_navigation_camera__{}.mp4'.format(self.taskname, self.subgoalname, save_minute, i))
                save_video_with_opencv(manip_images, self.save_path + '/{}/{}/{}/raw_manipulation_camera__{}.mp4'.format(self.taskname, self.subgoalname, save_minute, i))
                data = {
                    "robot_proprio": robot_proprio,
                    "action_labels": action_labels,
                    "subgoal_inst": self.subgoal_inst,
                }
                datas[i] = data
                save_hdf5(datas, self.sample_num, self.save_path + '/{}/{}/{}/hdf5_sensors.hdf5'.format(self.taskname, self.subgoalname, save_minute))
            # except Exception as e:
            #     print(e)
            # self._set_visualization_markers(env, self.short_path)
            # self.plot_short_path(env)
            
    def adjust_trav_map(self, env):
        """
        Adjust the traversability map (trav_map) by marking areas occupied by objects as impassable (value 0).
        
        Args:
            env: The environment object containing scene, floor_map, and objects.
            
        Returns:
            trav_map: The adjusted traversability map with obstacles.
        """
        # 获取 trav_map 的尺寸
        exemption_list = ['robot', 'floor', 'wall', 'ceiling', 'spotlight', 'caepet']
        floor_map = env.scene.trav_map.floor_map[0]  # 获取一个层级的 floor_map
        map_height, map_width = floor_map.shape  # 获取地图的高度和宽度

        # 深拷贝 trav_map，防止修改原始的地图
        trav_map = torch.clone(floor_map)

        # 获取每个物体的 AABB 和位置
        for obj in env.scene.objects:
            if all(sub not in obj.name for sub in exemption_list):
                continue
            obj_pos = obj.get_position()[:2]  # 只关心前两维，获取物体的 (x, y) 坐标
            obj_aabb = obj.aabb  # 获取物体的轴对齐边界框 [min_x, min_y, min_z, max_x, max_y, max_z]
            if 'bush' not in obj.name:
                if obj_aabb[0][2]>1.5:
                    continue
                if obj_aabb[1][2]<0.1:
                    continue
            # 计算物体的 AABB 在地图上的范围（离散化到地图网格空间）

            print(obj.name)
            min_x_map = int(env.scene.trav_map.world_to_map(obj_aabb[0])[0])  # 将最小坐标转换为地图索引
            max_x_map = int(env.scene.trav_map.world_to_map(obj_aabb[1])[0])  # 将最大坐标转换为地图索引
            min_y_map = int(env.scene.trav_map.world_to_map(obj_aabb[0])[1])  # 将最小坐标转换为地图索引
            max_y_map = int(env.scene.trav_map.world_to_map(obj_aabb[1])[1])  # 将最大坐标转换为地图索引

            # 确保索引在地图范围内
            min_x_map = max(0, min_x_map)
            max_x_map = min(map_width - 1, max_x_map)
            min_y_map = max(0, min_y_map)
            max_y_map = min(map_height - 1, max_y_map)

            # 将物体占据的区域标记为不可通行 (值为0)
            trav_map[min_y_map:max_y_map+1, min_x_map:max_x_map+1] = 0

        return trav_map


    def collect_data(self, env, device, args):
        # collect data
        # waypoint = PrimitiveObject(
        #         relative_prim_path=f"/task_waypoint_marker",
        #         primitive_type="Cylinder",
        #         name=f"task_waypoint_marker",
        #         radius=0.03,
        #         height=0.03,
        #         visual_only=True,
        #         rgba=torch.tensor([1, 0, 0, 1]),
        #     )
        # env.scene.add_object(waypoint)
        nav_images = []
        manip_images = []
        robot_proprios = []
        actions = []
        action_labels = []
        # for step in range(1000):
        step = 0
        replanning_times = 0
        self.last_robot_proprio = torch.cat((env.robots[0].get_position(), env.robots[0].get_orientation()))
        self.stuck_counter = 0
        og.sim.viewer_camera.set_orientation(T.axisangle2quat(torch.tensor([0.0, 0.0, -1.0])))
        while True:
            success = False
            obs = env.get_obs()
            robot_position = env.robots[0].get_position()
            robot_orientation = env.robots[0].get_orientation()
            robot_proprio = torch.cat((robot_position, robot_orientation))
            og.sim.viewer_camera.set_position(torch.cat([robot_position[:2], torch.tensor([2.25])], dim=0))
            # input = self.process_obs(obs[0], device)
            # action, action_probs = self.model(input)
            if step == 0:
                gt_action, gt_action_label = self.get_gt_action(env, replanning=True)
            else:
                gt_action, gt_action_label = self.get_gt_action(env, replanning=False)
            
            distance = robot_position[:2] - self.target_object_2dxy
            # try:
            #     next_obs = env.step(gt_action)
            # except Exception as e:
            #     success = False
            #     print(e)
            #     break
            next_obs = env.step(gt_action)
            if robot_position[2] < -0.5:
                success = False  
                break
            # next_filtered_obs = self.process_obs(next_obs)
            nav_images.append(Image.fromarray(obs[0]['external::external_sensor_nav::rgb'].cpu().numpy(), mode="RGBA")) 
            manip_images.append(Image.fromarray(obs[0]['external::external_sensor_manip::rgb'].cpu().numpy(), mode="RGBA")) 
            robot_proprios.append(robot_proprio)
            actions.append(gt_action)
            action_labels.append(gt_action_label)
            step += 1
            if step == 10000:
                success = False
                break
            
            if torch.norm(distance) < self.obj_threshold:
                success = True
                break

            is_stuck = self.is_stuck(env, robot_position, robot_orientation, self.target_object_2dxy)
            if is_stuck:
                self.short_path, geo_dis = env.scene.trav_map.get_wide_shortest_path(0, robot_position[:2], self.target_object_2dxy, entire_path=True, extra=2)
                if self.short_path is None:
                    success = False
                    break
                else:
                    self.short_path = torch.cat([self.short_path, self.target_object_2dxy.unsqueeze(0)], dim=0)
                    self.nav_point_index = 0
                    replanning_times += 1
                if replanning_times > 5:
                    success = False
                    break
            self.last_action = gt_action_label
            # waypoint.set_position_orientation(position = torch.cat([self.short_path[self.nav_point_index], torch.tensor([0.25])]))
            

        if success:
            robot_position = env.robots[0].get_position()
            robot_orientation = env.robots[0].get_orientation()
            robot_proprio = torch.cat((robot_position, robot_orientation))
            nav_images.append(Image.fromarray(next_obs[0]['external::external_sensor_nav::rgb'].cpu().numpy(), mode="RGBA")) 
            manip_images.append(Image.fromarray(next_obs[0]['external::external_sensor_manip::rgb'].cpu().numpy(), mode="RGBA")) 
            robot_proprios.append(robot_proprio)
            
        
        # self._remove_visualization_markers(env)
        
        return nav_images, manip_images, robot_proprios, actions, action_labels, success
            # next_obs, reward, done, info = env.step(action)
            # roll_outs_data.append((obs, action, reward, next_obs, done, info))
            

    def is_stuck(self, env, robot_position, robot_orientation, target_position):
        # Check if the robot is stuck
        robot_position = env.robots[0].get_position()
        robot_orientation = env.robots[0].get_orientation()
        robot_proprio = torch.cat((robot_position, robot_orientation))
        if torch.norm(robot_proprio - self.last_robot_proprio) < 0.0001:
            self.stuck_counter += 1
        
        elif self.last_action == 0 and torch.norm(robot_position[:2] - self.last_robot_proprio[:2]) < 0.001:
            self.stuck_counter += 10
        else:
            self.stuck_counter = 0
        self.last_robot_proprio = robot_proprio
        if self.stuck_counter > 200:
            return True
        return False

    def process_obs(self, obs, device):
        # obs is a dict
        obs = {k: v.unsqueeze(0).to(device) for k,v in obs.items()}
        return obs
    
    def _load_visualization_markers(self, env, shortest_path):
        waypoints = []
        num_nodes = shortest_path.shape[0]
        # floor_height = env.scene.get_floor_height(0)
        for i in range(num_nodes):
            waypoint = PrimitiveObject(
                relative_prim_path=f"/task_waypoint_marker{i}",
                primitive_type="Cylinder",
                name=f"task_waypoint_marker{i}",
                radius=0.02,
                height=0.02,
                visual_only=True,
                rgba=torch.tensor([0, 1, 0, 1]),
            )
            env.scene.add_object(waypoint)
            waypoint.set_position(torch.cat([shortest_path[i], torch.tensor([0.15])]))
            waypoints.append(waypoint)
    def _remove_visualization_markers(self, env):    
        
        # Store waypoints
        self.waypoints = waypoints
        # Remove waypoints
        for waypoint in waypoints:
            env.scene.remove_object(waypoint)
    
    def _set_visualization_markers(self, env, shortest_path):
        floor_height = 0.05
        for i in range(len(self.waypoints)):
            self.waypoints[i].set_position_orientation(
                    position=torch.tensor([shortest_path[i][0], shortest_path[i][1], floor_height])
                )

    def plot_short_path(self, env):
        obs = env.get_obs()[0]
        if 'external::external_sensor::rgb' in obs:
            # Extract and process the RGB data
            obs_rgb = obs['external::external_sensor::rgb'].cpu().numpy()
            obs_rgb = Image.fromarray(obs_rgb, mode="RGBA")
            obs_rgb.save('short_path.png')

    def get_gt_action(self, env, replanning=False):
        if replanning:
            robot = env.robots[0]
            # obj_env_name = env.scene._scene_info_meta_inst_to_name[env.task.subgoal_activity_goal_conditions[self.subgoalname][0].terms[-1]]
            try:
                self.target_object = env.scene.object_registry("name", env.scene._scene_info_meta_inst_to_name[env.task.subgoal_activity_goal_conditions[self.subgoalname][0].terms[-1]])
                room_name = self.target_object.in_rooms[0]
            except:
                self.target_object = env.scene.object_registry("name", env.scene._scene_info_meta_inst_to_name[env.task.subgoal_activity_goal_conditions[self.subgoalname][0].terms[1]])
                room_name = self.target_object.in_rooms[0]
            # self.target_object = env.scene.object_registry("name", obj_env_name)
            robot_position = robot.get_position()
            self.object_position = self.target_object.get_position()
            self.short_path, geo_dis = env.scene.trav_map.get_wide_shortest_path(0, robot_position[:2], self.target_object_2dxy, entire_path=True, extra=2)
            # torch.cat([self.short_path[i*2].unsqueeze(0) for i in range(int(self.short_path.shape[0]/2))] + [self.target_object_2dxy[:2].unsqueeze(0)] , dim=0)
            self.short_path = torch.cat([self.short_path, self.target_object_2dxy[:2].unsqueeze(0)], dim=0) 
            # self.short_path = torch.stack([short_path[i*2] for i in range(0, int((short_path.shape[0]+1)/2))], dim=0)
            
            self.nav_point_index = 0 
            self.last_action=0
            self._load_visualization_markers(env, self.short_path)
        
        
        try:
            target_position = self.short_path[self.nav_point_index]
        except:
            target_position = self.target_object_2dxy[:2]
        robot_position = env.robots[0].get_position()[:2]
        gt_label_prob = torch.tensor([0, 0, 0, 0, 0, 0, 0])

        distance_object = robot_position[:2] - self.target_object_2dxy
        distance_object_plan_point = robot_position[:2] - target_position
        if torch.norm(distance_object) < self.obj_threshold:
            gt_label_index = 5
            gt_label_prob[gt_label_index] = 1.0
            gt_label_action = self.model.policy_network.lower_level_action_tensors[gt_label_index]
            return gt_label_action, gt_label_index

        # check if reach object_position
        distance = target_position - robot_position
        if torch.norm(distance) < self.chtar_threshold:
            self.nav_point_index += 1
            if self.nav_point_index < len(self.short_path):
                target_position = self.short_path[self.nav_point_index]
            elif self.nav_point_index >= self.short_path.shape[0]:
                target_position = self.target_object_2dxy[:2]
            # else:
                # gt_label_index = 5
                # gt_label_prob[gt_label_index] = 1.0
                # gt_label_action = self.model.policy_network.lower_level_action_tensors[gt_label_index]
                # return gt_label_action, gt_label_index
                

        # calculate the direction difference between the robot base orientation and the robot target connecting line
        robot_direction = env.robots[0].get_orientation()
        angle, delta_direction = self.calculate_angle(robot_direction, robot_position, target_position)
        if delta_direction == 'slight_right':
            gt_label_index = 4
            gt_label_prob[gt_label_index] = 1.0
            gt_label_action = self.model.policy_network.lower_level_action_tensors[gt_label_index]
        elif delta_direction == 'sharp_right':
            gt_label_index = 3
            gt_label_prob[gt_label_index] = 1.0
            gt_label_action = self.model.policy_network.lower_level_action_tensors[gt_label_index]
        elif delta_direction == 'slight_left':
            gt_label_index = 2
            gt_label_prob[gt_label_index] = 1.0
            gt_label_action = self.model.policy_network.lower_level_action_tensors[gt_label_index]
        elif delta_direction == 'sharp_left':
            gt_label_index = 1
            gt_label_prob[gt_label_index] = 1.0
            gt_label_action = self.model.policy_network.lower_level_action_tensors[gt_label_index]
        else:
            gt_label_index = 0
            gt_label_prob[gt_label_index] = 1.0
            gt_label_action = self.model.policy_network.lower_level_action_tensors[gt_label_index]
        
        print(f"Now the delta angle is {angle}")
        print(delta_direction)
        print("DISTANCE", torch.norm(distance_object))
        print(f"DISTANCE_PLAN_POINT: {torch.norm(distance_object_plan_point)} | {self.nav_point_index}")
        return gt_label_action, gt_label_index




    def calculate_angle(self, robot_direction, robot_position, target_position):
    # Convert quaternion to yaw angle (rotation around the z-axis)
        r = R.from_quat(robot_direction.tolist())
        euler_angles = torch.tensor (r.as_euler('xyz', degrees=True))
        robot_yaw = euler_angles[2]  # yaw in degrees
        
        # Compute vector from robot to target
        direction_vector = target_position - robot_position
        
        # Calculate angle of the vector relative to the x-axis
        target_angle = torch.rad2deg(torch.atan2(direction_vector[1], direction_vector[0]))
        
        # Calculate the difference between target angle and robot's yaw
        angle_diff = target_angle - robot_yaw
        
        # Normalize angle to [-180, 180]
        angle_diff = (angle_diff + 180) % 360 - 180
        
        # Determine turning direction
        # 4 slight_right 3 sharp_right 2 slight_left 1 sharp_left 0 forward
        if self.last_action == 0:
            if angle_diff > self.small_angle_threshold:
                delta_direction = "slight_left"
            elif angle_diff < -self.small_angle_threshold:
                delta_direction = "slight_right"
            else:
                delta_direction = "aligned"

        
        elif self.last_action ==1:
            if angle_diff < self.small_angle_threshold:
                delta_direction = "slight_left"
            else:
                delta_direction = "sharp_left"
        
        elif self.last_action == 2:
            if angle_diff > self.large_angle_threshold:
                delta_direction = "sharp_left"
            elif angle_diff < self.forward_angle_threshold:
                delta_direction = "aligned"
            else:
                delta_direction = "slight_left"
        
        elif self.last_action == 3:
            if angle_diff > -self.large_angle_threshold:
                delta_direction = "slight_right"
            else:
                delta_direction = "sharp_right"

        elif self.last_action == 4:
            if angle_diff < -self.large_angle_threshold:
                delta_direction = "sharp_right"
            elif angle_diff > -self.forward_angle_threshold:
                delta_direction = "aligned"
            else:
                delta_direction = "slight_right"

        # if self.last_action != 0:
        #     if angle_diff > self.large_angle_threshold:
        #         delta_direction = "sharp_left"
        #     elif angle_diff > self.small_angle_threshold:
        #         delta_direction = "slight_left"
        #     elif angle_diff < -self.large_angle_threshold:
        #         delta_direction = "sharp_right"
        #     elif angle_diff < -self.small_angle_threshold:
        #         delta_direction = 'slight_right'
        #     else:
        #         delta_direction = "aligned"
        # else:
        #     if angle_diff > self.forward_angle_threshold:
        #         delta_direction = "slight_left"
        #     elif angle_diff < -self.forward_angle_threshold:
        #         delta_direction = "slight_right"
        #     else:
        #         delta_direction = "aligned"
        
        return angle_diff, delta_direction
