import torch
from omnigibson.model.baseline_discrete.baseline_model_IL import MultiModalRobotPolicy_Discrete
import math
from scipy.spatial.transform import Rotation as R
from omnigibson.utils.motion_planning_utils import plan_arm_motion_ik, plan_arm_motion
from omnigibson.action_primitives.starter_semantic_action_primitives import PlanningContext, RobotCopy, StarterSemanticActionPrimitives, indented_print
import omnigibson.utils.transform_utils as T
from omnigibson.utils.grasping_planning_utils import get_grasp_poses_for_object_sticky
import omnigibson as og
from omnigibson.macros import create_module_macros
import omnigibson.lazy as lazy
from tqdm import tqdm

import torch.nn.functional as F

class imitation_learning_trainner():
    def __init__(self, args, env):
        self.subgoalname = "subgoal1"
        self.angle_threshold=3
        self.chtar_threshold=0.1
        self.robot_copy = self._load_robot_copy(env, env.robots[0])
        self.operator = StarterSemanticActionPrimitives(env, env.robots[0], self.robot_copy)

    def _load_robot_copy(self, env, robot):
        """Loads a copy of the robot that can be manipulated into arbitrary configurations for collision checking in planning."""
        robot_copy = RobotCopy()

        robots_to_copy = {"original": {"robot": robot, "copy_path": robot.prim_path + "_copy"}}

        for robot_type, rc in robots_to_copy.items():
            copy_robot = None
            copy_robot_meshes = {}
            copy_robot_meshes_relative_poses = {}
            copy_robot_links_relative_poses = {}

            # Create prim under which robot meshes are nested and set position
            lazy.omni.usd.commands.CreatePrimCommand("Xform", rc["copy_path"]).do()
            copy_robot = lazy.omni.isaac.core.utils.prims.get_prim_at_path(rc["copy_path"])
            reset_pose = robot_copy.reset_pose[robot_type]
            translation = lazy.pxr.Gf.Vec3d(*reset_pose[0].tolist())
            copy_robot.GetAttribute("xformOp:translate").Set(translation)
            orientation = reset_pose[1][[3, 0, 1, 2]]
            copy_robot.GetAttribute("xformOp:orient").Set(lazy.pxr.Gf.Quatd(*orientation.tolist()))

            robot_to_copy = None
            if robot_type == "simplified":
                robot_to_copy = rc["robot"]
                env.scene.add_object(robot_to_copy)
            else:
                robot_to_copy = rc["robot"]

            # Copy robot meshes
            for link in robot_to_copy.links.values():
                link_name = link.prim_path.split("/")[-1]
                # if "base" not in link_name:
                #     continue
                continue
                for mesh_name, mesh in link.collision_meshes.items():
                    split_path = mesh.prim_path.split("/")
                    # Do not copy grasping frame (this is necessary for Tiago, but should be cleaned up in the future)
                    if "grasping_frame" in link_name:
                        continue

                    copy_mesh_path = rc["copy_path"] + "/" + link_name
                    copy_mesh_path += f"_{split_path[-1]}" if split_path[-1] != "collisions" else ""
                    lazy.omni.usd.commands.CopyPrimCommand(mesh.prim_path, path_to=copy_mesh_path).do()
                    copy_mesh = lazy.omni.isaac.core.utils.prims.get_prim_at_path(copy_mesh_path)
                    relative_pose = T.relative_pose_transform(
                        *mesh.get_position_orientation(), *link.get_position_orientation()
                    )
                    relative_pose = (relative_pose[0], torch.tensor([0, 0, 0, 1]))
                    if link_name not in copy_robot_meshes.keys():
                        copy_robot_meshes[link_name] = {mesh_name: copy_mesh}
                        copy_robot_meshes_relative_poses[link_name] = {mesh_name: relative_pose}
                    else:
                        copy_robot_meshes[link_name][mesh_name] = copy_mesh
                        copy_robot_meshes_relative_poses[link_name][mesh_name] = relative_pose

                copy_robot_links_relative_poses[link_name] = T.relative_pose_transform(
                    *link.get_position_orientation(), *robot.get_position_orientation()
                )

            if robot_type == "simplified":
                env.scene.remove_object(robot_to_copy)

            robot_copy.prims[robot_type] = copy_robot
            robot_copy.meshes[robot_type] = copy_robot_meshes
            robot_copy.relative_poses[robot_type] = copy_robot_meshes_relative_poses
            robot_copy.links_relative_poses[robot_type] = copy_robot_links_relative_poses

        og.sim.step()
        return robot_copy

    def train(self, env, tensorboard_log_dir, device, args):
        self.model = MultiModalRobotPolicy_Discrete(transformer_num_layers=args.num_transformer_block)

        for step in range(100000):
            # collect_data
            roll_outs_data = self.collect_data(env, device, args)

    def collect_data(self, env, device, args):
        # collect data
        roll_outs_data = []
        for step in tqdm(range(1000)):
            obs = env.get_obs()
            input = self.process_obs(obs[0], device)
            # action, action_probs = self.model(input)
            # if step  == 0:
            #     gt_action_prob, gt_action = self.get_gt_action(env, replanning=True)
            # else:
            #     gt_action_prob, gt_action = self.get_gt_action(env, replanning=False)
            if step % 30 == 0:
                replanning = True
            else:
                replanning = False
            gt_action = self.get_gt_action(env, replanning=True)
            next_obs = env.step(gt_action)
            roll_outs_data.append((obs, gt_action, next_obs))
            # next_obs, reward, done, info = env.step(action)
            # roll_outs_data.append((obs, action, reward, next_obs, done, info))
            
    def process_obs(self, obs, device):
        # obs is a dict
        obs = {k: v.unsqueeze(0).to(device) for k,v in obs.items()}
        return obs

    def get_gt_action(self, env, replanning=False):
        # if replanning:
        robot = env.robots[0]
        robot_copy = self.robot_copy
        obj_env_name = env.scene._scene_info_meta_inst_to_name['legal_document.n.01_2' ]
        self.target_object = env.scene.object_registry("name", obj_env_name)
        action = self.multiple_stage_plan(env, robot, robot_copy, replanning)
        
        # self.short_path = plan_arm_motion_ik(robot, object_position, env.scene)
        return action

    def multiple_stage_plan(self, env, robot, robot_copy, replanning):
        # First Stage: move arm to the x,y position of the target object
        eef_relative_postition_now = robot.get_relative_eef_pose()

        robot_position = robot.get_position()
        robot_orientation = robot.get_orientation()
        target_object_position = self.target_object.get_position()
        target_object_orientation = self.target_object.get_orientation()

        target_object_pose = self.target_object.get_position_orientation()
        robot_pose=robot.get_position_orientation()
        eef_relative_pose = robot.get_relative_eef_pose()

        relative_target_pose_T = T_get_relative_pose(target_object_pose, robot_pose)
        # robot_base_target_stage1 = torch.cat([relative_target_pose_T[0][:2], robot_position[2].unsqueeze(0), T.quat2axisangle(eef_relative_pose[1])])
        sticky_pose = get_grasp_poses_for_object_sticky(self.target_object)
        sticky_relative_pose = T_get_relative_pose(sticky_pose[0][0], robot_pose)
        sticky_relative_pose_6d = torch.cat([sticky_relative_pose[0], T.quat2axisangle(sticky_relative_pose[1])])
        robot_base_target_stage1 = torch.cat([eef_relative_pose[0]-torch.tensor([0,0,0.3]), T.quat2axisangle(eef_relative_pose[1])])

        if replanning:
            try:
                with PlanningContext(env, robot, robot_copy, "original") as context:
                    newplan = plan_arm_motion_ik(
                        robot=robot,
                        end_conf=robot_base_target_stage1,
                        context=context,
                        torso_fixed=False,
                        planning_time=30.0,
                    )
                    if newplan not in [None, []]:
                        self.plan = newplan
            except:
                pass
                
        plan = self.plan
        # indented_print(f"Plan has {len(plan)} steps")
        # for i, target_pose in enumerate(plan):
        if len(plan) >= 1:
            i=0
            target_pose=plan[0]
            target_pos = target_pose[:3]
            target_orn = T.axisangle2quat(target_pose[3:])
            indented_print(f"Executing grasp plan step {i + 1}/{len(plan)}")
            # yield from self._move_hand_direct_ik(
            #     (target_pos, target_quat), ignore_failure=True, in_world_frame=False, stop_if_stuck=stop_if_stuck
            # )
        
            controller_config = self.operator.robot._controller_config["arm_" + self.operator.arm]
            assert (
                controller_config["name"] == "InverseKinematicsController"
            ), "Controller must be InverseKinematicsController"
            assert controller_config["mode"] == "pose_absolute_ori", "Controller must be in pose_absolute_ori mode"
            
            target_pos = target_pose[0]
            target_orn_axisangle = target_pose[3:]
            # target_orn = target_pose[1]
            # target_orn_axisangle = T.quat2axisangle(target_pose[1])
            control_idx = self.operator.robot.controller_action_idx["arm_" + self.operator.arm]
            prev_pos = prev_orn = None

            # All we need to do here is save the target IK position so that empty action takes us towards it
            controller_name = f"arm_{self.operator.arm}"
            self.operator._arm_targets[controller_name] = (target_pos, target_orn_axisangle)
            pos_thresh=0.01
            ori_thresh=0.1
                    # for i in range(1000):
            current_pose = self.operator._get_pose_in_robot_frame(
                (self.operator.robot.get_eef_position(self.operator.arm), self.operator.robot.get_eef_orientation(self.operator.arm))
            )
            current_pos = current_pose[0]
            current_orn = current_pose[1]

            delta_pos = target_pos - current_pos
            target_pos_diff = torch.norm(delta_pos)
            target_orn_diff = T.get_orientation_diff_in_radian(current_orn, target_orn)
            reached_goal = target_pos_diff < pos_thresh and target_orn_diff < ori_thresh
            if reached_goal:
                return

            # if stop_on_contact and detect_robot_collision_in_sim(self.robot, ignore_obj_in_hand=False):
            #     return

            # if i > 0 and stop_if_stuck and detect_robot_collision_in_sim(self.robot, ignore_obj_in_hand=False):
            if i > 0 and False:
                pos_diff = torch.norm(prev_pos - current_pos)
                orn_diff = T.get_orientation_diff_in_radian(current_orn, prev_orn)
                if pos_diff < 0.0003 and orn_diff < 0.01:
                    # raise ActionPrimitiveError(ActionPrimitiveError.Reason.EXECUTION_ERROR, f"Hand is stuck")
                    print("Hand is stuck")

            prev_pos = current_pos
            prev_orn = current_orn

            # Since we set the new IK target as the arm_target, the empty action will take us towards it.
            action = self.operator._empty_action()
            # yield self.operator._postprocess_action(action)
            # action = self.operator._postprocess_action(action)


        return action


def T_get_relative_pose(eef_pose_ori, base_pose_ori, mat=False):
    """
    Args:
        arm (str): specific arm to grab eef pose. Default is "default" which corresponds to the first entry
            in self.arm_names
        mat (bool): whether to return pose in matrix form (mat=True) or (pos, quat) tuple (mat=False)

    Returns:
        2-tuple or (4, 4)-array: End-effector pose, either in 4x4 homogeneous
            matrix form (if @mat=True) or (pos, quat) tuple (if @mat=False), corresponding to arm @arm
    """
    # arm = self.default_arm if arm == "default" else arm
    # eef_link_pose = self.eef_links[arm].get_position_orientation()
    # base_link_pose = self.get_position_orientation()
    pose = T.relative_pose_transform(*eef_pose_ori, *base_pose_ori)
    return T.pose2mat(pose) if mat else pose


def calculate_angle(self, robot_direction, robot_position, target_position):
# Convert quaternion to yaw angle (rotation around the z-axis)
    r = R.from_quat(robot_direction.tolist())
    euler_angles = torch.tensor(r.as_euler('xyz', degrees=True))
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
    if angle_diff > self.angle_threshold:
        delta_direction = "turn left"
    elif angle_diff < -self.angle_threshold:
        delta_direction = "turn right"
    else:
        delta_direction = "aligned"
    
    return angle_diff, delta_direction

def quat_inverse(q):
    # 四元数 q = (w, x, y, z)
    # 四元数的逆是 q_inv = (w, -x, -y, -z)
    return torch.tensor([q[0], -q[1], -q[2], -q[3]])

# 四元数乘法
def quat_multiply(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return torch.tensor([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    ])

# 位置和方向从世界坐标系转换到机器人基座坐标系
def rotate_vector_by_quaternion(vector, quat):
    q_vector = torch.cat([torch.tensor([0.0]), vector])  # 形成四元数 (0, x, y, z)
    q_conjugate = quat_inverse(quat)  # 获取四元数的共轭
    q_rotated = quat_multiply(quat_multiply(quat, q_vector), q_conjugate)  # 进行旋转
    return q_rotated[1:]  # 返回旋转后的向量部分 (x, y, z)

# 物体的相对位置计算
def get_relative_pose(object_position, object_orientation, robot_position, robot_orientation):
    # 计算物体相对机器人基座的初始位置
    relative_position = object_position - robot_position

    # 旋转物体的相对位置到机器人基座坐标系
    relative_position_in_robot_base = rotate_vector_by_quaternion(relative_position, robot_orientation)

    # 将物体的世界坐标系下的方向变换到机器人基座坐标系下
    relative_orientation = quat_multiply(quat_inverse(robot_orientation), object_orientation)

    return relative_position_in_robot_base, relative_orientation
    # 将物体的世界坐标系下的方向变换到机器人基座坐标系下
    relative_orientation = quat_multiply(robot_orientation_inv, object_orientation)

    return relative_position, relative_orientation
