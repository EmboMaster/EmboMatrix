import matplotlib.pyplot as plt
import numpy as np

def sample_line_in_rectangle(x0, y0, dx, dy, xmin, ymin, xmax, ymax, N):
    """
    给定直线参数化公式 (x, y) = (x0, y0) + t*(dx, dy) ，
    计算直线与矩形 [xmin, ymin, xmax, ymax] 的交线段中 t 的取值范围，
    然后在该范围上均匀采样 N 个点，返回对应位置列表.
    如果直线与区域无交集，则返回空列表.
    """
    t_min = -float("inf")
    t_max = float("inf")
    
    # x方向的不等式：xmin <= x0 + t*dx <= xmax
    if dx != 0:
        t1 = (xmin - x0) / dx
        t2 = (xmax - x0) / dx
        t_low = min(t1, t2)
        t_high = max(t1, t2)
        t_min = max(t_min, t_low)
        t_max = min(t_max, t_high)
    else:
        if not (xmin <= x0 <= xmax):
            return []  # x0 不在区域内
    
    # y方向的不等式：ymin <= y0 + t*dy <= ymax
    if dy != 0:
        t1 = (ymin - y0) / dy
        t2 = (ymax - y0) / dy
        t_low = min(t1, t2)
        t_high = max(t1, t2)
        t_min = max(t_min, t_low)
        t_max = min(t_max, t_high)
    else:
        if not (ymin <= y0 <= ymax):
            return []
    
    if t_max < t_min:
        return []
    
    # 均匀采样 t 值
    if N == 1:
        ts = [(t_min + t_max) / 2.0]
    else:
        ts = [t_min + i * (t_max - t_min) / (N - 1) for i in range(N)]
    
    positions = [(x0 + t * dx, y0 + t * dy) for t in ts]
    return positions

def sample_uniform_in_rectangle(xmin, ymin, xmax, ymax, N):
    """
    在矩形内均匀采样点，生成 N x N 个候选位置。
    """
    xs = np.linspace(xmin, xmax, N)
    ys = np.linspace(ymin, ymax, N)
    positions = []
    for x in xs:
        for y in ys:
            positions.append((x, y))
    return positions

def get_B_candidates(xA=0, yA=0, aA=0, bA=1, xC=0, yC=0, aC=0, bC=1, xmin=0, ymin=0, xmax=10, ymax=10, N=20, mode=0):
    """
    根据 mode 生成物体 B 的候选位置及其朝向：
    
    参数：
      xA, yA, aA, bA: 物体 A 的中心位置和朝向（aA,bA）  
      xC, yC, aC, bC: 物体 C 的中心位置和朝向（C 的朝向仅用于备用方向）  
      xmin, ymin, xmax, ymax: 区域边界  
      N: 当 mode==0 或 mode==2 时，表示在 A 对齐直线上采样的点数；
         当 mode==1 时，表示每个维度上均匀采样的数量（总数为 N*N）。
      mode:
         0 – 仅 A center align（使用 A 的朝向产生 4 个备用方向）
         1 – 仅 face to C（在整个区域均匀采样，朝向指向 C）
         2 – 同时满足 A center align 和 face to C
         
    返回：
      positions: 候选位置列表 (x,y)
      orientations: 对应朝向列表 (ox, oy)
    """
    positions_all = []
    orientations_all = []
    
    if mode == 0:
        # 仅 A center align：候选位置采样自两条直线
        pos_line1 = sample_line_in_rectangle(xA, yA, aA, bA, xmin, ymin, xmax, ymax, N)
        pos_line2 = sample_line_in_rectangle(xA, yA, -bA, aA, xmin, ymin, xmax, ymax, N)
        all_positions = pos_line1 + pos_line2
        
        # 对于每个位置，输出 A 的朝向的4个旋转作为备用
        # 旋转公式：0°: (aA,bA), 90°: (-bA, aA), 180°: (-aA,-bA), 270°: (bA,-aA)
        backup_orientations = [
            (aA, bA),
            (-bA, aA),
            (-aA, -bA),
            (bA, -aA)
        ]
        for pos in all_positions:
            for ori in backup_orientations:
                positions_all.append(pos)
                orientations_all.append(ori)
                
    elif mode == 1:
        # 仅 face to C：候选位置在整个区域均匀采样（N x N个点）
        all_positions = sample_uniform_in_rectangle(xmin, ymin, xmax, ymax, N)
        
        # 对于每个位置，计算朝向指向 C 的单位向量
        # 如果候选点与 C 重合，则使用 C 的朝向的4个旋转备用
        backup_orientations = [
            (aC, bC),
            (-bC, aC),
            (-aC, -bC),
            (bC, -aC)
        ]
        for pos in all_positions:
            x, y = pos
            vx, vy = (xC - x, yC - y)
            norm = np.hypot(vx, vy)
            if norm > 1e-6:
                orientation = (vx / norm, vy / norm)
                positions_all.append(pos)
                orientations_all.append(orientation)
            else:
                for ori in backup_orientations:
                    positions_all.append(pos)
                    orientations_all.append(ori)
                    
    elif mode == 2:
        # 同时满足 A center align 和 face to C：
        # 候选位置采样自经过 A 中心的两条直线，
        # 每个候选位置的朝向为指向 C 的单位向量；若候选点与 C 重合，则备用 C 朝向的4个旋转。
        pos_line1 = sample_line_in_rectangle(xA, yA, aA, bA, xmin, ymin, xmax, ymax, N)
        pos_line2 = sample_line_in_rectangle(xA, yA, -bA, aA, xmin, ymin, xmax, ymax, N)
        all_positions = pos_line1 + pos_line2
        
        backup_orientations = [
            (aC, bC),
            (-bC, aC),
            (-aC, -bC),
            (bC, -aC)
        ]
        for pos in all_positions:
            x, y = pos
            vx, vy = (xC - x, yC - y)
            norm = np.hypot(vx, vy)
            if norm > 1e-6:
                orientation = (vx / norm, vy / norm)
                positions_all.append(pos)
                orientations_all.append(orientation)
            else:
                for ori in backup_orientations:
                    positions_all.append(pos)
                    orientations_all.append(ori)
    else:
        raise ValueError("mode 必须为 0, 1 或 2")
    
    return positions_all, orientations_all

def visualize_candidates(xA, yA, aA, bA, xC, yC, aC, bC, xmin, ymin, xmax, ymax, positions, orientations, mode):
    """
    可视化显示：
      - 区域边界
      - 物体 A（红色）及其朝向
      - 物体 C（绿色）及其朝向（仅作标记）
      - 候选的物体 B 位置及其朝向（蓝色箭头）
      - 在标题中标明当前 mode
    """
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # 绘制区域矩形
    rect = plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, fill=False, edgecolor='black', linewidth=2)
    ax.add_patch(rect)
    
    # 绘制物体 A（红色）
    ax.plot(xA, yA, 'ro', label='物体 A')
    ax.arrow(xA, yA, aA, bA, head_width=0.3, head_length=0.3, fc='r', ec='r')
    
    # 绘制物体 C（绿色）
    ax.plot(xC, yC, 'go', label='物体 C')
    ax.arrow(xC, yC, aC, bC, head_width=0.3, head_length=0.3, fc='g', ec='g')
    
    # 绘制候选的物体 B 位置及朝向（蓝色）
    for (x, y), (ox, oy) in zip(positions, orientations):
        ax.plot(x, y, 'bo', markersize=3)
        ax.arrow(x, y, 0.5*ox, 0.5*oy, head_width=0.15, head_length=0.15, fc='b', ec='b')
    
    ax.set_xlim(xmin - 1, xmax + 1)
    ax.set_ylim(ymin - 1, ymax + 1)
    ax.set_aspect('equal')
    ax.set_title(f"物体 B 候选位置 (mode={mode})")
    ax.legend()
    plt.grid(True)
    plt.show()

import numpy as np
import matplotlib.pyplot as plt

def check_B_position(xA, yA, aA, bA, xB, yB, eps=1e-6, scale_factor=1e6):
    """
    判断物体B是否在物体A的前面以及是否在A的侧面（左右侧均算）
    
    参数：
      xA, yA: 物体A的中心位置
      aA, bA: 物体A的朝向（假设为非零向量）
      xB, yB: 物体B的中心位置
      eps: 判断向量是否接近零的容差
      
    返回：
      is_front: 如果B在A的前方（夹角 < 45°），返回True，否则False
      is_side:  如果B在A的侧面（夹角在 [45°, 135°] 之间），返回True，否则False
    """
    # 计算 A 的单位朝向向量
    u = np.array([aA, bA]) * scale_factor
    norm_u = np.linalg.norm(u)
    if norm_u < eps:
        if abs(aA) <= abs(bA) and bA >= 0:
            u = np.array([0, 1])
        elif abs(aA) <= abs(bA) and bA < 0:
            u = np.array([0, -1])
        elif abs(aA) > abs(bA) and aA >= 0:
            u = np.array([1, 0])
        else:
            u = np.array([-1, 0])
    else:
        u = u / norm_u
    
    # 计算从 A 到 B 的向量
    v = np.array([xB - xA, yB - yA])
    norm_v = np.linalg.norm(v)
    if norm_v < eps:
        # A与B重合时，无法判断方向
        return False, False
    
    v_norm = v / norm_v
    # 计算夹角（单位为°）
    # 由于 arccos 返回 [0, pi]，角度范围为 [0, 180]
    theta = np.degrees(np.arccos(np.clip(np.dot(u, v_norm), -1.0, 1.0)))
    
    # 判断是否在前面（夹角小于45°）
    is_front = (theta < 45)
    # 判断是否在侧面（夹角介于45°和135°之间）
    is_side = (theta >= 45 and theta <= 135)
    return is_front, is_side

def random_B_samples(n, x_range, y_range):
    """
    随机生成 n 个物体B的位置和朝向  
    物体B的朝向为随机角度（单位向量），位置在 x_range 和 y_range 范围内  
    返回列表：positions=[(x,y), ...], orientations=[(ox,oy), ...]
    """
    positions = []
    orientations = []
    for _ in range(n):
        x = np.random.uniform(*x_range)
        y = np.random.uniform(*y_range)
        positions.append((x, y))
        angle = np.random.uniform(0, 2*np.pi)
        orientations.append((np.cos(angle), np.sin(angle)))
    return positions, orientations

def visualize_positions(xA, yA, aA, bA, B_positions, B_orientations, B_flags):
    """
    可视化：
      - 物体A的位置和朝向（红色）
      - 50个B的位置，根据判断结果上色：
          前面：绿色
          侧面（但不在前面）：蓝色
          其他（后面或重合）：灰色
      - B的朝向用短箭头表示（颜色同B点）
    参数：
      B_flags: 列表，每个元素为 (is_front, is_side)
    """
    fig, ax = plt.subplots(figsize=(8,8))
    
    # 绘制 A 的位置和朝向（红色）
    ax.plot(xA, yA, 'ro', markersize=10, label='物体 A')
    ax.arrow(xA, yA, aA, bA, head_width=0.3, head_length=0.3, fc='r', ec='r')
    
    # 绘制B
    for (xB, yB), (ox, oy), (is_front, is_side) in zip(B_positions, B_orientations, B_flags):
        # 根据判断结果选择颜色：
        # 如果在前面，用绿色
        # 如果在侧面（且不在前面），用蓝色
        # 否则用灰色
        if is_front:
            color = 'green'
            label = 'B in front'
        elif is_side:
            color = 'blue'
            label = 'B on side'
        else:
            color = 'gray'
            label = 'B others'
        ax.plot(xB, yB, 'o', color=color)
        # 绘制B的朝向（缩放系数0.5）
        ax.arrow(xB, yB, 0.5*ox, 0.5*oy, head_width=0.15, head_length=0.15, fc=color, ec=color)
    
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.set_aspect('equal')
    ax.set_title("随机生成50个B的位置及其相对于A的方位")
    ax.grid(True)
    # 为避免重复图例，可以只显示一次图例说明
    handles = [plt.Line2D([0], [0], marker='o', color='w', label='A (红)', markerfacecolor='r', markersize=10),
               plt.Line2D([0], [0], marker='o', color='w', label='B in front (绿)', markerfacecolor='green', markersize=8),
               plt.Line2D([0], [0], marker='o', color='w', label='B on side (蓝)', markerfacecolor='blue', markersize=8),
               plt.Line2D([0], [0], marker='o', color='w', label='B others (灰)', markerfacecolor='gray', markersize=8)]
    ax.legend(handles=handles, loc='upper right')
    plt.show()

import math
def angle_between_vectors(v1, v2):
    """计算两个二维向量之间的夹角（单位：度）"""
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    if norm1 < 1e-6 or norm2 < 1e-6:
        return None
    dot = np.dot(v1, v2)
    cos_angle = np.clip(dot / (norm1 * norm2), -1.0, 1.0)
    angle = math.degrees(math.acos(cos_angle))
    return angle

def vector_angle(v):
    """计算二维向量的角度，返回 [0,360) 度"""
    angle = math.degrees(math.atan2(v[1], v[0]))
    return angle % 360

def check_alignment_and_face(x1, y1, a1, b1,
                             x2, y2, a2, b2,
                             x3, y3, a3, b3,
                             tol_center_line=5,
                             tol_ori=5,
                             tol_face=5):
    """
    输入：
      物体A的位置 (x1, y1) 及方向 (a1, b1)
      物体B的位置 (x2, y2) 及方向 (a2, b2)
      物体C的位置 (x3, y3) 及方向 (a3, b3)（C 的方向本身不参与 face-to 的判断）
    
    返回：
      center_aligned: 是否能勉强实现 center aligned，
                      要求 A 到 B 的中心连线与下列四个方向之一的夹角不超过 tol_center_line：
                         - A 的方向
                         - A 的反方向
                         - A 的法向（+90°）
                         - A 的反法向（-90°）
      同时要求 B 的方向与 A 的方向的差值（取模360后与0,90,180,270中最接近值的误差）不超过 tol_ori。
      
      face_to: 是否能勉强实现 face to，
               要求 B 的方向与 B 到 C 的中心连线的夹角不超过 tol_face（默认10°）。
      
      另外还返回计算得到的各个角度：center_line 最小夹角、B与A方向的差值（diff）、B->C 连线与B方向的夹角。
    """
    # --- 判断 center aligned ---
    # 计算 A 到 B 的中心连线
    vec_AB = np.array([x2 - x1, y2 - y1])
    # 定义 A 的四个方向：
    A_dir = np.array([a1, b1])
    A_dir_inv = -A_dir
    # A 的法向：顺时针旋转90°，反法向：逆时针旋转90°
    A_dir_perp = np.array([-b1, a1])
    A_dir_perp_inv = -A_dir_perp

    # 计算与四个方向的夹角
    angles = []
    for d in [A_dir, A_dir_inv, A_dir_perp, A_dir_perp_inv]:
        angle = angle_between_vectors(vec_AB, d)
        if angle is not None:
            angles.append(angle)
    if not angles:
        min_angle = None
    else:
        min_angle = min(angles)
    cond1 = (min_angle is not None and min_angle <= tol_center_line)

    # 判断条件2：B 的方向与 A 的方向的相对差值应接近 0, 90, 180, 或 270°，误差不超过 tol_ori
    angle_A = vector_angle(A_dir)
    B_dir = np.array([a2, b2])
    angle_B = vector_angle(B_dir)
    diff = (angle_B - angle_A) % 360
    candidates = [0, 90, 180, 270]
    errors = [abs(diff - cand) for cand in candidates]
    min_error = min(errors)
    cond2 = (min_error <= tol_ori)

    center_aligned = cond1 and cond2

    # --- 判断 face to ---
    vec_BC = np.array([x3 - x2, y3 - y2])
    angle_face = angle_between_vectors(vec_BC, B_dir)
    face_to = (angle_face is not None and angle_face <= tol_face)
    
    return center_aligned, face_to, min_angle, diff, angle_face