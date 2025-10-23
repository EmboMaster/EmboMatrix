import os
import shutil
import math

def divide_and_copy(source_dir, target_dir):
    # 获取源目录下所有的子路径
    all_subpaths = [f for f in os.listdir(source_dir) if os.path.isdir(os.path.join(source_dir, f))]

    # 计算每一份应该包含的子路径数
    num_parts = 8
    part_size = math.ceil(len(all_subpaths) / num_parts)

    # 确保目标目录存在
    os.makedirs(target_dir, exist_ok=True)

    # 将子路径平均分成8份，并分别复制到目标目录
    for i in range(num_parts):
        part_subpaths = all_subpaths[i*part_size:(i+1)*part_size]
        part_target_dir = os.path.join(target_dir, str(i))  # 创建0,1,2,...的子目录
        os.makedirs(part_target_dir, exist_ok=True)
        
        # 将每个子路径复制到相应的目标目录
        for subpath in part_subpaths:
            src_path = os.path.join(source_dir, subpath)
            dst_path = os.path.join(part_target_dir, subpath)
            shutil.copytree(src_path, dst_path)
            print(f"复制 {src_path} 到 {dst_path}")

# 使用示例
source_dir = 'omnigibson/shengyin/new_results2'  # 源目录
target_dir = 'omnigibson/shengyin/new_results_experiment'  # 目标目录

divide_and_copy(source_dir, target_dir)