import os

# 目标目录路径
directory_path = "/data/zxlei/embodied/planner_bench/bddl/bddl/activity_definitions"

# 遍历目录下的所有文件
for root, dirs, files in os.walk(directory_path):
    for file in files:
        # 只处理 .bddl 文件
        if file.endswith(".bddl") and file != "problem0.bddl":
            old_file_path = os.path.join(root, file)
            new_file_path = os.path.join(root, "problem0.bddl")
            
            # 重命名文件为 problem0.bddl
            os.rename(old_file_path, new_file_path)

print("文件重命名完成。")
