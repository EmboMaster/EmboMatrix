from env_reward_fastapi import start_server  # 导入之前定义的 start_server 函数

def main():
    scene = "Beechwood_0_garden"  # 设置场景名称
    task = "bring_book_to_office"    # 设置任务名称
    cuda_device_index = 0    # 设置 CUDA 设备索引
    port = 5000              # 设置服务器端口

    # 调用 start_server 启动服务器
    start_server(scene, task, cuda_device_index, port)

# if __name__ == "__main__":
#     main()
# import os
# import shutil

# # 源路径
# source_dir = '/data/zxlei/embodied/planner_bench/omnigibson/result0304'
# # 目标路径
# destination_dir = '/data/zxlei/embodied/planner_bench/bddl/bddl/bddl_tocheck0303'

# # 确保目标路径存在
# os.makedirs(destination_dir, exist_ok=True)

# # 遍历源路径及其子目录
# for root, dirs, files in os.walk(source_dir):
#     for file in files:
#         # 如果文件以.bddl结尾
#         if file.endswith('.bddl'):
#             # 构建完整的源文件路径
#             source_file = os.path.join(root, file)
#             # 构建完整的目标文件路径
#             destination_file = os.path.join(destination_dir, file)
#             # 复制文件
#             shutil.copy(source_file, destination_file)
#             print(f"文件 {file} 复制到 {destination_file}")
# import os
# import shutil

# # 源路径
# source_dir = '/data/zxlei/embodied/planner_bench/omnigibson/result0304'
# # 目标路径
# destination_dir = '/data/zxlei/embodied/planner_bench/bddl/bddl/activity_definitions'

# # 确保目标路径存在
# os.makedirs(destination_dir, exist_ok=True)

# # 遍历源路径及其子目录
# for root, dirs, files in os.walk(source_dir):
#     for file in files:
#         # 如果文件以.bddl结尾
#         if file.endswith('.bddl'):
#             # 构建完整的源文件路径
#             source_file = os.path.join(root, file)
#             # 提取文件名（不包括路径）
#             filename_without_extension = os.path.splitext(file)[0]
#             # 创建目标文件夹路径
#             new_folder = os.path.join(destination_dir, filename_without_extension)

#             # 创建目标文件夹
#             os.makedirs(new_folder, exist_ok=True)

#             # 构建目标文件路径
#             destination_file = os.path.join(new_folder, 'problem0.bddl')

#             # 将源文件复制到新文件夹并重命名为 problem0.bddl
#             shutil.copy(source_file, destination_file)

#             print(f"文件 {file} 从 {source_file} 复制到 {destination_file}")
# import os 
# import shutil
# source_dir = '/data/zxlei/embodied/planner_bench/omnigibson/result0304'
# target_dir = '/data/zxlei/embodied/planner_bench/omnigibson/plannerdemo/plancache'

# for root, dirs, files in os.walk(source_dir):
#     for file in files:
#         if file.endswith('.json'):
#             source_file = os.path.join(root, file)
#             new_name = os.path.basename(root) + '_' + file
#             target_file = os.path.join(target_dir, new_name)
#             #print(f"source_file: {source_file}\ntarget_file: {target_file}")
#             #import pdb;pdb.set_trace()
#             shutil.copy(source_file, target_dir)
#             #print(f"文件 {file} 复制到 {target_dir}")