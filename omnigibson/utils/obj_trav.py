import os
import glob
import json

def process_json_file(file_path):
    # 直接调用 read_and_process_json 并返回数据
    return read_and_process_json(file_path)

def traverse_and_process_json_files(directory_path):
    all_data = {}
    pattern = os.path.join(directory_path, "**/*_best.json")
    for json_file in glob.glob(pattern, recursive=True):
        result = process_json_file(json_file)
        # 假设每个文件返回的是一个独立的字典，我们将这些字典合并为一个大字典
        all_data.update(result)
    return all_data

def read_and_process_json(file_path):
    result_dict = {}
    with open(file_path, 'r') as file:
        data = json.load(file)
    
    init_info = data['objects_info']['init_info']
    for key, value in init_info.items():
        category = value['args']['category']
        object_info = value
        if category not in result_dict:
            result_dict[category] = []
        result_dict[category].append(object_info)

    return result_dict

# 指定输入目录和输出文件路径
directory_path = '/home/magic-4090/miniconda3/envs/omnigibson/lib/python3.10/site-packages/omnigibson/data/og_dataset/scenes/'
output_path = '/home/magic-4090/miniconda3/envs/omnigibson/lib/python3.10/site-packages/omnigibson/data/a.json'

# 调用函数遍历和处理文件
data = traverse_and_process_json_files(directory_path)

# 确保输出目录存在
os.makedirs(os.path.dirname(output_path), exist_ok=True)

# 将所有结果写入一个 JSON 文件
with open(output_path, 'w') as f:
    json.dump(data, f, indent=4)
