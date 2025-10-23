import os
import json
import glob
import time
from datetime import datetime
from pathlib import Path
from openai import OpenAI

# 配置 OpenAI 客户端


# 基础路径
base_dir = "/GPFS/rhome/yuanzhuoding/planner-bench/planner_bench/omnigibson/data/og_dataset/scenes_with_newfetch_ys"
posecache_dir = "/GPFS/rhome/yuanzhuoding/planner-bench/planner_bench/omnigibson/plannerdemo/simulation_tools/posecache"
output_dir = "/GPFS/rhome/yuanzhuoding/planner-bench/planner_bench/omnigibson/data/og_dataset/llm_plan_results_ys/0423"
real_bddl_dir = "/GPFS/rhome/yuanzhuoding/planner-bench/planner_bench/omnigibson/shengyin/results0423"

if_deepseek = True

# 时间段过滤（示例：2024-10-01 00:00:00 之后）
start_time = datetime(2025, 4, 24).timestamp()

def get_tmp2(posecache_path):
    """读取 posecache JSON 文件，生成 tmp2 字典"""
    try:
        with open(posecache_path, 'r') as f:
            data = json.load(f)
        # 生成 tmp2：keys 是 JSON 的键，values 是索引
        tmp2 = {key: idx for idx, key in enumerate(data.keys())}
        return tmp2
    except Exception as e:
        # print(f"Error reading {posecache_path}: {e}")
        return {}

def call_llm(prompt):
    """调用 OpenAI API 获取 LLM 生成结果"""

    if if_deepseek:
        client = OpenAI(
            base_url='https://deepseek.ctyun.zgci.ac.cn:10443/v1',
            api_key='eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJZeXh4dXdCZUNzT21ZZDRETUFCTTVxY2pUSkJUZEpjbiJ9.K86sfQHEDnRd97pJpq7x8yCrj9YImq7M-Eo6SXKzrF4'
            )
        try:
            stream = client.chat.completions.create(
                messages=[
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
                model="DeepSeek-R1",
                stream=True,
            )
            result = []
            for chunk in stream:
                result.append(chunk.choices[0].delta.content or "")
            result = "".join(result)
            return result
        except Exception as e:
            print(f"Error calling LLM: {e}")
            return None
    else:

        try:
            response = client.chat.completions.create(
                model="o4-mini",
                messages=[
                    {"role": "user", "content": prompt},
                ],
                temperature=0.7,
                max_tokens=1000,
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error calling LLM: {e}")
            return None
        
def filter_tmp2_by_json_objects(bddl_dir, scene_name, task_name, tmp2):
    """
    读取 JSON 文件，提取 metadata.task.inst_to_name 中的 obj-name 列表，
    过滤 tmp2 字典，只保留键在 obj-name 列表中的项。

    参数：
        bddl_dir (str): JSON 文件所在的基础目录
        scene_name (str): 场景名称（如 Beechwood_0_int）
        task_name (str): 任务名称（如 please_bring_the_projector_...）
        tmp2 (dict): 需要过滤的字典

    返回：
        dict: 过滤后的 tmp2 字典
    """
    json_path = bddl_dir

    try:
        # 读取 JSON 文件
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 提取 inst_to_name 字典
        inst_to_name = data.get("metadata", {}).get("task", {}).get("inst_to_name", {})
        
        # 获取所有 obj-name（inst_to_name 的值）
        obj_names = list(inst_to_name.values())
        
        # 过滤 tmp2，只保留键在 obj_names 中的项
        filtered_tmp2 = {key: value for key, value in tmp2.items() if key in obj_names}
        
        return filtered_tmp2
    
    except FileNotFoundError:
        print(f"JSON file not found: {json_path}")
        return tmp2  # 返回原 tmp2
    except json.JSONDecodeError:
        print(f"Invalid JSON format in: {json_path}")
        return tmp2  # 返回原 tmp2
    except Exception as e:
        print(f"Error processing {json_path}: {e}")
        return tmp2  # 返回原 tmp2

def extract_bddl_problem(bddl_file_path):
    """
    读取 BDDL 文件，提取 problem 部分的名称。

    参数：
        bddl_file_path (str): BDDL 文件的完整路径

    返回：
        str: problem 部分的名称，如果未找到或文件无效则返回 None
    """
    try:
        with open(bddl_file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 查找 (define (problem ...) 行
        for line in content.splitlines():
            line = line.strip()
            if line.startswith("(define (problem "):
                # 提取 problem 名称
                start = len("(define (problem ")
                end = line.find(")", start)
                if end != -1:
                    problem_name = line[start:end]
                    return problem_name
        print(f"No problem definition found in {bddl_file_path}")
        return None
    
    except FileNotFoundError:
        print(f"BDDL file not found: {bddl_file_path}")
        return None
    except Exception as e:
        print(f"Error reading {bddl_file_path}: {e}")
        return None

def process_json_files():
    """主函数：处理 JSON 文件"""
    # 查找所有 JSON 文件
    json_files = glob.glob(os.path.join(base_dir, "**", "*_plan_prompt.json"), recursive=True)
    
    for json_file in json_files:
        # 检查文件是否在时间段后生成
        file_mtime = os.path.getmtime(json_file)
        if file_mtime < start_time:
            continue
            
        # 提取 scene_name 和 task_name
        json_path = Path(json_file)
        scene_name = json_path.parent.parent.name  # 子文件夹名，如 Beechwood_0_int
        task_name = json_path.stem.replace("_plan_prompt","")  # 文件名（去掉 .json），如 please_bring_...

        #import pdb; pdb.set_trace()

        # 读取原始 JSON 文件
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
        except Exception as e:
            # print(f"Error reading {json_file}: {e}")
            continue

        # 读取 posecache JSON 文件
        posecache_path = os.path.join(posecache_dir, scene_name, f"{task_name}.json")
        if not os.path.exists(posecache_path):
            continue

        bddl_dir = os.path.join(real_bddl_dir, scene_name, f"{task_name}.json")

        another_bddl_dir = os.path.join(real_bddl_dir, scene_name, f"{task_name}.bddl")
        problem = extract_bddl_problem(another_bddl_dir)

        tmp2 = get_tmp2(posecache_path)
        if not tmp2:
            print(f"Skipping {json_file} due to empty tmp2")
            continue

        tmp2 = filter_tmp2_by_json_objects(bddl_dir, scene_name, task_name, tmp2)

        print(tmp2)

        # 修改 prompt
        original_prompt = data.get("plan_prompt", "")
        modified_prompt = f"{original_prompt}\nHere are the object indexes: {tmp2}"

        # 调用 LLM
        llm_result = call_llm(modified_prompt)
        if llm_result is None:
            print(f"Skipping {json_file} due to LLM failure")
            continue

        # 保存结果
        output_scene_dir = os.path.join(output_dir, scene_name)
        os.makedirs(output_scene_dir, exist_ok=True)
        if if_deepseek:
            output_path = os.path.join(output_scene_dir, f"{task_name}_deepseek.json")
        else:
            output_path = os.path.join(output_scene_dir, f"{task_name}.json")
        
        try:
            with open(output_path, 'w') as f:
                json.dump({"problem":problem,"object_index": tmp2, "result": llm_result}, f, indent=4)
            print(f"Saved result to {output_path}")
        except Exception as e:
            print(f"Error saving {output_path}: {e}")

if __name__ == "__main__":
    process_json_files()