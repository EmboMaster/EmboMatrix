def process_goal_section(bddl_content):
    """
    处理 PDDL 文件内容，提取 goal 部分的动作队列。
    
    参数:
        pddl_content (str): PDDL 文件的内容
    
    返回:
        list: 动作队列，例如 [['open', 'carton.n.02_1'], ['ontop', 'water_bottle.n.01_1', 'shelf.n.01_1'], ['inside', 'backpack.n.01_1', 'hutch.n.01_1']]
    """
    # 找到 goal 部分，剪切掉之前的内容
    goal_start = bddl_content.find("(:goal")
    if goal_start == -1:
        raise ValueError("未找到 goal 部分")
    goal_content = bddl_content[goal_start:]

    # 按行分割为字符串队列
    lines = goal_content.splitlines()

    # 只保留首尾都是括号的内容
    actions = [line.strip() for line in lines if line.strip().startswith("(") and line.strip().endswith(")")]

    # 去除问号并分割为列表
    processed_goals = [action.replace("?", "").strip("()").split() for action in actions]

    return processed_goals

def readbddl(file_path):
    """
    读取 BDDL 文件并返回其内容的字符串形式。
    
    参数:
        file_path (str): BDDL 文件的路径
    
    返回:
        str: 文件内容的字符串形式
    """
    try:
        with open(file_path, 'r') as file:
            content = file.read()
        return content
    except FileNotFoundError:
        raise FileNotFoundError(f"文件未找到: {file_path}")
    except Exception as e:
        raise RuntimeError(f"读取文件时发生错误: {e}")

import json

def replace_objects_in_bddl(bddl_content, json_path):
    """
    将 BDDL 文件中的物体名称替换为 JSON 文件中 inst_to_name 对应键的键值。
    
    参数:
        bddl_content (str): BDDL 文件的内容（字符串形式）。
        json_path (str): JSON 文件的路径。
    
    返回:
        str: 替换后的 BDDL 文件内容。
    """
    # 读取 JSON 文件
    try:
        with open(json_path, 'r') as json_file:
            json_data = json.load(json_file)
            inst_to_name = json_data["metadata"]["task"]["inst_to_name"]
    except FileNotFoundError:
        raise FileNotFoundError(f"JSON 文件未找到: {json_path}")
    except KeyError:
        raise KeyError("JSON 文件中缺少 'metadata.task.inst_to_name' 键")
    except Exception as e:
        raise RuntimeError(f"读取 JSON 文件时发生错误: {e}")
    
    # 替换 BDDL 内容中的物体名称
    for key, value in inst_to_name.items():
        bddl_content = bddl_content.replace(key, value)
    
    return bddl_content

def main():

    # 示例用法
    bddl_path = "/GPFS/rhome/yuanzhuoding/planner-bench/planner_bench/omnigibson/shengyin/results0424/Beechwood_1_int/Bathroom_Water_Shelf_Backpack_Hutch.bddl"
    json_path = "/GPFS/rhome/yuanzhuoding/planner-bench/planner_bench/omnigibson/shengyin/results0424/Beechwood_1_int/Bathroom_Water_Shelf_Backpack_Hutch.json"

    # 读取 BDDL 文件内容
    with open(bddl_path, 'r') as bddl_file:
        bddl_content = bddl_file.read()

    # 替换物体名称
    updated_bddl_content = replace_objects_in_bddl(bddl_content, json_path)

    # 输出替换后的内容
    print(updated_bddl_content)

actions = process_goal_section(updated_bddl_content)
print(actions)