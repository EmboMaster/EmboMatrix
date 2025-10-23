import json
import re
import os
def read_lines_to_array(file_path):
    """
    从指定文件路径读取内容，将每一行作为字符串存储在数组中。
    
    参数:
        file_path (str): 文件的路径。
    
    返回:
        list: 包含文件中每一行内容的字符串数组。
    """
    try:
        # 打开文件并读取所有行
        with open(file_path, 'r', encoding='utf-8') as file:
            lines = file.readlines()
        
        # 去除每行末尾的换行符
        lines = [line.strip() for line in lines]
        
        return lines
    except FileNotFoundError:
        print(f"文件未找到，请检查路径是否正确：{file_path}")
        return []
    except Exception as e:
        print(f"读取文件时发生错误：{e}")
        return []
    
def read_json_file(file_path):
    """从指定路径读取JSON文件内容"""
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            return json.load(file)
    except FileNotFoundError:
        print(f"错误：文件 {file_path} 未找到。")
        return None
    except json.JSONDecodeError:
        print(f"错误：文件 {file_path} 不是有效的JSON格式。")
        return None

def extract_room_areas(data):
    """从JSON数据中提取每个场景的房间面积"""
    result = []
    for scene, rooms in data.items():
        room_list = []
        for room, area in rooms.items():
            room_list.append(f'"{room}": {area}')
        result.append(",\n".join(room_list))
    return result

def extract_commands(data):
    # 使用正则表达式匹配 [[]] 包裹的内容，支持多行
    pattern = r'\[\[(.*?)\]\]'
    # 使用 re.DOTALL 使 . 匹配包括换行符在内的所有字符
    matches = re.findall(pattern, data, re.DOTALL)
    # 将所有匹配的内容拼接成一个字符串
    return "\n".join(matches).strip()

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

    # 找到首尾都是括号的内容中最小的括号包含的内容
    actions = [line.strip() for line in lines if line.strip().startswith("(") and line.strip().endswith(")")]

    # 使用正则表达式提取最小括号中的内容
    processed_goals = []
    for action in actions:
    # 匹配最小括号中的内容
        matches = re.findall(r'\([^\(\)]+\)', action)
        processed_goals.extend([
            [item if i == 0 else item.rsplit('_', 1)[0]  # 保留第一个元素（状态），对后续元素去除 '_1'
            for i, item in enumerate(match.strip("()").replace("?", "").split())]
            for match in matches
        ])

    return processed_goals

def process_init_section(bddl_content):
    """
    处理 PDDL 文件内容，提取 goal 部分的动作队列。
    
    参数:
        pddl_content (str): PDDL 文件的内容
    
    返回:
        list: 动作队列，例如 [['open', 'carton.n.02_1'], ['ontop', 'water_bottle.n.01_1', 'shelf.n.01_1'], ['inside', 'backpack.n.01_1', 'hutch.n.01_1']]
    """
    # 找到 goal 部分，剪切掉之前的内容
    init_start = bddl_content.find("(:init")
    goal_end = bddl_content.find("(:goal")
    if init_start == -1:
        raise ValueError("未找到 init 部分")
    if goal_end == -1:
        raise ValueError("未找到 goal 部分")
    init_content = bddl_content[init_start:goal_end]

    # 按行分割为字符串队列
    lines = init_content.splitlines()



    # 找到首尾都是括号的内容中最小的括号包含的内容
    actions = [line.strip() for line in lines if line.strip().startswith("(") and line.strip().endswith(")")]

    # 使用正则表达式提取最小括号中的内容
    processed_inits = []
    for action in actions:
    # 匹配最小括号中的内容
        matches = re.findall(r'\([^\(\)]+\)', action)
        processed_inits.extend([
            [item if i == 0 else item.rsplit('_', 1)[0]  # 保留第一个元素（状态），对后续元素去除 '_1'
            for i, item in enumerate(match.strip("()").replace("?", "").split())]
            for match in matches
        ])

    return processed_inits

def validate_goal_states(goal_actions, bddl_content, base_path):
    """
    校验 goal 动作中的物体是否在对应的状态文件中。

    参数:
        goal_actions (list): 动作队列，例如 [['cook', 'apple.n.01'], ['freeze', 'water.n.01']]
        base_path (str): 状态文件的基路径，例如 '/GPFS/rhome/yuanzhuoding/embodied-bench/omnigibson/shengyin/bddl_gen2_for_hwh/data/objects_sorted_by_states/'
        bddl_content (str): BDDL 文件的内容，用于检查是否存在切割工具等。
    返回:
        bool: 如果所有物体都存在于对应的状态文件中，返回 True；否则返回 False。
    """
    # 定义状态与文件的映射
    state_files = {
        'cooked': 'cookable_object.txt',
        'frozen': 'freezable_object.txt',
        'open': 'openable_object.txt',
        'toggled_on': 'toggleable_object.txt',
        'ontop': '',
        'inroom': '',
        'folded': '',
        'unfolded': '',
        'hot': '',
        'on_fire': '',
        'real': '',
        'saturated': '',
        'covered': '',
        'filled': '',
        'contains': '',
        'under': '',
        'touching': '',
        'inside': 'fillable_object.txt',
        'overlaid': '',
        'attached': '',
        'draped': '',
        'insource': '',
        'broken': '',
        'dusty': '',
        'stained': 'stainable_object.txt',
        'soaked': '',
        'sliced': 'sliceable_object.txt'  # 添加 sliced 状态的文件映射
    }
    with open(os.path.join(base_path,'cleaner_object.txt'), 'r') as f:
        cleaner_objects = [line.strip() for line in f if line.strip()]   
    with open(os.path.join(base_path,'knife_object.txt'), 'r') as f:
        knife_objects = [line.strip() for line in f if line.strip()] 

    for action in goal_actions:
        if len(action) < 2:
            print(f"动作格式无效: {action}")
            return False  # 动作格式无效，直接返回 False

        state, obj = action[0], action[1]
        if action[1] == '-':
            continue  # 如果中间的是 '-'，则说明不是动作，而是逻辑内容，例如forall和exists等
        if state == 'inside':
            obj = action[2]  # 如果状态是 inside，物体是第三个参数
        if state == 'inroom':
            # 如果状态是 inroom，检查物体是否在房间内
            print(f"物体 {obj} 的状态为 inroom，不能出现在状态文件中。")
            return False
        if state in state_files:
            #检查是否需要特定工具
            if state == 'soaked':
                if 'sink' not in bddl_content:
                    print(f"物体 {obj} 需要清洗工具来达到浸湿状态，但在 BDDL 中未找到相关工具。")
                    return False
            if state  == 'sliced':
                get_knife = False
                for knife in knife_objects:
                    if knife in bddl_content:
                        get_knife = True
                if not get_knife:
                    print(f"物体 {obj} 需要切割工具来达到切割状态，但在 BDDL 中未找到相关工具。")
                    return False
            if state in ['stained', 'dusty']:
                get_cleaner = False
                for cleaner in cleaner_objects:
                    if cleaner in bddl_content:
                        get_cleaner = True
                if 'sink' not in bddl_content:
                    get_cleaner = False
                if not get_cleaner:
                    print(f"物体 {obj} 需要清洗工具来达到洁净状态，但在 BDDL 中未找到相关工具。")
                    return False
            if state_files[state] == '':
                # 对于没有对应文件的状态，直接跳过校验
                continue
            # 拼接文件路径
            file_path = os.path.join(base_path, state_files[state])
            try:
                with open(file_path, 'r') as f:
                    valid_objects = set(line.strip() for line in f)
                if obj not in valid_objects:
                    print(f"物体 {obj} 不在状态 {state} 的文件中。")
                    return False  # 物体不在对应的状态文件中，直接返回 False
            except FileNotFoundError:
                print(f"状态文件 {file_path} 未找到。")
                return False  # 状态文件不存在，直接返回 False
        else:
            print(f"未知状态: {state}，请检查状态文件。")
            return False
    print("goal 动作校验通过")
    return True  # 所有校验通过，返回 True

def validate_init_states(init_actions, base_path):
    """
    校验 init 动作中的物体是否在对应的状态文件中。

    参数:
        init_actions (list): 动作队列，例如 [['cooked', 'apple.n.01'], ['frozen', 'water.n.01']]
        base_path (str): 状态文件的基路径，例如 '/GPFS/rhome/yuanzhuoding/embodied-bench/omnigibson/shengyin/bddl_gen2_for_hwh/data/objects_sorted_by_states/'

    返回:
        bool: 如果所有物体都存在于对应的状态文件中，返回 True；否则返回 False。
    """
    # 定义状态与文件的映射
    state_files = {
        'cooked': 'cookable_object.txt',
        'frozen': 'freezable_object.txt',
        'open': 'openable_object.txt',
        'toggled_on': 'toggleable_object.txt',
        'ontop': '',
        'inroom': '',
        'folded': '',
        'unfolded': '',
        'hot': '',
        'on_fire': '',
        'real': '',
        'saturated': '',
        'covered': '',
        'filled': '',
        'contains': '',
        'under': '',
        'touching': '',
        'inside': 'fillable_object.txt',
        'overlaid': '',
        'attached': '',
        'draped': '',
        'insource': '',
        'broken': '',
        'dusty': '',
        'soaked': '',
        'stained': '',
        'sliced': ''
    }

    for action in init_actions:
        if len(action) < 2:
            print(f"动作格式无效: {action}")
            return False  # 动作格式无效，直接返回 False

        state, obj = action[0], action[1]
        if state == 'inside':
            obj = action[2]  # 如果状态是 inside，物体是第三个参数
        if state in state_files:
            if state_files[state] == '':
                # 对于没有对应文件的状态，直接跳过校验
                continue
            file_path = os.path.join(base_path, state_files[state])
            try:
                with open(file_path, 'r') as f:
                    valid_objects = set(line.strip() for line in f)
                if obj not in valid_objects:
                    print(f"物体 {obj} 不在状态 {state} 的文件中。")
                    return False  # 物体不在对应的状态文件中，直接返回 False
            except FileNotFoundError:
                print(f"状态文件 {file_path} 未找到。")
                return False  # 状态文件不存在，直接返回 False
        else:
            print(f"未知状态: {state}，请检查状态文件。")
            return False  # 未知状态，直接返回 False
    print("init 动作校验通过")
    return True  # 所有校验通过，返回 True



def extract_character_names(data):
    """
    从给定的字符串中提取角色名称，并去重（保持顺序）。

    参数:
        data (str): 包含角色名称的字符串。

    返回:
        list: 提取到的不重复角色名称列表。
    """
    # 使用正则表达式匹配所有以 [ 开头和 ] 结尾的内容
    pattern = r'\[(.*?)\]'
    matches = re.findall(pattern, data)
    
    # 去重并保持顺序
    unique_matches = list(dict.fromkeys(matches))
    return unique_matches

def clean_bddl_content(content):
    """清理 BDDL 文件内容，去除在 :init 之后出现的带关键词的行。"""

    in_init = False
    cleaned_lines = []
    skip_keywords = {"dusty", "stained", "soaked", "sliced"}

    for line in content.splitlines():
        stripped = line.strip()

        # 进入 :init 后开始处理
        if not in_init and stripped.startswith("(:init"):
            in_init = True
            cleaned_lines.append(line)
            continue

        if in_init:
            # 只跳过含有关键字的行，其余保留
            if any(keyword in stripped for keyword in skip_keywords):
                continue

        cleaned_lines.append(line)

    return "\n".join(cleaned_lines)


import json
import re
import os

def from_bddl_file_to_jsonl_file_for_training(bddl_path):
    def clean_varname(name):
        return name.lstrip('?')  # 去掉前缀问号

    def parse_predicates(lines):
        state = {}
        relation = []

        for line in lines:
            line = line.strip()
            if not line or line.startswith(';'):
                continue

            is_neg = line.startswith('(not')
            predicate_match = re.match(r'\(not\s*\((.*?)\)\)', line) if is_neg else re.match(r'\((.*?)\)', line)

            if not predicate_match:
                continue

            tokens = predicate_match.group(1).split()
            predicate = tokens[0]
            args = [clean_varname(arg) for arg in tokens[1:]]

            if len(args) == 1:
                obj = args[0]
                if obj not in state:
                    state[obj] = {}
                state[obj][predicate] = not is_neg
            elif len(args) >= 2:
                if is_neg:
                    continue
                relation.append(tuple([args[0], predicate] + args[1:]))

        return {'state': state, 'relation': relation}

    with open(bddl_path, 'r') as f:
        lines = f.readlines()

    init_lines = []
    goal_lines = []
    in_init = False
    in_goal = False
    paren_count = 0

    for line in lines:
        line_strip = line.strip()
        if line_strip.startswith('(:init'):
            in_init = True
            paren_count = line.count('(') - line.count(')')
            continue
        elif line_strip.startswith('(:goal'):
            in_goal = True
            paren_count = line.count('(') - line.count(')')
            continue

        if in_init:
            init_lines.append(line_strip)
            paren_count += line.count('(') - line.count(')')
            if paren_count <= 0:
                in_init = False
        elif in_goal:
            goal_lines.append(line_strip)
            paren_count += line.count('(') - line.count(')')
            if paren_count <= 0:
                in_goal = False

    init_data = parse_predicates(init_lines)
    goal_data = parse_predicates(goal_lines)

    output_path = os.path.join(os.path.dirname(bddl_path), 'problem0.jsonl')

    with open(output_path, 'w') as out_f:
        out_f.write(json.dumps(init_data) + '\n')
        out_f.write(json.dumps(goal_data) + '\n')

    print(f"Written to {output_path}")

import os

def process_all_bddl_files(root_folder):
    """
    遍历 root_folder/level1/level2 结构，处理每个 level2 文件夹中的 .bddl 文件，
    并在对应目录下输出 problem0.jsonl。
    假设每个 level2 文件夹中只有一个 .bddl 文件。
    """
    for level1_name in os.listdir(root_folder):
        level1_path = os.path.join(root_folder, level1_name)
        if not os.path.isdir(level1_path):
            continue

        for level2_name in os.listdir(level1_path):
            level2_path = os.path.join(level1_path, level2_name)
            if not os.path.isdir(level2_path):
                continue

            # 查找该 level2 文件夹中的 .bddl 文件
            bddl_files = [f for f in os.listdir(level2_path) if f.endswith('.bddl')]
            if not bddl_files:
                print(f"[Skip] No .bddl file in {level2_path}")
                continue

            if len(bddl_files) > 1:
                print(f"[Warning] More than one .bddl file in {level2_path}, only processing the first one.")

            bddl_path = os.path.join(level2_path, bddl_files[0])
            try:
                print(f"Processing: {bddl_path}")
                from_bddl_file_to_jsonl_file_for_training(bddl_path)
            except Exception as e:
                print(f"[Error] Failed to process {bddl_path}: {e}")

def copy_init_first_to_goal(content: str) -> str:
    lines = content.splitlines()

    # 找到 (:init 的位置
    init_idx = None
    for i, line in enumerate(lines):
        if "(:init" in line:
            init_idx = i + 1
            break
    if init_idx is None or init_idx >= len(lines):
        return content

    # 从 init 后面开始查找第一个以 (inroom floor 开头的行
    init_first_line = None
    for j in range(init_idx, len(lines)):
        if lines[j].strip().startswith("(inroom floor"):
            init_first_line = lines[j]
            break
    if init_first_line is None:
        return content  # 没找到就直接返回原内容

    # 找到 (:goal 位置并插入
    for i, line in enumerate(lines):
        if "(:goal" in line:
            lines.insert(i + 1, init_first_line)
            break

    return "\n".join(lines)


# from_bddl_file_to_jsonl_file_for_training('data/data-0730/school_gym/CleanBenchesGym/problem0.bddl')

# process_all_bddl_files('data/data-0730')
# content ='''
# (define (problem Clean_all_3_benches_in_gym_0_using_the_rag_from_locker_room_1-0)
#     (:domain omnigibson)
#     (:objects
#         bench.n.01_1 bench.n.01_2 bench.n.01_3 - bench.n.01
#         rag.n.01_1 - rag.n.01
#         floor.n.01_1 - floor.n.01
#         agent.n.01_1 - agent.n.01
#     )

#     (:init
#         (stained bench.n.01_1)
#         (not(soaked bench.n.01_2))
#         (soaked bench.n.01_3)
#         (inroom rag.n.01_1 locker_room_1)
#         (ontop agent.n.01_1 floor.n.01_1)
#         (inroom floor.n.01_1 gym_0)
#     )

#     (:goal
#         (and
#             (not (dusty ?bench.n.01_1))
#             (not (dusty ?bench.n.01_2))
#             (not (dusty ?bench.n.01_3))
#             (soaked ?rag.n.01_1)
#         )
#     )
# )'''
# cleaned_content = clean_bddl_content(content)
# print(cleaned_content)
bddl_content = '''(define (problem slice_the_green_onion.n.01_on_the_chopping_board_in_the_kitchen_0_using_the_cleaver.n.01)
    (:domain omnigibson)
    (:objects
        green_onion.n.01_1 - green_onion.n.01
        chopping_board.n.01_1 - chopping_board.n.01
        cleaver.n.01_1 - cleaver.n.01
        floor.n.01_1 - floor.n.01
        agent.n.01_1 - agent.n.01
    )

    (:init
        (ontop green_onion.n.01_1 chopping_board.n.01_1)
        (inroom chopping_board.n.01_1 kitchen_0)
        (inroom cleaver.n.01_1 kitchen_0)
        (inroom floor.n.01_1 kitchen_0)
        (ontop agent.n.01_1 floor.n.01_1)
    )

    (:goal
        (and
            (sliced green_onion.n.01_1)
        )
    )
)'''
print(copy_init_first_to_goal(bddl_content))
