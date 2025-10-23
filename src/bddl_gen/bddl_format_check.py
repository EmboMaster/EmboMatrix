import re
import os
# 示例输出内容
output = """(define (problem water_the_plants-0)
    (:domain omnigibson)

    (:objects
        watering_can.n.01_1 - watering_can.n.01
        plant_pot.n.01_1 plant_pot.n.01_2 - plant_pot.n.01
        water.n.06_1 - water.n.06
        garden.n.01_1 - garden.n.01
        agent.n.01_1 - agent.n.01
    )
    (:init
        (filled watering_can.n.01_1 water.n.06_1)
        (ontop plant_pot.n.01_1 garden.n.01_1)
        (ontop plant_pot.n.01_2 garden.n.01_1)
        (ontop agent.n.01_1 garden.n.01_1)
    )
    (:goal
        (and
            (filled ?plant_pot.n.01_1 ?water.n.06_1)
            (filled ?plant_pot.n.01_2 ?water.n.06_1)
        )
    )
)"""

# 正则表达式提取每个 BDDL 文件的任务名称和内容
def extract_bddl_files(output):
    # 匹配整个 BDDL 文件内容的正则表达式
    if not isinstance(output, str):
        print("Error: output is not a string!")
        output = str(output)  # 将非字符串类型转换为字符串
    
    pattern = r'(\(define \(problem [^\)]+\)[^)]*\))\n(.*?)\n\)\n'

    # 查找所有符合的内容
    matches = re.findall(pattern, output, re.DOTALL)

    # 提取任务名称和文件内容并存储到列表
    bddl_files = []

    for match in matches:
        full_content = match[0] + match[1] + '\n' + ')' + '\n'  + '\n'# 合并 (define (problem ...) 和任务内容
        task_name_match = re.search(r'\(problem ([^\)]+)\)', match[0])  # 提取任务名称
        task_name = task_name_match.group(1) if task_name_match else None
        bddl_files.append({"name": task_name, "content": full_content})

    return bddl_files

def validate_and_fix_bddl_format(task_name, task_content):
    try:
        # 检查 :domain 是否存在
        if not re.search(r"\(:domain [^\)]+\)", task_content):
            return False, f"任务 {task_name} 缺少 :domain 声明。", task_content

        # 检查 :objects 是否存在
        if not re.search(r"\(:objects(.*?)\)", task_content, re.DOTALL):
            return False, f"任务 {task_name} 缺少 :objects 声明。", task_content

        # 检查 :init 是否存在
        if not re.search(r"\(:init(.*?)\)", task_content, re.DOTALL):
            return False, f"任务 {task_name} 缺少 :init 声明。", task_content

        # 检查 :goal 是否存在
        if not re.search(r"\(:goal(.*?)\)", task_content, re.DOTALL):
            return False, f"任务 {task_name} 缺少 :goal 声明。", task_content

        # 检查括号是否匹配并尝试修复
        stack = []
        fixed_content = task_content[:]
        unmatched_right_parens = 0

        # 遍历任务内容并检查括号匹配
        for i, char in enumerate(fixed_content):
            if char == '(':
                stack.append(i)  # 记录左括号位置
            elif char == ')':
                if not stack:  # 如果没有匹配的左括号
                    unmatched_right_parens += 1  # 记录多余的右括号
                else:
                    stack.pop()  # 匹配一个左括号

        # 如果栈不为空，表示有左括号没有匹配
        if stack:
            return False, f"任务 {task_name} 括号不匹配。"

        # 如果格式检查通过，则返回修复后的内容
        return True, f"任务 {task_name} 格式正确。"
    except Exception as e:
        return False, f"任务 {task_name} 检查时发生错误: {str(e)}"
    
def save_task_content(task_name, task_content, result,create,path):
    filename = f"{path}/txt/checked_bddl.txt"
    if result and create:#创建一个新的txt文档
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, "w") as file:
            file.write(task_content)
    elif result and ~create:#后续添加内容
        with open(filename, "a") as file:
            file.write(task_content)
    else:
        print("任务"+task_name+"有误，未保存文件内容。")

def bddl_format_check(raw_bddl,path):
        # 提取所有 BDDL 文件
    bddl_files = extract_bddl_files(raw_bddl)
    create_bddl_files = True
    # 验证每个 BDDL 文件的格式
    for file in bddl_files:
        result, message= validate_and_fix_bddl_format(file['name'], file['content'])
        print(message)
        save_task_content(file['name'], file['content'], result,create_bddl_files,path)
        create_bddl_files = False

def remove_first_and_last_lines(text):
    """
    删除字符串的第一行和最后一行
    :param text: 输入的字符串
    :return: 删除第一行和最后一行后的字符串
    """
    # 按行分割字符串
    lines = text.splitlines()
    # 删除第一行和最后一行
    if len(lines) > 2:
        new_lines = lines[0:-1]  # 删除第一行和最后一行
    elif len(lines) == 2:
        new_lines = []  # 如果只有两行，删除后为空
    else:
        new_lines = []  # 如果只有一行或为空，返回空列表
    # 将剩余的行重新组合成一个字符串
    return "\n".join(new_lines)
