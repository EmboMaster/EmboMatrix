import os
import sys

def parse_s_expression(s: str):
    """
    将一个 Lisp 风格的 S-表达式字符串解析成一个 Python 嵌套列表。
    例如："(a (b c) d)" -> ['a', ['b', 'c'], 'd']
    """
    # 在括号周围添加空格，方便按空格分割成 token
    s = s.replace('(', ' ( ').replace(')', ' ) ')
    tokens = s.split()
    
    # 一个内部递归函数，用于从 token 列表中构建树形结构
    def read_from_tokens(token_list):
        if not token_list:
            raise SyntaxError("在解析时遇到预料之外的文件结尾")
        
        token = token_list.pop(0)
        if token == '(':
            # 如果是左括号，开始一个新的列表
            L = []
            # 持续递归解析，直到遇到右括号
            while token_list and token_list[0] != ')':
                L.append(read_from_tokens(token_list))
            
            # 如果没有找到匹配的右括号，说明语法错误
            if not token_list:
                raise SyntaxError("缺少右括号 ')'")
            
            # 弹出右括号
            token_list.pop(0) 
            return L
        elif token == ')':
            raise SyntaxError("遇到预料之外的右括号 ')'")
        else:
            # 如果不是括号，就是一个原子元素（字符串）
            return token

    try:
        return read_from_tokens(tokens)
    except (SyntaxError, IndexError) as e:
        print(f"解析 S-表达式时出错: {e}", file=sys.stderr)
        print(f"出错的字符串: '{s}'", file=sys.stderr)
        return None


def extract_goal_content(text: str) -> str | None:
    """
    从 BDDL 文件内容中提取 (:goal ...) 块内的 S-表达式字符串。
    这个函数会处理括号的配对，确保提取完整。
    """
    try:
        # 1. 定位 ':goal' 标记
        goal_marker = '(:goal'
        start_index = text.find(goal_marker)
        if start_index == -1:
            return None

        # 2. 从标记后找到第一个 '('，这是 goal 表达式的开始
        first_paren_index = text.find('(', start_index + len(goal_marker))
        if first_paren_index == -1:
            return None

        # 3. 计算括号层级来找到匹配的 ')'
        open_parens = 1
        i = first_paren_index + 1
        while i < len(text) and open_parens > 0:
            if text[i] == '(':
                open_parens += 1
            elif text[i] == ')':
                open_parens -= 1
            i += 1
        
        if open_parens == 0:
            # 如果 open_parens 归零，说明在 i-1 位置找到了匹配的括号
            last_paren_index = i
            return text[first_paren_index:last_paren_index]
        else:
            # 如果循环结束时 open_parens 仍大于 0，说明括号不匹配
            return None
    except Exception:
        return None


def main():
    """
    主函数，执行遍历、提取和解析的整个流程。
    """
    # ========================== 设置 ==========================
    # 请将这里替换成你的实际根目录路径
    ROOT_DIRECTORY = 'omnigibson/shengyin/bddl_gen2_for_hwh2/data/data-0806'
    # ==========================================================

    if not os.path.isdir(ROOT_DIRECTORY):
        print(f"错误：目录 '{ROOT_DIRECTORY}' 不存在。请检查路径是否正确。")
        return

    all_goals_list = []
    print(f"正在遍历目录: {ROOT_DIRECTORY}...")

    # 使用 os.walk 递归遍历所有子目录和文件
    for dirpath, _, filenames in os.walk(ROOT_DIRECTORY):
        for filename in filenames:
            # 检查文件名是否符合 'problem' 开头和 '.bddl' 结尾的模式
            if filename.startswith('problem') and filename.endswith('.bddl'):
                file_path = os.path.join(dirpath, filename)
                
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()

                    # 1. 提取 goal 块的原始内容
                    goal_string = extract_goal_content(content)

                    if goal_string:
                        # 2. 解析提取出的字符串
                        parsed_goal = parse_s_expression(goal_string)
                        
                        # 3. BDDL 的 goal 通常是 (and ...) 结构
                        # 我们需要的是 'and' 后面的条件列表
                        if parsed_goal and isinstance(parsed_goal, list) and parsed_goal[0].lower() == 'and':
                            # 提取 'and' 之后的所有条件
                            conditions = parsed_goal[1:]
                            all_goals_list.extend(conditions)
                            # 如果你希望每个文件的 goal 是一个独立的子列表，可以用下面这行代替
                            # all_goals_list.append(conditions)
                        elif parsed_goal:
                            # 如果 goal 不是 (and ...) 结构，直接添加
                            all_goals_list.append(parsed_goal)

                except Exception as e:
                    print(f"处理文件 {file_path} 时出错: {e}")

    print("\n✅ 提取完成！")
    print("所有 Goal 的组合列表如下:")
    # 打印最终结果
    # 为了美观，可以一个元素一行地打印
    for goal in all_goals_list:
        print(goal)

# 运行主函数
if __name__ == '__main__':
    main()