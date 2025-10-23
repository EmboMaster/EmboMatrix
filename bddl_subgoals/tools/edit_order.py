#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import sys

def fix_agent_relation_in_line(line):
    """
    使用正则表达式，修正形如:
      (relationship ?object ?agent.something)
    变为:
      (relationship ?agent.something ?object)
    """
    pattern = re.compile(r"\(\s*([a-zA-Z_]+)\s+(\?[^\s)]+)\s+(\?agent[^\s)]+)\s*\)")
    fixed_line = pattern.sub(r"(\1 \3 \2)", line)
    return fixed_line

def fix_agent_relations_in_file(filepath):
    """
    打开一个 .bddl 文件，对其中所有行做修复，然后写回文件。
    """
    with open(filepath, "r", encoding="utf-8") as f:
        lines = f.readlines()

    # 逐行修复
    new_lines = [fix_agent_relation_in_line(line) for line in lines]

    # 判断是否发生了修改，如果没有则不需要写回
    # （当然你也可以直接写回）
    old_content = "".join(lines)
    # old_content = old_content.replace("agent.n.01 ", "agent.n.01_1 ")
    new_content = "".join(new_lines)
    new_content = new_content.replace("agent.n.01 ", "agent.n.01_1 ")

    if old_content != new_content:
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(new_content)
        print(f"Fixed file: {filepath}")
    else:
        print(f"No changes needed: {filepath}")

def main(directory):
    """
    批量处理目录下所有 .bddl 文件，对其进行修复。
    """
    if not os.path.isdir(directory):
        print(f"Error: {directory} is not a valid directory.")
        return

    # 遍历目录下的所有 .bddl 文件
    for root, dirs, files in os.walk(directory):
        for filename in files:
            print(filename)
            if filename.endswith('.bddl'):
                file_path = os.path.join(root, filename)
                print(f"正在处理文件: {file_path}")
                fix_agent_relations_in_file(file_path)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python fix_agent_relations.py <bddl_directory>")
        sys.exit(1)

    bddl_directory = sys.argv[1]
    main(bddl_directory)



# def process_directory(directory_path):
#     # 遍历目录中的所有文件和子目录
#     for root, dirs, files in os.walk(directory_path):
#         for filename in files:
#             print(filename)
#             if filename.endswith('.bddl'):
#                 file_path = os.path.join(root, filename)
#                 print(f"正在处理文件: {file_path}")
#                 modify_bddl_file(file_path)

# if __name__ == "__main__":
#     # 请修改为你的目标目录路径
#     directory_path = '/data/zxlei/embodied/embodied-bench/omnigibson/shengyin/new_results_experiment/0'
#     process_directory(directory_path)
