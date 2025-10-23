import json
import os
import re

# --- 配置区域 ---
# 请根据您的实际情况修改以下路径和密钥

# BDDL 文件根目录路径
# 例如: "data/bddl_files/bddl/activity_definitions"
# 或者，如果 "bddl" 文件夹与脚本在同一目录下，则可以是 "bddl/bddl/activity_definitions"
BDDL_BASE_PATH = "bddl/bddl/activity_definitions"  # TODO: 用户需要确认或修改此路径

# 您的 OpenAI API 密钥


# --- OpenAI API 调用函数 (占位符) ---
def get_new_task_description_from_openai(full_description: str) -> str | None:
    """
    调用 OpenAI API (例如 gpt-4o) 将 full_description 总结为
    一个不超过10个词的新任务描述 (人物概述)。

    !!! 警告: 这是一个占位符实现 !!!
    !!! 用户需要使用 OpenAI Python 库来实现真实的 API 调用 !!!
    """
    if not OPENAI_API_KEY or OPENAI_API_KEY == "YOUR_OPENAI_API_KEY":
        print("错误: OpenAI API 密钥未设置。请在脚本中设置 OPENAI_API_KEY。")
        print("将返回一个临时的占位符描述。")
        # 返回一个非常基础的占位符，以便脚本可以运行，但结果不是 GPT 生成的
        return f"Summarized: {' '.join(full_description.split()[:5])}..."

    # 真实场景下的 OpenAI API 调用示例 (需要安装 openai 库: pip install openai)
    from openai import OpenAI
    try:
        client = OpenAI(base_url="http://152.53.53.64:3000/v1", api_key=OPENAI_API_KEY)
        response = client.chat.completions.create(
            model="gpt-4o", # 或者其他您希望使用的模型
            messages=[
                {"role": "system", "content": "You are an assistant that summarizes task descriptions into a concise persona description of no more than 10 words. For example, old description is 'go to room storage room 0 pick up carton.n.02 from the shelf in the storage room 0 go to room closet 0 open carton.n.02 pick up white turnip.n.02 from the carton.n.02 go to room television room 0 put white turnip.n.02 on top of the coffee table in the television room 0'. The new description should be 'open carton and move turnip to top of the coffee table'."},
                {"role": "user", "content": f"Summarize the following task into a persona description (max 15 words): '{full_description}'. You should only return the new description, and no other text."},
            ],
            max_tokens=20 # 限制输出长度
        )
        new_description = response.choices[0].message.content.strip()
        print(f"OpenAI API 调用成功")
        print(f"old descrpition is {full_description}")
        print(f"new description is {new_description}")
        return new_description
    except Exception as e:
        print(f"OpenAI API 调用失败: {e}")
        return None

    # # 当前的临时占位符实现：
    # print(f"注意: 正在使用占位符 OpenAI 函数。请替换为真实的 API 调用。")
    # words = full_description.split()
    # if len(words) > 10:
    #     # 一个非常简单的缩短方法，您应该用真实的 OpenAI 调用替换它
    #     return " ".join(words[:7]) + "..."
    # return full_description


def get_full_task_description_from_bddl(task_name: str) -> str | None:
    """
    从指定任务的 BDDL 文件中读取并提取完整的任务描述。
    """
    # 确保 BDDL_BASE_PATH 和 task_name 结合能形成正确的路径
    # task_name 可能来自 JSONL 文件，例如 "pick_pill_bathroom_to_biology"
    # 路径格式: bddl/bddl/activity_definitions/{task_name}/problem0.bddl
    bddl_file_path = os.path.join(BDDL_BASE_PATH, task_name, "problem0.bddl")

    if not os.path.exists(bddl_file_path):
        print(f"警告: BDDL 文件未找到: {bddl_file_path}")
        return None
    
    try:
        with open(bddl_file_path, 'r', encoding='utf-8') as f:
            first_line = f.readline().strip()
        
        # BDDL 文件第一行格式: (define (problem task_description_with_underscores) ... )
        # 我们需要提取 "task_description_with_underscores"
        match = re.search(r'\(define \(problem\s+([^\s\)]+)\)', first_line)
        if match:
            description_with_underscores = match.group(1)
            # 将下划线替换为空格得到完整描述
            full_description = description_with_underscores.replace('_', ' ')
            return full_description
        else:
            print(f"警告: 无法从 BDDL 文件解析任务描述: {bddl_file_path}")
            print(f"文件第一行内容: {first_line}")
            return None
    except Exception as e:
        print(f"读取或解析 BDDL 文件 {bddl_file_path} 时出错: {e}")
        return None

def process_line_data(line_data: dict) -> dict:
    """
    处理从输入文件读取的单行 JSON 数据。
    """
    task_name = line_data.get("task_name")
    plan_prompt = line_data.get("plan_prompt")

    if not task_name or not plan_prompt:
        print(f"警告: 行数据中缺少 'task_name' 或 'plan_prompt': {line_data}")
        return line_data # 如果关键数据缺失，返回原始数据

    # 1. 从 BDDL 获取完整任务描述
    full_bddl_description = get_full_task_description_from_bddl(task_name)
    if not full_bddl_description:
        print(f"由于 BDDL 问题，跳过对任务 '{task_name}' 的修改。")
        return line_data # 返回原始数据

    # 2. 通过 OpenAI (占位符) 获取新的任务描述
    new_task_description = get_new_task_description_from_openai(full_bddl_description)
    if not new_task_description:
        print(f"未能为任务 '{task_name}' 生成新描述，跳过修改。")
        return line_data # 返回原始数据

    # 3. 在 plan_prompt 中替换旧的任务描述
    # 原始结构: "... You are tasked with: {old_description}. The objects in the scenarios include: ..."
    # 目标结构: "... You are tasked with: {new_description}. The objects in the scenarios include: ..."

    prefix = "You are tasked with: "
    # old_description 位于 prefix 和 ". The objects in the scenarios include:" 之间
    
    start_prefix_idx = plan_prompt.find(prefix)
    if start_prefix_idx == -1:
        print(f"警告: 在任务 '{task_name}' 的 plan_prompt 中未找到前缀 '{prefix}'。Prompt 保持不变。")
        return line_data

    # 定位 old_description 的开始位置 (在 prefix 之后)
    start_old_desc_idx = start_prefix_idx + len(prefix)
    
    # 定位 old_description 的结束位置 (在 ". The objects..." 之前)
    # old_description 以句号结束
    end_marker = ". The objects in the scenarios include:"
    end_old_desc_idx = plan_prompt.find(end_marker, start_old_desc_idx)

    if end_old_desc_idx == -1:
        # 尝试备用方案：如果完整的 end_marker 未找到，可能 prompt 结构略有不同
        # 查找 prefix 后的第一个句号，并检查其后是否是预期的文本片段
        first_period_after_prefix = plan_prompt.find(".", start_old_desc_idx)
        if first_period_after_prefix != -1 and plan_prompt[first_period_after_prefix:].startswith(end_marker):
            end_old_desc_idx = first_period_after_prefix
        else:
            print(f"警告: 无法在任务 '{task_name}' 的 plan_prompt 中可靠地找到 old_task_description 的结束位置。Prompt 保持不变。")
            # 打印相关片段以便调试
            # print(f"Prompt 片段: {plan_prompt[start_prefix_idx : start_prefix_idx + 150]}") 
            return line_data

    # 提取替换前后的部分
    part_before_description = plan_prompt[:start_old_desc_idx]
    part_after_description = plan_prompt[end_old_desc_idx:] # 这部分包含句号和后续文本

    # 清理 new_task_description，确保它不以句号结尾 (因为句号已在 part_after_description 中)
    clean_new_task_description = new_task_description.strip()
    if clean_new_task_description.endswith('.'):
        clean_new_task_description = clean_new_task_description[:-1].strip()
    
    modified_plan_prompt = f"{part_before_description}{clean_new_task_description}{part_after_description}"
    
    line_data["plan_prompt"] = modified_plan_prompt
    return line_data

def main(input_file_path: str, output_file_path: str):
    processed_lines_count = 0
    # 检查输出文件是否存在，以实现断点续传
    if os.path.exists(output_file_path):
        try:
            with open(output_file_path, 'r', encoding='utf-8') as f_out_check:
                processed_lines_count = sum(1 for _ in f_out_check)
            print(f"继续处理: 在 {output_file_path} 中发现 {processed_lines_count} 条已处理的行。")
        except Exception as e:
            print(f"检查输出文件时出错: {e}。将从头开始处理。")
            processed_lines_count = 0


    try:
        with open(input_file_path, 'r', encoding='utf-8') as f_in, \
             open(output_file_path, 'a', encoding='utf-8') as f_out: # 以追加模式打开输出文件

            for i, line_content in enumerate(f_in):
                if i < processed_lines_count:
                    continue # 跳过已处理的行

                try:
                    # 解析原始行数据
                    current_line_data = json.loads(line_content.strip())
                except json.JSONDecodeError:
                    print(f"警告: 跳过格式错误的 JSON 行 {i+1}: {line_content.strip()}")
                    #可以选择将原始错误行写入输出，以保持行号对应，或写入单独的错误日志
                    f_out.write(line_content) 
                    f_out.flush()
                    continue
                
                # 处理数据
                modified_line_data = process_line_data(current_line_data)
                
                # 写入修改后的数据
                f_out.write(json.dumps(modified_line_data) + '\n')
                f_out.flush() # 确保立即写入磁盘

                # 定期打印进度
                if (i + 1 - processed_lines_count) % 10 == 0: # 每处理10条新数据打印一次
                     print(f"已处理 {i+1} 总行数 (本次运行处理 {i+1-processed_lines_count} 行)...")

    except FileNotFoundError:
        print(f"错误: 输入文件未找到: {input_file_path}")
        print("请确保 input_file_path 和 BDDL_BASE_PATH 配置正确。")
    except Exception as e:
        print(f"处理过程中发生意外错误: {e}")

    print(f"处理完成。输出已写入 {output_file_path}")

if __name__ == "__main__":
    # --- 用户需要配置的输入/输出文件路径 ---
    # 从用户提供的上下文获取输入文件名
    # /home/zxlei/data/embodied/planner_bench/train_20250513_141140.jsonl
    input_jsonl_file = "train_20250513_141140.jsonl" # TODO: 用户确认此路径
    
    # 构建输出文件名，例如在原文件名后添加 "_modified"
    input_dir = os.path.dirname(input_jsonl_file)
    input_filename_base, input_filename_ext = os.path.splitext(os.path.basename(input_jsonl_file))
    output_jsonl_file = os.path.join(input_dir, f"{input_filename_base}_modified{input_filename_ext}")

    print(f"输入文件: {input_jsonl_file}")
    print(f"输出文件: {output_jsonl_file}")
    print(f"BDDL 基础路径: {BDDL_BASE_PATH}")
    print("重要提示: 请确保已在脚本中配置 OpenAI API 密钥，并根据需要修改 BDDL_BASE_PATH。")
    print("同时，您需要实现 get_new_task_description_from_openai 函数中的 OpenAI API 调用逻辑。")
    
    main(input_jsonl_file, output_jsonl_file)
