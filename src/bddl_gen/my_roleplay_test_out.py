from my_roleplay import Roleplay
from my_roleplay import generate_prompts_from_social_description
import read_files
import social_character_gen
from src.utils.config_loader import config
api_key = config['task_generation']['api_key']
base_url = config['task_generation']['base_url']
model = config['task_generation']['model']
if __name__ == "__main__":


    # 加载角色环境
    social_description = social_character_gen.social_character_generation_in_detail(
        read_files.read_lines_to_array("extracted_rooms.txt")[39]  
    )

    # 提取角色名
    agent_names = read_files.extract_character_names(social_description)
    if len(agent_names) < 2:
        raise ValueError("The social description must contain at least two characters.")

    agent_names.append("Robot")  # 添加机器人角色

    # 生成系统 prompt
    system_prompts = generate_prompts_from_social_description(social_description, agent_names)

    # 初始化 Roleplay 系统
    roleplay = Roleplay(
        agent_names=agent_names,
        system_prompts=system_prompts,
        model="gpt-4o",
        api_key=api_key,
        base_url=base_url,
        max_turns=30
    )

    # 启动对话
    roleplay.run(
        initial_message="Hello!",
        initial_speaker_index=0
    )

    # 打印最终对话记录
    print("\n==== 多智能体对话日志 ====\n")
    print(roleplay.get_transcript())