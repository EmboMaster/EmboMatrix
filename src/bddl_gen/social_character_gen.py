from openai import OpenAI
import os
import re
import sys
current_file_path = os.path.abspath(__file__)
current_dir = os.path.dirname(current_file_path)
src_dir = os.path.dirname(current_dir)
if src_dir not in sys.path:
    sys.path.append(src_dir)
from prompts.templates import SOCIAL_CHARACTER_GEN_PROMPT_IN_DETAIL,SOCIAL_CHARACTER_GEN_PROMPT
from src.utils.config_loader import config
api_key = config['task_generation']['api_key']
base_url = config['task_generation']['base_url']
model = config['task_generation']['model']
def social_character_generation(scene_description,room_description):
    client = OpenAI(api_key=api_key, base_url=base_url)
    print(scene_description)
    print(room_description)

    completion = client.chat.completions.create(
        model=model,
        messages = [
            {
            "role": "user",
            "content": [
                    {"type": "text", "text": SOCIAL_CHARACTER_GEN_PROMPT.format(
                        scene_description=scene_description,
                        room_description=room_description
                    )}
            ],
            }
        ],
        temperature=0.7,
        max_tokens=10000,
    )
    
    response = completion.choices[0].message.content
    print(response)
    return response

def social_character_generation_in_detail(room_description):
    client = OpenAI(api_key=api_key, base_url=base_url)
    print(room_description)

    completion = client.chat.completions.create(
        model=model,
        messages = [
            {
            "role": "user",
            "content": [
                    {"type": "text", "text": SOCIAL_CHARACTER_GEN_PROMPT_IN_DETAIL.format(
                        room_description=room_description
                    )}
                
            ],
            }
        ],
        temperature=0.7,
        max_tokens=10000,
    )

    response = completion.choices[0].message.content
    # with open("command_data/social_description.txt", "a") as file:
    #     file.write(f"{response}\n")
    #     file.write("-" * 50 + "\n")
    print(f'scene name and room list:{room_description} \nscene description:{response}')

    check_room_coverage(room_description, response)

    return f'scene description:{response}'


def check_room_coverage(full_room_list_str: str, scene_description: str):
    # 提取冒号右边的房间名列表
    if ':' in full_room_list_str:
        room_list_str = full_room_list_str.split(":", 1)[1]
    else:
        room_list_str = full_room_list_str
    
    room_names = [r.strip() for r in room_list_str.split(",")]

    print("🔍 房间覆盖检查结果：\n")
    all_covered = True
    for room in room_names:
        # 构建一个更宽容的正则表达式，忽略下划线/数字/复数等微小差异
        pattern = re.compile(rf"\b{room.replace('_', '[ _]?')}\b", re.IGNORECASE)
        if pattern.search(scene_description):
            print(f"✅ 已提及房间：{room}")
        else:
            print(f"❌ 未提及房间：{room}")
            all_covered = False

    print("\n🌟 全覆盖" if all_covered else "\n⚠️ 存在遗漏，请考虑重新生成")
    return all_covered