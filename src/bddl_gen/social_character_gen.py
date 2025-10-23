from openai import OpenAI
import os
import re
import sys
current_file_path = os.path.abspath(__file__)
current_dir = os.path.dirname(current_file_path)
src_dir = os.path.dirname(current_dir)
if src_dir not in sys.path:
    sys.path.append(src_dir)
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
                    {"type": "text", "text": f'''I need you to help me to generate a social description to a scene description, Here is an example. This is a description of a scene:"A multi-room office space featuring numerous cubicles, private offices, a conference hall, a meeting room, a lobby, and a copy room, equipped with various office furniture and equipment.". Generate a social description of this scene, such as:"In this scene, there is a family of four: a father, a mother, a 14-year-old girl, and an 8-year-old boy. They have a mobile robot (composed of a mobile platform and a simple gripper) as their household assistant. These five characters will give simple commands to the robot throughout the day." The social description should be related to the scene and there must be more than two humanbeings and only one robot to serve the humans.
                     Here is an other example:
                     scene description:A school scene featuring 7 rooms, including classrooms and corridors, with numerous desks, chairs, lockers, maps, and educational tools.
                     social description:In this scene, there is a dedicated teacher and a diligent school administrator, who work together to ensure the smooth operation of the school. They are assisted by a single educational robot that navigates through the classrooms and corridors. 
                     Here is a scene description:{scene_description}. This is the room list you can use :{room_description}, Please MARK:the generated social description can't include other rooms, only the rooms in the room list can be used to Generate the social description. Generate the social description.'''},
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
                    {"type": "text", "text": 
                f'''I need you to help me generate a social description for a simulated scene.

                These are the rooms in this scene: {room_description}.

                Your job is to write a detailed social description that:
                1. Describes at least two human characters (name in [], with age, gender, occupation, and vivid hobbies);
                2. Clearly assigns typical **daily activities** to **multiple rooms** in the list;
                3. Ensures that **every room** in the list appears with a relevant activity;
                4. Avoids generic sentences — be specific and creative (e.g. “they often fold towels in the bathroom while chatting”, “they prepare fruit platters in the kitchen and bring them to the dining room”);
                5. The humans can be friends, family, or coworkers — but must live/interact together in the space.

                Important:
                - **Do not include rooms not in the list**.
                - **Do not describe the robot.** Only the humans and their use of the rooms.
                - The robot exists and will be controlled later, but is not mentioned in this social description.

                Here's an example of the kind of description I want:

                "In this scene, [Wang Lei], a 40-year-old hotel manager who enjoys classical music and wine tasting, works alongside [Liu Fang], a 32-year-old front desk supervisor passionate about floral arrangement and interior decor. In the lobby, they prepare welcome baskets and arrange flowers for VIP guests. In the kitchen, they coordinate meal prep with the chef. The dining room is where they occasionally sample dishes and check table setups. The bathroom is used for quick grooming before events. The corridor serves as the path for moving supplies and decorations."

                Your goal is to create a similar detailed description for this room list: {room_description}. Remember: each room must appear. Make it vivid, grounded, and specific. Only output the description. Do not include any explanation or list.'''}

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