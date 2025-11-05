from openai import OpenAI
import social_character_gen
import read_files
import os    
import re
import random
import json
from dotenv import load_dotenv
from role_playing import RolePlaying_2Users
from colorama import Fore
from src.utils.config_loader import config
api_key = config['task_generation']['api_key']
base_url = config['task_generation']['base_url']
model = config['task_generation']['model']
def command_generation(social_description,amount,rooms,room_size,command_difficulty):
    client = OpenAI(api_key=api_key, base_url=base_url)

    #从文件中采样10个togglable_object
    togglable_object = read_files.read_lines_to_array('src/bddl_gen/data/objects_sorted_by_states/toggleable_object.txt')
    random.seed()
    random.shuffle(togglable_object)
    togglable_object = togglable_object[0:10]
    print(togglable_object)

    #从文件中采样20个openable_object
    openable_object = read_files.read_lines_to_array('src/bddl_gen/data/objects_sorted_by_states/openable_object.txt')
    random.seed()
    random.shuffle(openable_object)
    openable_object = openable_object[0:20]
    print(openable_object)

    #从文件中采样10个cookable_object
    cookable_object = read_files.read_lines_to_array('src/bddl_gen/data/objects_sorted_by_states/cookable_object.txt')
    random.seed()
    random.shuffle(cookable_object)
    cookable_object = cookable_object[0:10]
    print(cookable_object)

    #从文件中采样20个fillable_object
    fillable_object = read_files.read_lines_to_array('src/bddl_gen/data/objects_sorted_by_states/fillable_object.txt')
    random.seed()
    random.shuffle(fillable_object)
    fillable_object = fillable_object[0:20]
    print(fillable_object)

    #从文件中采样6个heatsource_object
    heatsource_object = read_files.read_lines_to_array('src/bddl_gen/data/objects_sorted_by_states/heatsource.txt')
    random.seed()
    random.shuffle(heatsource_object)
    heatsource_object = heatsource_object[0:6]
    print(heatsource_object)

    #从文件中采样2个coldsource_object
    coldsource_object = read_files.read_lines_to_array('src/bddl_gen/data/objects_sorted_by_states/coldsource.txt')
    random.seed()
    random.shuffle(coldsource_object)
    coldsource_object = coldsource_object[0:2]
    print(coldsource_object)

    #从文件中采样10个freezable_object
    freezable_object = read_files.read_lines_to_array('src/bddl_gen/data/objects_sorted_by_states/freezable_object.txt')
    random.seed()
    random.shuffle(freezable_object)
    freezable_object = freezable_object[0:10]
    print(freezable_object)

    print(room_size)
    if command_difficulty == 1:
        #TODO
        extra_requirment = "For each command, you should make sure that there are mutiples (2-3) categories of objects to be operated on different places in different rooms. Not make all operating in the same room. "
    else:
        extra_requirment = ''
    completion = client.chat.completions.create(
        model="gpt-4o",
        messages = [
            {
            "role": "user",
            "content": [
                    {"type": "text", "text": f'''{social_description} You need to generate reasonable commands for the social members.The requirements for the commands are as follows:
1. The command must be broken down into the robot's subtasks as follows: pick up something, place something, open something, close something, toggle on appliance, toggle off appliance, go to room, put something on top of something, put something inside something, heat something, cook something, frozen something.
2. The objects and locations involved in the command should not exceed eight.
3. The commands should cover different needs of the four family members.
4. The completion criteria of the commands must be clear, such as ensuring an object is placed in a specific location or on something. Avoid commands where the result is unclear.
5. The commands should not exceed the mobile robot's capabilities. The robot can only: Move to a nearby location around an object, Turn by a specific angle, Pick up an object, Place an object, Move forward to a specific location, open something, close something, toggle on appliance, toggle off appliance, go to room, put something on top of something, put something inside something, heat something, cook something, frozen something. The robot can't hang something or sort something or water plants, it can only perform simple tasks.
6. The scene contains rooms:{rooms}, commands can't exceed the range.
7. The size of the rooms:{room_size}, consider the size of the room when generating items in each room.
8. The rooms which objects are in should be specific in the commands, don't use phrases like "the same room", use the specific roomname like "in the bedroom_1". The rooms which objects in should be appropriate, for example, the desk_phone should not be in the bathroom_0, the bed should not be in the kitchen_0, the refrigerator should not be in the living room_0.
9. The commands should be concise for example, use "Please bring the juice pitcher from the counter in the kitchen_0 to the garden table in the garden_0 " instead of "Head to the kitchen_0, grab the juice pitcher from the counter, and bring it to the garden_0. Place it on the garden table."
10. The robot is not that tall, like 0.5m to 1.5m. So the command can't exceed its height limit
11. If the robot needs to pick up something, the commmand should specify the object and the location, like "pick up the juice pitcher from the counter in the kitchen_0" instead of "pick up the juice pitcher".
12. Make sure the commands are diverse enough, not all the commands are similar. For example, don't make all the commands like "pick up something and put it on something".
13. If you need to use togglable objects, you can use items in the list :{togglable_object}
14. If you need to use openable objects, you can use items in the list :{openable_object}
15. If you need to use cookable objects, you can use items in the list :{cookable_object}
16. If you need to use heatsource objects, you can use items in the list :{heatsource_object}
17. If you need to use coldsource objects, you can use items in the list :{coldsource_object}, Please mark: cooler can't be toggled on, it's always toggled on. So to freeze something you just need to put it in the cooler.
18. If you need to use freezable objects, you can use items in the list :{freezable_object}
19. If you need to use put something inside an object, you can use items in the list :{fillable_object} to be the container.
20. The pick and place task don't need to consider whether the object is openable, toggleable, cookable, freezable or not. The robot can pick up and place any object. But the openable, toggleable, cookable, freezable task need to consider whether the object is openable, toggleable, cookable, freezable or not.
21. If you want to cook or freeze something, the scene needs to have a heatsource or a coldsource objects.
22. Each command needs to be diverse, involving operating different objects and rooms. But don't force to be diverse, for example, cook something or freeze something is not reasonable in a school gym scene while in a kitchen is reasonable to cook or freeze something.The commands must be complex enough.

Please generate {amount} command. And make sure they are all reasonable fot the social members.{extra_requirment}
Output in the format like:
1.Analyse the social_description, find out what kind of command is needed and reasonable. And what kinds of objects are reasonable to appear in this scene. Then how can these objects be operated by humans and robots. For example, we won't heat food in the bathroom, instead we will heat food in the kitchen. We won't put a bed in the kitchen, instead we will put a bed in the bedroom. We won't put a refrigerator in the living room, instead we will put a refrigerator in the kitchen. We won't put a desk_phone in the bathroom, instead we will put a desk_phone in the bedroom. 
2.Analyse the objects we may need for the commands, where they are supposed to be, and how to operate them. 
3.Generate the command, and make sure the command is reasonable and diverse.
ALL commands MUST be included by [[]] altogether, which will make it easier to parse. For example, [[
command1
command2
command3]]. all the commands you generated must be included in a single [[]].You should give a list of command which means no any extra punctuations and extra indexs.'''},
            ],
            }
        ],
        temperature=0.7,
        max_tokens=10000,
    )
    
    response = completion.choices[0].message.content
    print(response)
    print(read_files.extract_commands(response))
    return(read_files.extract_commands(response))


def command_gen(sceneid,command_num,command_difficulty):
    return command_generation(social_character_gen.social_character_generation(read_files.read_lines_to_array("src/bddl_gen/extracted_descriptions.txt")[sceneid-1],read_files.read_lines_to_array("extracted_rooms.txt")[sceneid-1]),command_num,read_files.read_lines_to_array("src/bddl_gen/extracted_rooms.txt")[sceneid-1],room_size=read_files.extract_room_areas(read_files.read_json_file('src/bddl_gen/area.json'))[sceneid-1],command_difficulty=command_difficulty)

def command_extract_from_dialogue(dialogue_text):
    """
    从对话文本中提取被[]框住的命令，并去除长度小于20的命令。

    参数:
        dialogue_text (str): 包含命令的对话文本。

    返回:
        list: 提取的命令列表。
    """
    # 匹配被[]包裹的内容，支持跨行
    commands = re.findall(r'\[(.*?)\]', dialogue_text, re.DOTALL)
    # 去除首尾空白并过滤空字符串和长度小于20的命令
    commands = [cmd.strip() for cmd in commands if cmd.strip() and len(cmd.strip()) >= 20]
    return commands

def command_generation_from_dialogue(sceneid, chat_turn_limit=6, model=None):


    task_prompt = '''You need to play one role of two users who is interacting with each other. One is User1 , the other one is User2. They have a robot. The robot can only: Move to a nearby location around an object, Turn by a specific angle, Pick up an object, Place an object, Move forward to a specific location, open something, close something, toggle on appliance, toggle off appliance, go to room, put something on top of something, put something inside something, heat something, cook something, frozen something. The robot can't hang something or sort something or water plants, it can only perform simple tasks.Both of them can give commands to the robot. Although the robot can only perform single tasks, it can perform combined tasks like "pick the carrot from the fridge and cook it " or "collect all the balls on the floor and put them into the box". You don't need to break one combined task into several simple tasks. You are naturally interacting with each other, doing your own things, and during the conversation, you will give robot some commands naturally. The command should be in the form of [Robot, get the carrot out of the fridge and cook it]. And you should cooperate with the other user to create longer commands, which means don't give command like"Pick up the corrat" which is too easy, you need to communicate with the other user to create longer commands like "Robot, get the carrot out of the fridge and cook it" or give commands that is not so specific but also combined task like "prepare the raw material of today's dinner on the table."'''
    
    social_description = social_character_gen.social_character_generation_in_detail(read_files.read_lines_to_array("extracted_rooms.txt")[sceneid - 1])

    role_names = read_files.extract_character_names(social_description)
    if len(role_names) < 2:
        print(
            Fore.RED
            + "Error: The social description must contain at least two characters."
        )
        return
    role_play_session = RolePlaying_2Users(
        social_description=social_description,
        user1_role_name=role_names[0],
        user1_agent_kwargs=dict(model=model),
        user2_role_name=role_names[1],
        user2_agent_kwargs=dict(model=model),
        task_prompt=task_prompt,
        with_task_specify=True,
        task_specify_agent_kwargs=dict(model=model),
    )
    print(
        Fore.GREEN
        + f"{role_names[0]} sys message:\n{role_play_session.user1_sys_msg}\n"
    )
    print(
        Fore.BLUE + f"{role_names[1]} sys message:\n{role_play_session.user2_sys_msg}\n"
    )
    print(Fore.YELLOW + f"Original task prompt:\n{task_prompt}\n")

    n = 0
    input_msg = role_play_session.init_chat()
    dialogue_text = ""
    while n < chat_turn_limit:
        n += 1
        user1_response, user2_response = role_play_session.step(input_msg)

        if user1_response.terminated:
            print(
                Fore.GREEN
                + (
                    "AI Assistant terminated. Reason: "
                    f"{user1_response.info['termination_reasons']}."
                )
            )
            break
        if user2_response.terminated:
            print(
                Fore.GREEN
                + (
                    "AI User terminated. "
                    f"Reason: {user2_response.info['termination_reasons']}."
                )
            )
            break

        print(Fore.BLUE + f"User2:\n\n{user2_response.msg.content}\n")
        print(Fore.GREEN + "User1:\n\n" + f"{user1_response.msg.content}\n")

        dialogue_text += user2_response.msg.content + "\n"
        dialogue_text += user1_response.msg.content + "\n"

        if "CAMEL_TASK_DONE" in user2_response.msg.content:
            break

        input_msg = user1_response.msg

    commands = command_extract_from_dialogue(dialogue_text)
    print(commands)
    feasible_commands = []
    # 检查每个命令的可行性，进行初步过滤
    for command in commands:
        if command_feasibility_check(command):
            print(Fore.GREEN + f"Command is feasible: {command}")
            feasible_commands.append(command)
        else:
            print(Fore.RED + f"Command is NOT feasible: {command}")
    print(feasible_commands)
    #对命令进行具体化
    specified_commands = []
    original_commands_and_specified_commands = []
    for command in feasible_commands:
        specified_command = specify_command(command, social_description, sceneid)
        if specified_command:
            specified_commands.append(specified_command)
            original_commands_and_specified_commands.append((command, specified_command))
        else:
            print(Fore.RED + f"Failed to specify command: {command}")
    print(original_commands_and_specified_commands)
    return original_commands_and_specified_commands

def command_feasibility_check(command):
    client = OpenAI(api_key=api_key, base_url=base_url)
    completion = client.chat.completions.create(
        model="gpt-4o",
        messages = [
            {
            "role": "user",
            "content": [
                    {"type": "text", "text": f'''This is a command for a robot :{command}.Please check whether this command is feasible. The command should not exceed the mobile robot's capabilities. The robot can only: Move to a nearby location around an object, Turn by a specific angle, Pick up an object, Place an object, Move forward to a specific location, open something, close something, toggle on appliance, toggle off appliance, go to room, put something on top of something, put something inside something, heat something, cook something, frozen something. The robot can't hang something or sort something or water plants or operate liquid, it can only perform simple tasks. For example, commands like"Robot, grab the vegetables from the fridge and start cooking them in the pot","Robot, go to the storage room, collect the vase and place it in the dining room on the main table" are feasible; commands like"Robot, once the soup is cooked, serve it in the bowls and place them on the kitchen counter" are not feasible because robot can't operate liquid. If the command is feasible, please output [[FEASIBLE]]. If the command is not feasible, please output [[NOTFEASIBLE]].'''},
            ],
            }
        ],
        temperature=0.7,
        max_tokens=10000,
    )
    
    response = completion.choices[0].message.content

    #检测恢复里是否有[[FEASIBLE]]或[[NOTFEASIBLE]]
    if "[[FEASIBLE]]" in response:
        return True
    elif "[[NOTFEASIBLE]]" in response:
        return False
    else:
        print("Error: The response does not contain [[FEASIBLE]] or [[NOTFEASIBLE]].")
        return None
    
def specify_command(command,social_description, sceneid):
    client = OpenAI(api_key=api_key, base_url=base_url)
    rooms = read_files.read_lines_to_array("extracted_rooms.txt")[sceneid - 1]
    # 获取房间大小
    room_size = read_files.extract_room_areas(read_files.read_json_file('area.json'))[sceneid - 1]

    #从文件中采样10个togglable_object
    togglable_object = read_files.read_lines_to_array('src/bddl_gen/data/objects_sorted_by_states/toggleable_object.txt')
    random.seed()
    random.shuffle(togglable_object)
    togglable_object = togglable_object[0:10]
    #print(togglable_object)

    #从文件中采样20个openable_object
    openable_object = read_files.read_lines_to_array('src/bddl_gen/data/objects_sorted_by_states/openable_object.txt')
    random.seed()
    random.shuffle(openable_object)
    openable_object = openable_object[0:20]
    #print(openable_object)

    #从文件中采样10个cookable_object
    cookable_object = read_files.read_lines_to_array('src/bddl_gen/data/objects_sorted_by_states/cookable_object.txt')
    random.seed()
    random.shuffle(cookable_object)
    cookable_object = cookable_object[0:30]
    #print(cookable_object)

    #从文件中采样20个fillable_object
    fillable_object = read_files.read_lines_to_array('src/bddl_gen/data/objects_sorted_by_states/fillable_object.txt')
    random.seed()
    random.shuffle(fillable_object)
    fillable_object = fillable_object[0:20]
    #print(fillable_object)

    #从文件中采样6个heatsource_object
    heatsource_object = read_files.read_lines_to_array('src/bddl_gen/data/objects_sorted_by_states/heatsource.txt')
    random.seed()
    random.shuffle(heatsource_object)
    heatsource_object = heatsource_object[0:6]
    #print(heatsource_object)

    #从文件中采样2个coldsource_object
    coldsource_object = read_files.read_lines_to_array('src/bddl_gen/data/objects_sorted_by_states/coldsource.txt')
    random.seed()
    random.shuffle(coldsource_object)
    coldsource_object = coldsource_object[0:2]
    #print(coldsource_object)

    #从文件中采样10个freezable_object
    freezable_object = read_files.read_lines_to_array('src/bddl_gen/data/objects_sorted_by_states/freezable_object.txt')
    random.seed()
    random.shuffle(freezable_object)
    freezable_object = freezable_object[0:10]
    #print(freezable_object)   
     
    completion = client.chat.completions.create(
        model="gpt-4o",
        messages = [
            {
            "role": "user",
            "content": [
                    {"type": "text", "text": f'''This is a original command for a robot :{command}. It is given by a human, but for a robot to execute you need to specify it, here is the requirement of a specified command:
                    1. The command must be broken down into the robot's subtasks as follows: pick up something, place something, open something, close something, toggle on appliance, toggle off appliance, go to room, put something on top of something, put something inside something, heat something, cook something, frozen something.
                    2. The objects and locations involved in the command should not exceed eight.
                    3. The completion criteria of the commands must be clear, such as ensuring an object is placed in a specific location or on something. Avoid commands where the result is unclear.
                    4. The commands should not exceed the mobile robot's capabilities. The robot can only: Move to a nearby location around an object, Turn by a specific angle, Pick up an object, Place an object, Move forward to a specific location, open something, close something, toggle on appliance, toggle off appliance, go to room, put something on top of something, put something inside something, heat something, cook something, frozen something. The robot can't hang something or sort something or water plants, it can only perform simple tasks.
                    5. The scene contains rooms:{rooms}, commands can't exceed the range.
                    6. The size of the rooms:{room_size}, consider the size of the room when generating items in each room.
                    7. The rooms which objects are in should be specific in the commands, don't use phrases like "the same room", use the specific roomname like "in the bedroom_1". The rooms which objects in should be appropriate, for example, the desk_phone should not be in the bathroom_0, the bed should not be in the kitchen_0, the refrigerator should not be in the living room_0.
                    8. The commands should be concise for example, use "Please bring the juice pitcher from the counter in the kitchen_0 to the garden table in the garden_0 " instead of "Head to the kitchen_0, grab the juice pitcher from the counter, and bring it to the garden_0. Place it on the garden table."
                    9. If the robot needs to pick up something, the commmand should specify the object and the location, like "pick up the juice pitcher from the counter in the kitchen_0" instead of "pick up the juice pitcher".
                    10. If you need to use togglable objects, you can use items in the list :{togglable_object}
                    11. If you need to use openable objects, you can use items in the list :{openable_object}
                    12. If you need to use cookable objects, you MUST use items in the list :{cookable_object}, even though the original command don't mention exactly the same item, just make sure the items are all used to be cooked.
                    13. If you need to use heatsource objects, you MUST use items in the list :{heatsource_object}, even though the original command don't mention exactly the same item, just make sure the items are all used to cook.
                    14. If you need to use coldsource objects, you MUST use items in the list :{coldsource_object}, Please mark: cooler can't be toggled on, it's always toggled on. So to freeze something you just need to put it in the cooler. replace all the coldsource objects in the original command with cooler.
                    15. If you need to use freezable objects, you can use items in the list :{freezable_object}
                    16. If you need to use put something inside an object, you can use items in the list :{fillable_object} to be the container.
                    17. The pick and place task don't need to consider whether the object is openable, toggleable, cookable, freezable or not. The robot can pick up and place any object. But the openable, toggleable, cookable, freezable task need to consider whether the object is openable, toggleable, cookable, freezable or not.
                    18. If you want to cook or freeze something, the scene needs to have a heatsource or a coldsource objects.
Please generate a sepcified command from the command:{command}, the social description of the scene is:{social_description}.
Output in the format like:
1.Analyse the social_description, find out what kinds of objects are reasonable to appear in this scene and be involved by this command. And where they should be at.
2.Generate the specified command.
The command should be combined into one single sentence, don't break it into several sentences.
The command MUST be included by [[]] , which will make it easier to parse. For example, [[
command]].'''},
            ],
            }
        ],
        temperature=0.7,
        max_tokens=10000,
    )
    
    response = completion.choices[0].message.content
    print(response)
    # 检查是否包含[[和]]
    if "[[" in response and "]]" in response:
        # 提取[[和]]之间的内容
        specified_command = re.search(r'\[\[(.*?)\]\]', response, re.DOTALL)
        if specified_command:
            specified_command = specified_command.group(1).strip()
            print(f'original command: {command}\n')
            print(f"Specified command: {specified_command}")
            return specified_command
        else:
            print("Error: No command found in the response.")
            return None
    else:
        print("Error: The response does not contain [[ and ]].")
        return None