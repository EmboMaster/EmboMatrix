from openai import OpenAI
import social_character_gen
import read_files
import os    
import random
from src.utils.config_loader import config
api_key = config['task_generation']['api_key']
base_url = config['task_generation']['base_url']
model = config['task_generation']['model']
def command_generation(social_description,amount,rooms,room_size,command_difficulty):

    client = OpenAI(api_key=api_key, base_url=base_url)

    #从文件中采样10个togglable_object
    togglable_object = read_files.read_lines_to_array('src/bddl_gen/data/objects_sorted_by_states/togglable_object.txt')
    random.seed()
    random.shuffle(togglable_object)
    togglable_object = togglable_object[0:10]
    print(togglable_object)

    #从文件中采样10个openable_object
    openable_object = read_files.read_lines_to_array('src/bddl_gen/data/objects_sorted_by_states/openable_object.txt')
    random.seed()
    random.shuffle(openable_object)
    openable_object = openable_object[0:10]
    print(openable_object)

    #从文件中采样10个cookable_object
    cookable_object = read_files.read_lines_to_array('src/bddl_gen/data/objects_sorted_by_states/cookable_object.txt')
    random.seed()
    random.shuffle(cookable_object)
    cookable_object = cookable_object[0:10]
    print(cookable_object)

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
13. If you need to use togglable objects, must use items in the list :{togglable_object}
14. If you need to use openable objects, must use items in the list :{openable_object}
15. If you need to use cookable objects, must use items in the list :{cookable_object}
16. If you need to use heatsource objects, must use items in the list :{heatsource_object}
17. If you need to use coldsource objects, must use items in the list :{coldsource_object}
18. If you need to use freezable objects, must use items in the list :{freezable_object}
19. The pick and place task don't need to consider whether the object is openable, toggleable, cookable, freezable or not. The robot can pick up and place any object. But the openable, toggleable, cookable, freezable task need to consider whether the object is openable, toggleable, cookable, freezable or not.
20. If you want to cook or freeze something, the scene needs to have a heatsource or a coldsource objects.
21. Each command needs to be diverse, involving operating different objects and rooms. For example, cook the egg in the kitchen_0, and then put the egg on the table in the living room_0. Don't make all the commands like "pick up something and put it on something".

Please generate {amount} command. And make sure they are all reasonable fot the social members.{extra_requirment}You should give a list of command directly which means no any extra punctuations and extra indexs. Don't add a '-' in the front of the commands.'''},
            ],
            }
        ],
        temperature=0.7,
        max_tokens=10000,
    )
    
    response = completion.choices[0].message.content
    print(response)
    return(response)


def command_gen(sceneid,command_num,command_difficulty):
    return command_generation(social_character_gen.social_character_generation(read_files.read_lines_to_array("src/bddl_gen/extracted_descriptions.txt")[sceneid-1],read_files.read_lines_to_array("src/bddl_gen/extracted_rooms.txt")[sceneid-1]),command_num,read_files.read_lines_to_array("src/bddl_gen/extracted_rooms.txt")[sceneid-1],room_size=read_files.extract_room_areas(read_files.read_json_file('src/bddl_gen/area.json'))[sceneid-1],command_difficulty=command_difficulty)