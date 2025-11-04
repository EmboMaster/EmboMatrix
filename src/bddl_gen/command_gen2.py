from openai import OpenAI
import social_character_gen
import read_files
import os    
import random
from prompts.templates import COMMEND_GEN_PROMPT
from src.utils.config_loader import config
api_key = config['task_generation']['api_key']
base_url = config['task_generation']['base_url']
model = config['task_generation']['model']
def command_generation(social_description,amount,rooms,room_size,command_difficulty):
    client = OpenAI(api_key=api_key, base_url=base_url)

    #从文件中采样3个cleaner_object
    cleaner_object = read_files.read_lines_to_array('src/bddl_gen/data/objects_sorted_by_states/cleaner_object.txt')
    random.seed()
    random.shuffle(cleaner_object)
    cleaner_object = cleaner_object[0:3]
    print(cleaner_object)

    #从文件中采样2个knife_object
    knife_object = read_files.read_lines_to_array('src/bddl_gen/data/objects_sorted_by_states/knife_object.txt')
    random.seed()
    random.shuffle(knife_object)
    knife_object = knife_object[0:2]
    print(knife_object)

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

    #从文件中采样2个heatsource_object
    heatsource_object = read_files.read_lines_to_array('src/bddl_gen/data/objects_sorted_by_states/heatsource.txt')
    random.seed()
    random.shuffle(heatsource_object)
    heatsource_object = heatsource_object[0:2]
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

    #从文件中采样10个sliceable_object
    sliceable_object = read_files.read_lines_to_array('src/bddl_gen/data/objects_sorted_by_states/sliceable_object.txt')
    random.seed()
    random.shuffle(sliceable_object)
    sliceable_object = sliceable_object[0:10]
    print(sliceable_object)

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
                    {"type": "text", "text": COMMEND_GEN_PROMPT.format(
                        social_description=social_description,
                        rooms=rooms,
                        room_size=room_size,
                        togglable_object=togglable_object,
                        openable_object=openable_object,
                        cookable_object=cookable_object,
                        heatsource_object=heatsource_object,
                        coldsource_object=coldsource_object,
                        freezable_object=freezable_object,
                        fillable_object=fillable_object,
                        cleaner_object=cleaner_object,
                        cookable_object=cookable_object,
                        amount=amount,
                        extra_requirment=extra_requirment,
                        knife_object=knife_object)},
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
    return command_generation(social_character_gen.social_character_generation(read_files.read_lines_to_array("src/bddl_gen/extracted_descriptions.txt")[sceneid-1],read_files.read_lines_to_array("src/bddl_gen/extracted_rooms.txt")[sceneid-1]),command_num,read_files.read_lines_to_array("src/bddl_gen/extracted_rooms.txt")[sceneid-1],room_size=read_files.extract_room_areas(read_files.read_json_file('src/bddl_gen/area.json'))[sceneid-1],command_difficulty=command_difficulty)