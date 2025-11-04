import pandas as pd
from openai import OpenAI
from nltk.corpus import wordnet
import os
import re
import heapq
import bddl_generation
import importlib.resources
import io
import command_gen
import command_gen2
# import command_gen3
import read_files
import bddl_format_check
from src.utils.config_loader import config
api_key = config['task_generation']['api_key']
base_url = config['task_generation']['base_url']
model = config['task_generation']['model']
def call_gpt(model, prompt, system_prompt="You are a helpful assistant.", temperature=0.7, max_tokens=4096):

    client = OpenAI(api_key=api_key, base_url=base_url)
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    
    output = response.choices[0].message.content.strip()
    return output

def extract_objects(bddl_problem):
    if not isinstance(bddl_problem, str):
        raise TypeError("Input must be a string.")
    match = re.search(r"\(:objects\s+(.*?)\)", bddl_problem, re.DOTALL)
    if match:
        objects_str = match.group(1).strip() 
        object_names = re.findall(r"([a-zA-Z0-9_.-]+)(?:\s*-\s*[a-zA-Z0-9_.-]+)?(?=\s|$)", objects_str)
        unique_object_names = []
        for name in object_names:
            base_name = re.sub(r"(_\d+)$", "", name)
            if base_name not in unique_object_names:
                unique_object_names.append(base_name)
        return unique_object_names
    else:
        return []
    
def extract_problem(bddl_problem):
    pattern = r"\(\s*define\s*\(\s*problem\s+(.*?)\s*\)"  # 匹配任意字符直到右括号
    matches = re.findall(pattern, bddl_problem)
    return matches

def get_all_problems(path, extract_problem):
    problems = []
    
    for root, _, files in os.walk(path):
        for file in files:
            if file == "problem0.bddl":
                file_path = os.path.join(root, file)
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()
                    problem = extract_problem(content)
                    problems.append(problem)
    
    return problems

def wordnet_similarity(word1, word2):
    """使用 WordNet 计算 Wu-Palmer 相似度。"""
    syns1 = wordnet.synsets(word1)
    syns2 = wordnet.synsets(word2)

    if not syns1 or not syns2:
        return 0
    
    syn1 = syns1[0]
    syn2 = syns2[0]

    max_similarity = syn1.wup_similarity(syn2)
    return max_similarity if max_similarity > 0 else 0

def phrase_similarity(phrase1, phrase2):
    if not phrase1 or not phrase2:
        return 0

    phrase1 = re.sub(r'[^\w\s]', '', phrase1).lower()
    phrase2 = re.sub(r'[^\w\s]', '', phrase2).lower()

    words1 = phrase1.split("_")
    words2 = phrase2.split("_")
    final_sim = 1
    if len(words1) > len(words2):
        return 0
    for word1 in words1:
            max_similarity = 0
            for word2 in words2:
                sim = wordnet_similarity(word1,word2)
                if sim is not None and sim > max_similarity:
                    max_similarity = sim
            final_sim = final_sim * max_similarity
    return final_sim if final_sim > 0 else 0

def find_top_similar_wordnet(target_word, word_list, top_n=30):
    """使用 WordNet 查找最相似的单词。"""
    similarities1 = []
    similarities2 = []
    
    for word in word_list:
        similarity1 = phrase_similarity(target_word, word)
        similarities1.append((word, similarity1))
        similarity2 = phrase_similarity(word, target_word)
        similarities2.append((word, similarity2))
    top_similar1 = heapq.nlargest(top_n//2, similarities1, key=lambda item: item[1])
    top_similar2 = heapq.nlargest(top_n//2, similarities2, key=lambda item: item[1])
    return top_similar1+top_similar2

def the_closest_word(word,name):
    lis = ""
    for i in range(len(name)):
        lis += f"{i+1}. name:{name[i]}\n" 
    prompt=f'''You'll be given a word, and list of {len(name)} names and the corresponding definitions,and you should output the name from the list that has the closest meaning to the given word.
                You should only output the closest name, and you don't need to explain or do other things,please just output the name.You shouldn't change the name in the list.The name you output should in the list.
                The list:{lis}
                Given word:{word}
                '''
    return call_gpt('gpt-4o-2024-08-06',prompt)

    
def get_word(word,special_state = ''):
    w = word.split(".")[0]
    df = pd.read_csv('src/bddl_gen/data/BEHAVIOR-1K Synsets.csv', encoding='utf-8')
    name = df["Name"]
    object_dic = read_files.read_json_file('src/bddl_gen/data/synset_object_category.json')
    name = list(object_dic.keys())
    Name = [word.split(".")[0] for word in list(name)]
    for n in Name:
        if w.replace("_", "") ==n.replace("_", ""):
            return name[Name.index(n)]

    top_word = find_top_similar_wordnet(w,Name)
    word_list = [item[0] for item in top_word]
    choice = ''
    while choice not in Name:
        choice = the_closest_word(word,word_list)
    return name[Name.index(choice)]

def ensure_newline_after_domain(p):
    target = "(:domain omnigibson)"
    index = p.find(target)
    
    if index != -1:  # 找到目标字符串
        end_index = index + len(target)
        if end_index < len(p) and p[end_index] != '\n':  
            p = p[:end_index] + '\n' + p[end_index:]  # 插入换行符
    return p

def change_bddl(p,path):
    print(f'p = {p}')
    ob = extract_objects(p)
    problem = extract_problem(p)[0].split('-')[0]
    change_dict = {}
    for wo in ob:
        change_wo = get_word(wo)
        p = p.replace(wo,change_wo)
        change_dict[wo.split('.')[0]] = change_wo.split('.')[0]
    # p = regenerate_problem_name(p)
    new_problem = problem
    for word in change_dict.keys():
        new_problem = new_problem.replace(word,change_dict[word])
    p = p.replace(problem,new_problem)
    print(p)
    problem = extract_problem(p)[0].split('-')[0]
    p = bddl_format_check.remove_first_and_last_lines(p)
    p = ensure_newline_after_domain(p)

    already_problems = get_all_problems(path, extract_problem)
    if already_problems != []:
        same_flag = if_already_exist_problem(problem, already_problems)
    else:
        same_flag = True
            #rule based check
    if not read_files.validate_goal_states(read_files.process_goal_section(p),p,'data/objects_sorted_by_states'):
        print(f"任务的目标状态不完整，未加入到输出列表中")
        return ''
    if not read_files.validate_init_states(read_files.process_init_section(p),'data/objects_sorted_by_states'):
        print(f"任务的初始状态不完整，未加入到输出列表中")
        return ''
    if bddl_completeness_check(p) and same_flag == True:

        flag = True
        while flag:
            try:
                print(f'{problem}太长了，重新缩短')
                problem = shorten_problem_name(problem)
                filename = f"{path}/{problem}/problem0.bddl"
                directory = os.path.dirname(filename)
                os.makedirs(directory, exist_ok=True)
                flag = False
            except:
                flag = True
        with open(filename, 'w', encoding='utf-8') as f:  # 指定 utf-8 编码，处理中文等特殊字符
            f.write(p)
        return p
    else:
        if same_flag != True:
            print(f"{problem}任务已经存在")
        else:
            print(f'{problem}任务不完整，未加入到输出列表中')
        return ''

def generate_bddl(scene_index,task='',path='bddl_data',roomids = ''):
    problems = bddl_generation.bddl_generation(task,path,roomids=roomids)
    lis=[]
    test=''
    for p in problems:

        test=test+change_bddl(p["content"],f"{path}/{read_files.read_lines_to_array('src/bddl_gen/room_names.txt')[scene_index-1]}")+'\n'    
    filename = f"{path}/txt/final_bddl.txt"
    directory = os.path.dirname(filename)
    os.makedirs(directory, exist_ok=True)
    with open(filename, 'w', encoding='utf-8') as f:  # 指定 utf-8 编码，处理中文等特殊字符
        f.write(test)
    return test

# def generate_bddl2(scene_index,task='',path='bddl_data',roomids = ''):
#     problems = bddl_generation.bddl_generation_dialogue(task,path,roomids=roomids)
#     lis=[]
#     test=''
#     print(f"problems = {problems}") 
#     for p in problems:

#         test=test+change_bddl(p["content"],f"{path}/{read_files.read_lines_to_array('room_names.txt')[scene_index-1]}")+'\n'    
#     filename = f"{path}/txt/final_bddl.txt"
#     directory = os.path.dirname(filename)
#     os.makedirs(directory, exist_ok=True)
#     with open(filename, 'w', encoding='utf-8') as f:  # 指定 utf-8 编码，处理中文等特殊字符
#         f.write(test)
#     return test

#这是给用户直接调用的
#结果会直接输出到final_bddl.txt中，然后bddl文件会被分别放在problem里面，是常规的bddl格式和命名
def generate_bddl_api(scene_index,command_num,command_difficulty,save_path):
    generate_bddl(scene_index=scene_index,task=command_gen2.command_gen(sceneid=scene_index,command_num=command_num,command_difficulty=command_difficulty),path=save_path,roomids=read_files.read_lines_to_array("src/bddl_gen/extracted_rooms.txt")[scene_index-1])

#使用对话来获取命令的api
# def generate_bddl_api2(scene_index,dialogue_turns,save_path):
#     generate_bddl2(scene_index=scene_index,task=command_gen3.command_generation_from_dialogue(sceneid=scene_index,chat_turn_limit=dialogue_turns,model=None),path=save_path,roomids=read_files.read_lines_to_array("extracted_rooms.txt")[scene_index-1])
def if_already_exist_problem(problem, problems):


    client = OpenAI(api_key=api_key, base_url=base_url)

    completion = client.chat.completions.create(
        model=model,
        messages = [
            {
            "role": "user",
            "content": [
                    {"type": "text", "text": f'''You are given two inputs, a new task and a list of existed tasks. You should output whether the new task is already generated before. It is important if the involved objects, actions and the involved rooms are the same, the task is considered as generated before. The input problem name is {problem}. The list of problem names is {problems}. Please output a conclusion in the format like [[Generated]] or [[New]]:'''},
            ],
            }
        ],
        temperature=0.7,
        max_tokens=4095,
    )
    
    response = completion.choices[0].message.content
    if "[[New]]" in response or "[[new]]" in response:
        return True
    elif "[[Generated]]" in response or "[[generated]]" in response:
        return False
    else:
        return None  # 如果没有找到，返回 None 或者可以自定义其他返回值

def shorten_problem_name(problem_name):


    client = OpenAI(api_key=api_key, base_url=base_url)

    completion = client.chat.completions.create(
        model=model,
        messages = [
            {
            "role": "user",
            "content": [
                    {"type": "text", "text": f'''You should shorten and simplify the input problem name to make it be able to as a folder name. The input problem name is {problem_name}. Please output the shortened problem name in a format as below:
                    <result>XXXXX</result>'''},
            ],
            }
        ],
        temperature=0.7,
        max_tokens=4095,
    )
    
    response = completion.choices[0].message.content

    response1 = response.split("<result>")[1]
    response2 = response1.split("</result>")[0]
    return response2
    


def bddl_completeness_check(bddl_file):

    client = OpenAI(api_key=api_key, base_url=base_url)

    completion = client.chat.completions.create(
        model=model,
        messages = [
            {
            "role": "user",
            "content": [
                    {"type": "text", "text": f'''The Behavior Domain Definition Language (BDDL) is a domain-specific language for defining activity scenarios in virtual environments, designed for the BEHAVIOR benchmark. Its syntax and structure include:
        1. Problem Definition:
        - Each activity is defined as a problem.
        - Includes references to the :domain it belongs to (e.g., omnigibson).
        2. Components:
        - :objects: Lists the objects involved, categorized by types (e.g., pool.n.01_1 - pool.n.01).task-relevant objects, where each line represents a WordNet synset of the object. For example, candle.n.01_1 candle.n.01_2 candle.n.01_3 candle.n.01_4 - candle.n.01 indicates that four objects that belong to the candoe.n.01 synset are needed for this task.The object you choose should not be abstract(eg:ground),it should be concrete(eg:floor).The name of the room should not be listed in the object.(eg:kitchen)
        - :init: Defines the initial state with ground literals, such as object locations or states (e.g., ontop, inroom).initial conditions of the task, where each line represents a ground predicate that holds at the beginning of the task. For example, (ontop candle.n.01_1 table.n.02_1) indicates that the first candle is on top of the first table when the task begins.
        - :goal: Specifies the conditions that need to be satisfied for success, usingS logical expressions like and, not.goal conditions of the task, where each line represents a ground predicate and each block represents a non-ground predicate (e.g. forall, forpairs, and, or, etc) that should hold for the task to be considered solved. For example, (inside ?candle.n.01 ?wicker_basket.n.01) indicates that the candle should be inside the wicker basket at the end of the task.The format of objects here needs a '?' when :1. (?boot.n.01 - boot.n.01) 2.(inside ?pebble.n.01_1 ?tank.n.02_1). The first type represents exist or ergodic, the second type represents the judgement of state.
        3. Predicates:
        - Binary predicates (e.g., ontop, inside) specify spatial relationships or configurations.
        - Unary predicates (e.g., stained, cooked) describe individual object properties or states.    - Here are the predicates that are available,and some examples about how to use them.    (cooked kabob.n.01_1)
        (frozen chicken.n.01_1)
        (open window.n.01_1)
        (folded ?sock.n.01)
        (unfolded dress.n.01_1)
        (toggled_on air_conditioner.n.01_1)
        (hot potato.n.01_1)
        (on_fire ?firewood.n.01)
        (future sugar_cookie.n.01_5)
        (real ?cheese_tart.n.01_1)
        (saturated rag.n.01_1 water.n.06_1)
        (covered book.n.02_1 dust.n.01_1)
        (filled pool.n.01_1 water.n.06_1)
        (contains ?bowl.n.01_1 ?white_rice.n.01_1)
        (ontop agent.n.01_1 floor.n.01_1)
        (under ?gym_shoe.n.01 ?table.n.02_1)
        (touching ?chair.n.01_1 ?bed.n.01_1)
        (inside mug.n.04_1 cabinet.n.01_1)
        (overlaid ?sheet.n.03_2 ?bed.n.01_1)
        (attached light_bulb.n.01_1 table_lamp.n.01_1)
        (draped ?long_trousers.n.01_1 ?shelf.n.01_1)
        (insource vanilla__bottle.n.01_1 vanilla.n.02_1)
        (inroom floor.n.01_1 living_room)
        (broken mailbox.n.01_1)
        (grasped ?obj1 ?obj2)

        Here are the definitions of logical expression.
        - Conjunction (and): Logical "AND" operation, performing an "AND" operation on multiple subexpressions.
        - Disjunction (or): Logical "OR" operation, performing an "OR" operation on multiple subexpressions.
        - Negation (not): Logical "NOT" operation, performing a "NOT" operation on a single subexpression.
        - Implication (imply): Logical "IMPLY" operation, representing "if the premise holds, then the conclusion holds."
        - Universal (forall): Universal quantifier operation, representing that a condition holds for all objects within a given range.
        - Existential (exists): Existential quantifier operation, representing that a condition holds for some objects within a given range.
        - NQuantifier (forn): Represents "exactly N" elements that satisfy a given condition.
        - ForPairs (forpairs): Quantification over pairs of objects, representing that the condition holds for all object pairs.
        - ForNPairs (fornpairs): Represents that exactly N pairs of objects satisfy a given condition.

        Here are some examples of bddl files which are without the description.

(define (problem preparing_clothes_for_the_next_day-0)
                (:domain omnigibson)

                (:objects
                    coat.n.01_1 - coat.n.01
                    wardrobe.n.01_1 - wardrobe.n.01
                    boot.n.01_1 boot.n.01_2 - boot.n.01
                    trouser.n.01_1 - trouser.n.01
                    cedar_chest.n.01_1 - cedar_chest.n.01
                    bed.n.01_1 - bed.n.01
                    wallet.n.01_1 - wallet.n.01
                    floor.n.01_1 floor.n.01_2 - floor.n.01
                    agent.n.01_1 - agent.n.01
                )

                (:init
                    (inside coat.n.01_1 wardrobe.n.01_1)
                    (ontop boot.n.01_1 floor.n.01_1)
                    (ontop boot.n.01_2 floor.n.01_1)
                    (inside trouser.n.01_1 wardrobe.n.01_1)
                    (inside wallet.n.01_1 cedar_chest.n.01_1)
                    (inroom wardrobe.n.01_1 closet)
                    (ontop cedar_chest.n.01_1 floor.n.01_1)
                    (inroom floor.n.01_1 closet)
                    (inroom floor.n.01_2 childs_room)
                    (inroom bed.n.01_1 childs_room)
                    (ontop agent.n.01_1 floor.n.01_1)
                )
                (:goal
                    (and
                        (ontop ?coat.n.01_1 ?bed.n.01_1)
                        (ontop ?trouser.n.01_1 ?bed.n.01_1)
                        (forall
                            (?boot.n.01 - boot.n.01)
                            (ontop ?boot.n.01 ?floor.n.01_2)
                        )
                        (ontop ?wallet.n.01_1 ?bed.n.01_1)
                    )
                )
            )

            (define (problem setup_a_fish_tank-0)
                (:domain omnigibson)

                (:objects
                    tank.n.02_1 - tank.n.02
                    table.n.02_1 - table.n.02
                    bucket.n.01_1 - bucket.n.01
                    floor.n.01_1 - floor.n.01
                    water.n.06_1 - water.n.06
                    water_filter.n.01_1 - water_filter.n.01
                    pebble.n.01_1 - pebble.n.01
                    agent.n.01_1 - agent.n.01
                )
                (:init
                    (ontop tank.n.02_1 table.n.02_1)
                    (open tank.n.02_1)
                    (ontop bucket.n.01_1 floor.n.01_1)
                    (filled bucket.n.01_1 water.n.06_1)
                    (ontop water_filter.n.01_1 table.n.02_1)
                    (ontop pebble.n.01_1 table.n.02_1)
                    (inroom table.n.02_1 living_room)
                    (inroom floor.n.01_1 living_room)
                    (ontop agent.n.01_1 floor.n.01_1)
                )
                (:goal
                    (and
                        (not
                            (open ?tank.n.02_1)
                        )
                        (filled ?tank.n.02_1 ?water.n.06_1)
                        (inside ?water_filter.n.01_1 ?tank.n.02_1)
                        (inside ?pebble.n.01_1 ?tank.n.02_1)
                    )
                )
            )

(define (problem cook_carrots_in_the_kitchen-0)
    (:domain omnigibson)

    (:objects
        saucepot.n.01_1 - saucepot.n.01
        stove.n.01_1 - stove.n.01
        carrot.n.03_1 carrot.n.03_2 carrot.n.03_3 carrot.n.03_4 carrot.n.03_5 carrot.n.03_6 - carrot.n.03
        water.n.06_1 - water.n.06
        sink.n.01_1 - sink.n.01
        electric_refrigerator.n.01_1 - electric_refrigerator.n.01
        floor.n.01_1 - floor.n.01
        cabinet.n.01_1 - cabinet.n.01
        agent.n.01_1 - agent.n.01
    )
    
    (:init 
        (ontop saucepot.n.01_1 stove.n.01_1) 
        (inside carrot.n.03_1 electric_refrigerator.n.01_1) 
        (inside carrot.n.03_2 electric_refrigerator.n.01_1) 
        (inside carrot.n.03_3 electric_refrigerator.n.01_1) 
        (inside carrot.n.03_4 electric_refrigerator.n.01_1) 
        (inside carrot.n.03_5 electric_refrigerator.n.01_1) 
        (inside carrot.n.03_6 electric_refrigerator.n.01_1) 
        (not 
            (cooked carrot.n.03_1)
        )
        (not 
            (cooked carrot.n.03_2)
        )
        (not 
            (cooked carrot.n.03_3)
        )
        (not 
            (cooked carrot.n.03_4)
        )
        (not 
            (cooked carrot.n.03_5)
        )
        (not 
            (cooked carrot.n.03_6)
        )
        (insource sink.n.01_1 water.n.06_1)
        (inroom sink.n.01_1 kitchen)
        (inroom stove.n.01_1 kitchen) 
        (inroom floor.n.01_1 kitchen) 
        (inroom electric_refrigerator.n.01_1 kitchen) 
        (inroom cabinet.n.01_1 kitchen) 
        (ontop agent.n.01_1 floor.n.01_1)
    )
    
    (:goal 
        (and 
            (forall 
                (?carrot.n.03 - carrot.n.03) 
                (cooked ?carrot.n.03)
            )
        )
    )
)


            (define (problem bringing_glass_to_recycling-0)
                (:domain omnigibson)

                (:objects
                    water_glass.n.02_1 - water_glass.n.02
                    recycling_bin.n.01_1 - recycling_bin.n.01
                    door.n.01_1 - door.n.01
                    floor.n.01_1 floor.n.01_2 - floor.n.01
                    agent.n.01_1 - agent.n.01
                )
                (:init
                    (ontop water_glass.n.02_1 floor.n.01_1)
                    (ontop recycling_bin.n.01_1 floor.n.01_2)
                    (inroom door.n.01_1 kitchen)
                    (inroom floor.n.01_1 kitchen)
                    (inroom floor.n.01_2 garden)
                    (ontop agent.n.01_1 floor.n.01_1)
                )
                (:goal
                    (and
                        (inside ?water_glass.n.02_1 ?recycling_bin.n.01_1)
                        (not
                            (open ?recycling_bin.n.01_1)
                        )
                    )
                )
            )

            (define (problem can_syrup-0)
                (:domain omnigibson)

                (:objects
                    lid.n.02_1 lid.n.02_2 lid.n.02_3 - lid.n.02
                    cabinet.n.01_1 - cabinet.n.01
                    mason_jar.n.01_1 mason_jar.n.01_2 mason_jar.n.01_3 - mason_jar.n.01
                    stockpot.n.01_1 - stockpot.n.01
                    stove.n.01_1 - stove.n.01
                    maple_syrup.n.01_1 - maple_syrup.n.01
                    floor.n.01_1 - floor.n.01
                    agent.n.01_1 - agent.n.01
                )
    
                (:init 
                    (inside lid.n.02_1 cabinet.n.01_1) 
                    (inside lid.n.02_2 cabinet.n.01_1) 
                    (inside lid.n.02_3 cabinet.n.01_1) 
                    (inside mason_jar.n.01_1 cabinet.n.01_1) 
                    (inside mason_jar.n.01_2 cabinet.n.01_1) 
                    (inside mason_jar.n.01_3 cabinet.n.01_1) 
                    (ontop stockpot.n.01_1 stove.n.01_1) 
                    (not 
                        (toggled_on stove.n.01_1)
                    )
                    (filled stockpot.n.01_1 maple_syrup.n.01_1) 
                    (inroom cabinet.n.01_1 kitchen) 
                    (inroom stove.n.01_1 kitchen) 
                    (inroom floor.n.01_1 kitchen) 
                    (ontop agent.n.01_1 floor.n.01_1)
                )
    
                (:goal 
                    (and 
                        (forpairs 
                            (?lid.n.02 - lid.n.02)
                            (?mason_jar.n.01 - mason_jar.n.01)
                            (ontop ?lid.n.02 ?mason_jar.n.01)
                        )
                        (forall 
                            (?mason_jar.n.01 - mason_jar.n.01)
                            (filled ?mason_jar.n.01 ?maple_syrup.n.01_1)
                        )
                    )
                )
            )

(define (problem turning_off_the_hot_tub-0)
    (:domain omnigibson)

    (:objects
        hot_tub.n.02_1 - hot_tub.n.02
        water.n.06_1 - water.n.06
        floor.n.01_1 - floor.n.01
        agent.n.01_1 - agent.n.01
    )
    
    (:init 
        (toggled_on hot_tub.n.02_1)
        (filled hot_tub.n.02_1 water.n.06_1)
        (inroom hot_tub.n.02_1 spa)
        (inroom floor.n.01_1 spa) 
        (ontop agent.n.01_1 floor.n.01_1)
    )
    
    (:goal 
        (and 
            (not 
                (toggled_on ?hot_tub.n.02_1)
            )
        )
    )
)
                         
(define (problem preparing_food_or_drink_for_sale-0)
    (:domain omnigibson)

    (:objects
        hamburger.n.01_1 - hamburger.n.01
        platter.n.01_1 - platter.n.01
        tupperware.n.01_1 - tupperware.n.01
        french_fries.n.02_1 - french_fries.n.02
        cabinet.n.01_1 - cabinet.n.01
        electric_refrigerator.n.01_1 - electric_refrigerator.n.01
        microwave.n.02_1 - microwave.n.02
        floor.n.01_1 - floor.n.01
        agent.n.01_1 - agent.n.01
    )
    
    (:init 
        (inside hamburger.n.01_1 electric_refrigerator.n.01_1) 
        (inside french_fries.n.02_1 tupperware.n.01_1)
        (inside tupperware.n.01_1 electric_refrigerator.n.01_1)
        (inside platter.n.01_1 cabinet.n.01_1) 
        (inroom cabinet.n.01_1 kitchen) 
        (inroom electric_refrigerator.n.01_1 kitchen) 
        (inroom microwave.n.02_1 kitchen) 
        (inroom floor.n.01_1 kitchen) 
        (ontop agent.n.01_1 floor.n.01_1)
    )
    
    (:goal 
        (and 
            (ontop ?french_fries.n.02_1 ?platter.n.01_1)
            (hot ?french_fries.n.02_1)
            (ontop ?hamburger.n.01_1 ?platter.n.01_1)
            (hot ?hamburger.n.01_1)
        )
    )
)

        Now here is a bddl file {bddl_file}, the format of this bddl file is correct. You need to check whether the bddl file set up a correct scene for the task. You should check the init and the goal to see whether it completes the task. Then output your analysis and end with a conclusion in the format like [[complete]] or [[incomplete]]
        Output in the format like:
        1. Analyse the task name, find out the objects, rooms, actions and predicates in the task name.Check whether all the operated objects have a specific init place, which will be the same as the init section. If an object does not have a specific init place, the bddl is incomplete.
        2. Analyse the whole task, check whether it's reasonable and proper in a logical way. And check whether it's safe to do the task in the scene. If it's not safe or not reasonable, the bddl is incomplete. 
        3. Analyse the init section, check whether the objects are in the right rooms, and whether the predicates are correct. Check whether all the objects in the object section have a specific init place. If there are conflicts, the bddl is incomplete.
        4. Analyse the goal section, check whether the states of the objects are correct.For example, a vaccum can't be soaked. No conflict should exist. Thing like:      (soaked ?scrub_brush.n.01_1)(not (soaked ?scrub_brush.n.01_1)) is a conflict. If there are conflicts, the bddl is incomplete.
        5. Finally, output a conclusion in the format like [[complete]] or [[incomplete]].'''},

            ],
            }
        ],
        temperature=0.4,
        max_tokens=4095,
    )
    
    response = completion.choices[0].message.content
    if "[[complete]]" in response:
        return True
    elif "[[incomplete]]" in response:
        return False
    else:
        return False  # 如果没有找到，返回 None 或者可以自定义其他返回值

def regenerate_problem_name(bddl_file):


    client = OpenAI(api_key=api_key, base_url=base_url)

    completion = client.chat.completions.create(
        model=model,
        messages = [
            {
            "role": "user",
            "content": [
                    {"type": "text", "text": f'''The Behavior Domain Definition Language (BDDL) is a domain-specific language for defining activity scenarios in virtual environments, designed for the BEHAVIOR benchmark. Its syntax and structure include:
        1. Problem Definition:
        - Each activity is defined as a problem.
        - Includes references to the :domain it belongs to (e.g., omnigibson).
        2. Components:
        - :objects: Lists the objects involved, categorized by types (e.g., pool.n.01_1 - pool.n.01).task-relevant objects, where each line represents a WordNet synset of the object. For example, candle.n.01_1 candle.n.01_2 candle.n.01_3 candle.n.01_4 - candle.n.01 indicates that four objects that belong to the candoe.n.01 synset are needed for this task.The object you choose should not be abstract(eg:ground),it should be concrete(eg:floor).The name of the room should not be listed in the object.(eg:kitchen)
        - :init: Defines the initial state with ground literals, such as object locations or states (e.g., ontop, inroom).initial conditions of the task, where each line represents a ground predicate that holds at the beginning of the task. For example, (ontop candle.n.01_1 table.n.02_1) indicates that the first candle is on top of the first table when the task begins.
        - :goal: Specifies the conditions that need to be satisfied for success, usingS logical expressions like and, not.goal conditions of the task, where each line represents a ground predicate and each block represents a non-ground predicate (e.g. forall, forpairs, and, or, etc) that should hold for the task to be considered solved. For example, (inside ?candle.n.01 ?wicker_basket.n.01) indicates that the candle should be inside the wicker basket at the end of the task.The format of objects here needs a '?' when :1. (?boot.n.01 - boot.n.01) 2.(inside ?pebble.n.01_1 ?tank.n.02_1). The first type represents exist or ergodic, the second type represents the judgement of state.
        3. Predicates:
        - Binary predicates (e.g., ontop, inside) specify spatial relationships or configurations.
        - Unary predicates (e.g., stained, cooked) describe individual object properties or states.    - Here are the predicates that are available,and some examples about how to use them.    (cooked kabob.n.01_1)
        (frozen chicken.n.01_1)
        (open window.n.01_1)
        (folded ?sock.n.01)
        (unfolded dress.n.01_1)
        (toggled_on air_conditioner.n.01_1)
        (hot potato.n.01_1)
        (on_fire ?firewood.n.01)
        (future sugar_cookie.n.01_5)
        (real ?cheese_tart.n.01_1)
        (saturated rag.n.01_1 water.n.06_1)
        (covered book.n.02_1 dust.n.01_1)
        (filled pool.n.01_1 water.n.06_1)
        (contains ?bowl.n.01_1 ?white_rice.n.01_1)
        (ontop agent.n.01_1 floor.n.01_1)
        (under ?gym_shoe.n.01 ?table.n.02_1)
        (touching ?chair.n.01_1 ?bed.n.01_1)
        (inside mug.n.04_1 cabinet.n.01_1)
        (overlaid ?sheet.n.03_2 ?bed.n.01_1)
        (attached light_bulb.n.01_1 table_lamp.n.01_1)
        (draped ?long_trousers.n.01_1 ?shelf.n.01_1)
        (insource vanilla__bottle.n.01_1 vanilla.n.02_1)
        (inroom floor.n.01_1 living_room)
        (broken mailbox.n.01_1)
        (grasped ?obj1 ?obj2)
        PLEASE PAY ATTENTION:All the predicates you use must come from above.for example, you shouldn't use predicates like (clean ?floor.n.01_1),instead you can use the given predicates like (covered ?dust.n.01_1 ?floor.n.01_1) to describe this kind of situation.    4. Description:
        - For object types ,beacuse there are too many you can use the types you think are needed in the task. After using a type , please write a short expression about it at the end of the bddl file which uses the type.Every object type and roomtype in the Init section and objects section needs an expression.

        Here are the definitions of logical expression.
        - Conjunction (and): Logical "AND" operation, performing an "AND" operation on multiple subexpressions.
        - Disjunction (or): Logical "OR" operation, performing an "OR" operation on multiple subexpressions.
        - Negation (not): Logical "NOT" operation, performing a "NOT" operation on a single subexpression.
        - Implication (imply): Logical "IMPLY" operation, representing "if the premise holds, then the conclusion holds."
        - Universal (forall): Universal quantifier operation, representing that a condition holds for all objects within a given range.
        - Existential (exists): Existential quantifier operation, representing that a condition holds for some objects within a given range.
        - NQuantifier (forn): Represents "exactly N" elements that satisfy a given condition.
        - ForPairs (forpairs): Quantification over pairs of objects, representing that the condition holds for all object pairs.
        - ForNPairs (fornpairs): Represents that exactly N pairs of objects satisfy a given condition.

        Here are some examples of bddl files which are without the description.

        (define (problem place_the_bowl_from_the_kitchen_0_booth_on_the_conference_table_in_the_dining_room_0)
        (:domain omnigibson)    (:objects
        bowl.n.01_1 - bowl.n.01
        booth.n.01_1 - booth.n.01
        conference_table.n.01_1 - conference_table.n.01
        floor.n.01_1 - floor.n.01
        agent.n.01_1 - agent.n.01
        )

        (:init
        (ontop bowl.n.01_1 booth.n.01_1)
        (inroom booth.n.01_1 kitchen_0)
        (inroom conference_table.n.01_1 dining_room_0)
        (ontop agent.n.01_1 floor.n.01_1)
        (inroom floor.n.01_1 kitchen_0)
        )

        (:goal
        (ontop ?bowl.n.01_1 ?conference_table.n.01_1)
        )
        )             

        

        Now here is a bddl file {bddl_file}, the format of this bddl file is correct. You need to regenerate its problem name to match the objects' name in the bddl description, output only the entire bddl file'''},
            ],
            }
        ],
        temperature=0.7,
        max_tokens=4095,
    )
    response = completion.choices[0].message.content

    return response

p = """
(define (problem chopping_vegetables-0)
    (:domain omnigibson)

    (:objects
        tomato.n.01_1 tomato.n.01_2 - tomato.n.01
        table_knife.n.01_1 - table_knife.n.01
        dish.n.01_1 dish.n.01_2 - dish.n.01
        cabinet.n.01_1 - cabinet.n.01
        fridge.n.01_1 - fridge.n.01
        countertop.n.01_1 - countertop.n.01
        floor.n.01_1 - floor.n.01
    )

    (:init
        (ontop tomato.n.01_1 countertop.n.01_1)
        (ontop tomato.n.01_2 countertop.n.01_1)
        (ontop table_knife.n.01_1 countertop.n.01_1)
        (inside dish.n.01_1 cabinet.n.01_1)
        (inside dish.n.01_2 cabinet.n.01_1)
        (ontop agent.n.01_1 floor.n.01_1)
        (ontop countertop.n.01_1 floor.n.01_1)
        (ontop cabinet.n.01_1 floor.n.01_1)
        (inroom floor.n.01_1 kitchen)
    )

    (:goal
        (and
            (forall (?tomato.n.01 - tomato.n.01)
                (and
                    (exists (?dish.n.01 - dish.n.01)
                        (inside ?tomato.n.01 ?dish.n.01)
                    )
                    (sliced ?tomato.n.01)
                )
            )
    )
)
"""
# result = extract_objects(p)
# print(result)

# if not read_files.validate_goal_states(read_files.process_goal_section(p),'data/objects_sorted_by_states'):
#     print(f"任务的目标状态不完整，未加入到输出列表中")
#print(get_word('sink'))
 