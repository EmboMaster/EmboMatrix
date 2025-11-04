#python bddl_generation.py
from openai import OpenAI
import os
import bddl_format_check
from prompts.templates import TASK_GENERATION_PROMPT
from src.utils.config_loader import config
api_key = config['task_generation']['api_key']
base_url = config['task_generation']['base_url']
def bddl_generation(task,path,roomids):
    #适配task是字符串输入和列表输入的情况
    if isinstance(task, str):
        task_list = task.split("\n")
        task = []
        for tmp_task in task_list:
            if len(tmp_task) < 10:
               continue
            task.append(tmp_task)
    roomids = roomids.split(":")[1].split(",")

    def check_inroom_locations(problem_str, valid_locations):
        """
        检查问题字符串中所有 (inroom A B) 的 B 是否都在 valid_locations 列表中
        
        参数:
            problem_str (str): 问题定义字符串
            valid_locations (list): 有效的房间名称列表
            
        返回:
            bool: 如果所有 inroom 位置都有效返回 True,否则返回 False
        """
        import re
        # 使用正则表达式匹配所有 (inroom A B) 中的 B
        inroom_matches = re.findall(r'\(inroom\s+\S+\s+(\S+)\)', problem_str)
        
        # 检查每个匹配的位置是否在有效列表中
        for location in inroom_matches:
            if location not in valid_locations:

                return False, f"Attention!! {location} is not a valid room."
        
        return True,""
    
    def generation_each_task(task, error_hint):

        if task=='':
            input="10 new bddl files"
        else:
            input=f"the one and only bddl file for the task:{task}"

        client = OpenAI(api_key=api_key, base_url=base_url)

        completion = client.chat.completions.create(
            model="gpt-4o",
            messages = [
                {
                "role": "user",
                "content": [
                        {"type": "text", "text": TASK_GENERATION_PROMPT.format(
                            roomids=roomids,
                            error_hint=error_hint,
                            input=input)}],
                }
            ],
            temperature=0.7,
            max_tokens=4095,
        )
        
        response = completion.choices[0].message.content
        print(response)
        if not os.path.exists(path):
            os.makedirs(path)
        filename = f'{path}/txt/raw_bddl.txt'
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, 'w', encoding='utf-8') as file:
            file.write(response)
        
        bddl_format_check.bddl_format_check(response,path)

        filename = f'{path}/txt/checked_bddl.txt'
        with open(filename, 'r', encoding='utf-8') as file:
            bddl_files = file.read()

        return bddl_files
    
    bddl_files = []
    for i in range(len(task)):
        flag = True
        error_hint = ""
        max_iteration = 3
        while flag and max_iteration > 0:
            max_iteration -= 1
            new_task = generation_each_task(task[i], error_hint)
            flag, error_hint = check_inroom_locations(new_task, roomids)
            flag = not flag
            if not flag:
                bddl_files.append(new_task)
    
    bddl_files = "\n\n".join(bddl_files) + "\n\n"

    return bddl_format_check.extract_bddl_files(bddl_files)


def bddl_generation_dialogue(task,path,roomids):
    #适配task是字符串输入和列表输入的情况
    #这里task数据格式是一个列表，每一个列表元素是一个tuple，第一个量是原始命令，第二个量是具体命令
    if isinstance(task, str):
        task_list = task.split("\n")
        task = []
        for tmp_task in task_list:
            if len(tmp_task) < 10:
               continue
            task.append(tmp_task)
    roomids = roomids.split(":")[1].split(",")

    def check_inroom_locations(problem_str, valid_locations):
        """
        检查问题字符串中所有 (inroom A B) 的 B 是否都在 valid_locations 列表中
        
        参数:
            problem_str (str): 问题定义字符串
            valid_locations (list): 有效的房间名称列表
            
        返回:
            bool: 如果所有 inroom 位置都有效返回 True,否则返回 False
        """
        import re
        # 使用正则表达式匹配所有 (inroom A B) 中的 B
        inroom_matches = re.findall(r'\(inroom\s+\S+\s+(\S+)\)', problem_str)
        
        # 检查每个匹配的位置是否在有效列表中
        for location in inroom_matches:
            if location not in valid_locations:

                return False, f"Attention!! {location} is not a valid room."
        
        return True,""
    
    def generation_each_task(task, error_hint):

        if task=='':
            input="10 new bddl files"
        else:
            input=f"the one and only bddl file for the task:{task[1]}"

        client = OpenAI(api_key=api_key, base_url=base_url)

        completion = client.chat.completions.create(
            model="gpt-4o",
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
            (unfolded dress.n.01_1)
            (toggled_on air_conditioner.n.01_1)
            (hot potato.n.01_1)
            (on_fire ?firewood.n.01)
            (real ?cheese_tart.n.01_1)
            (contains ?bowl.n.01_1 ?white_rice.n.01_1)
            (ontop agent.n.01_1 floor.n.01_1)
            (under ?gym_shoe.n.01 ?table.n.02_1)
            (inside mug.n.04_1 cabinet.n.01_1)
            (inroom floor.n.01_1 living_room)
            PLEASE PAY ATTENTION:All the predicates you use must come from above.for example, you shouldn't use predicates like (closed ?closet.n.01_1),instead you can use the given predicates like (not\n (open ?closet.n.01_1)\n) to describe this kind of situation.    4. Description:
            - For object types ,beacuse there are too many you can use the types you think are needed in the task. After using a type , please write a short expression about it at the end of the bddl file which uses the type.Every object type and roomtype in the Init section and objects section needs an expression.

            Here are the definitions of logical expression.
            - Conjunction (and): Logical "AND" operation, performing an "AND" operation on multiple subexpressions.
            - Disjunction (or): Logical "OR" operation, performing an "OR" operation on multiple subexpressions.
            - Negation (not): Logical "NOT" operation, performing a "NOT" operation on a single subexpression.
            - Implication (imply): Logical "IMPLY" operation, representing "if the premise holds, then the conclusion holds."
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

            Important rules:
            1.You don't need to add object like hands.    
            2.NEVER invent supporting or other objects not mentioned in the task (e.g., if task says "shoes on bench", don't add mannequin)
            3.If you want to add two floor, please use floor.n.01_1 floor.n.01_2 - floor.n.01 in one row,don't split them into floor.n.01_1 - floor.n.01 and floor.n.01_2 - floor.n.01.
            4. If some item is to be cooked or frozen, it must be in/on a heater or freezer, and the heater or freezer must be toggled on. You need to check theses conditions in the goal section. And the heater or freezer needs to be not toggled_on in the init section.
            5. If the last step is to heat/freeze an object, the heater/cooler should be toggled_on
            6. When you generate the goal, you needn't care about the relationship between the objects and the agent, just make sure the objects are in the right place, under the right conditions. Don't generate things like (inside ?carton.n.02_1 ?agent.n.01_1)
            7.'inroom' can't be used in the goal section, it can only be used in the init section.
            8.Cooler can't be toggled on, it's always toggled on. So to freeze something you just need to put it in the cooler. However, you still need to check if the object is frozen in the goal section. Which means the object is frozen and the object is in the cooler.
            9. If something is cooked or frozen and be picked up to other places, you don't need to check whether the heatsource or cooler is toggled on or whether the object is inside the heater or cooler, but you need to check if the object is cooked or frozen in the goal section. This is very important. If the object is not picked up after cooked or frozen, you need to check the heater is toggled on and the object is inside the heater or cooler in the goal section.
                         
            You can only use rooms as follows:{roomids}, the bddl file you generate can't exceed the range. {error_hint}             
            Now please output {input} for me. replace the problem name in the bddl file to {task[0]}.If the tasks don't emphasize the object is on the floor, then don't need to add the object floor, just use the inroom predicates to init the objects. However, the agent need to be init ontop of a floor, so you need at least one floor object to place the agent. Please remember the '?' in the goal section is a must when the object is a particular object instead of an object type. The task name is the command sentence which will give the full information of the task, don't simplify its content, just change the format
            Output in the format like:
            1.Analyse the task, find out the objects and the relationships between them. Make sure you know all the objects and their relationships in the task.
            2.Analyse the init states of the objects and the relationships between them. Make sure you know all the init states of the objects and their relationships in the task.
            3.Analyse the goal states of the objects and the relationships between them. Make sure you know all the goal states of the objects and their relationships in the task.
            4.Analyse the predicates of the objects and the relationships between them. Make sure you know all the predicates of the objects and their relationships in the task.
            5.Check whether the task involves cooking or freezing. If so, check the rule 9, which means if the cooked or frozen object is picked up, you need to check if the object is cooked or frozen in the goal section and do not check the heater or cooler is toggled on or the object is inside the heater or cooler.If the object is not picked up after cooked or frozen, you need to check the heater is toggled on and the object is inside the heater or cooler in the goal section.
            6.Generate the bddl file according to the above analysis. The bddl file should be in the format of the BDDL file, and the objects and their relationships should be in the format of the BDDL file. The predicates of the objects and their relationships should be in the format of the BDDL file.'''},
                ],
                }
            ],
            temperature=0.7,
            max_tokens=4095,
        )
        
        response = completion.choices[0].message.content
        print(response)
        if not os.path.exists(path):
            os.makedirs(path)
        filename = f'{path}/txt/raw_bddl.txt'
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, 'w', encoding='utf-8') as file:
            file.write(response)
        
        bddl_format_check.bddl_format_check(response,path)

        filename = f'{path}/txt/checked_bddl.txt'
        with open(filename, 'r', encoding='utf-8') as file:
            bddl_files = file.read()

        return bddl_files
    
    bddl_files = []
    for i in range(len(task)):
        flag = True
        error_hint = ""
        max_iteration = 3
        while flag and max_iteration > 0:
            max_iteration -= 1
            new_task = generation_each_task(task[i], error_hint)
            flag, error_hint = check_inroom_locations(new_task, roomids)
            flag = not flag
            if not flag:
                bddl_files.append(new_task)
    
    bddl_files = "\n\n".join(bddl_files) + "\n\n"

    return bddl_format_check.extract_bddl_files(bddl_files)
