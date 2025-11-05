#python bddl_subtask_gen.py
from openai import OpenAI
import os

def bddl_subtask_gen(checked_files,path):

    os.environ['OPENAI_API_KEY'] = 'sk-AjgPUQzxcuKCscN3R0IPEru7G4hsAku16srLfzinmmn2AZKE'

    client = OpenAI(base_url="https://ai.sorasora.top/v1")
    completion = client.chat.completions.create(
        model="gpt-4o",
        messages = [
            {
            "role": "user",
            "content": [
                    {"type": "text", "text": '''The Behavior Domain Definition Language (BDDL) is a domain-specific language for defining activity scenarios in virtual environments, designed for the BEHAVIOR benchmark. Its syntax and structure include:
Problem Definition:
Each activity is defined as a problem.
Includes references to the :domain it belongs to (e.g., igibson).
Components:
:objects: Lists the objects involved, categorized by types (e.g., pool.n.01_1 - pool.n.01).
:init: Defines the initial state with ground literals, such as object locations or states (e.g., ontop, inroom).
:goal: Specifies the conditions that need to be satisfied for success, using logical expressions like and, not. The format of objects here needs a '?':1. (?boot.n.01 - boot.n.01) 2.(inside ?pebble.n.01_1 ?tank.n.02_1). The first type represents exist or ergodic, the second type represents the judgement of state.
Predicates:
Binary predicates (e.g., ontop, inside) specify spatial relationships or configurations.
Unary predicates (e.g., stained, cooked) describe individual object properties or states. 
Here are the predicates that are available,and some examples about how to use them.    
(cooked kabob.n.01_1)
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
(nextto ?firewood.n.01 ?fireplace.n.01_1)
(under ?gym_shoe.n.01 ?table.n.02_1)
(touching ?chair.n.01_1 ?bed.n.01_1)
(inside mug.n.04_1 cabinet.n.01_1)
(overlaid ?sheet.n.03_2 ?bed.n.01_1)
(attached light_bulb.n.01_1 table_lamp.n.01_1)
(draped ?long_trousers.n.01_1 ?shelf.n.01_1)
(insource vanilla__bottle.n.01_1 vanilla.n.02_1)
(inroom floor.n.01_1 living_room)
(broken mailbox.n.01_1)
                               

Here are the definitions of logical expression.
Conjunction (and): Logical "AND" operation, performing an "AND" operation on multiple subexpressions.
Disjunction (or): Logical "OR" operation, performing an "OR" operation on multiple subexpressions.
Negation (not): Logical "NOT" operation, performing a "NOT" operation on a single subexpression.
Implication (imply): Logical "IMPLY" operation, representing "if the premise holds, then the conclusion holds."
Universal (forall): Universal quantifier operation, representing that a condition holds for all objects within a given range.
Existential (exists): Existential quantifier operation, representing that a condition holds for some objects within a given range.
NQuantifier (forn): Represents "exactly N" elements that satisfy a given condition.
ForPairs (forpairs): Quantification over pairs of objects, representing that the condition holds for all object pairs.
ForNPairs (fornpairs): Represents that exactly N pairs of objects satisfy a given condition.

Here are a examples of bddl files.
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

Please divide this long-term task into the following four subtasks:
1. Navigation: Locate and reach the target object in the environment.
2. Grasp: Securely grasp the target object and maintain a stable hold.
3. Carry: Transport the object to the designated location (typically after grasping).
4. Put: Place the object in the specified position (typically after carrying).
Taking above preparing_clothes_for_the_next_day as an example, the task can be divied into following subtasks:

1. Find coat. (Navigation task)
2. Grasp coat. (Grasp task)
3. Carry the coad to the bed. (Carry task)
4. Put coat ontop of the bed. (Put task)
5. Find trouser. (Navigation task)
6. Grasp trouser. (Grasp task)
7. Carry the coad to the bed. (Carry task)
8. Put trouser ontop of the bed. (Put task)
9. Find left boot. (Navigation task)
10. Grasp left boot. (Grasp task)
11. Carry the left boot to childs' room. (Carry task)
12. Put left boot ontop of the floor in childs' room. (Put task)
13. Find right boot. (Navigation task)
14. Grasp right boot. (Grasp task)
15. Carry the right boot to find childs' room. (Carry task)
16. Put right boot ontop of the floor in childs' room. (Put task)

ATTENTION: Our robot is a robot with four wheel, a cloud terrace and a robot arm. The subtask should be as simple as you can. For example, "Water the plant" is not simple enough for the agent to train, so we need to describe it as "find watering can","grasp watering can","find plant","incline the watering can".
Then, you need to create bddl "goal" description for each subtasks. For example, the first two subtasks
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
    (:subgoal1 Navigation task:Find coat.
        (nextto ?coat.n.01_1 ?agent.n.01_1)
    (:subgoal2 Grasp task:Grasp coat.
        (attached ?coat.n.01_1 ?agent.n.01_1)
    ......
    (:subgoal15 Carry task:Carry right boot to floor in childs' room.
        (nextto ?floor.n.01_2 ?agent.n.01_1)
    (:subgoal16 Put task:Put right boot ontop of the floor in childs' room.
        (ontop ?boot.n.01_2 ?floor.n.01_2)
        
    The format for each subgoal should be:
    (:subgoal{n} {task type}:{instruction}. 
        ({relation} ?{object1} ?{object2})
    (Use States[cooked, frozen, open, folded,unfolded, toggled_on, hot, on_fire, future, real, saturated, covered, filled, contains,ontop, nextto, under, touching,inside,overlaid, attached,draped,insource,broken.] to represent the subtask goal.)
To be note that if you want to put the object on table,you should carry it to the table,so the agent should next to the table.
Now please split following bddl task into subtasks.'''
+checked_files+
'''To be note that you shold output a complete file and only contains task definition without any annotations.'''},
            ],
            }
        ],
        temperature=0.7,
        max_tokens=10000,
    )
    response = completion.choices[0].message.content
    print(response)
    filename = f'{path}/txt/subtask_bddl.txt'
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, 'w', encoding='utf-8') as file:
        file.write(response)
    return response

def bddl_subtask_fix(checked_files,reply_from_gpt,path):

    os.environ['OPENAI_API_KEY'] = 'sk-AjgPUQzxcuKCscN3R0IPEru7G4hsAku16srLfzinmmn2AZKE'

    client = OpenAI(base_url="https://ai.sorasora.top/v1")
    completion = client.chat.completions.create(
        model="gpt-4o",
        messages = [
            {
            "role": "user",
            "content": [
                    {"type": "text", "text": '''The Behavior Domain Definition Language (BDDL) is a domain-specific language for defining activity scenarios in virtual environments, designed for the BEHAVIOR benchmark. Its syntax and structure include:
Problem Definition:
Each activity is defined as a problem.
Includes references to the :domain it belongs to (e.g., igibson).
Components:
:objects: Lists the objects involved, categorized by types (e.g., pool.n.01_1 - pool.n.01).
:init: Defines the initial state with ground literals, such as object locations or states (e.g., ontop, inroom).
:goal: Specifies the conditions that need to be satisfied for success, using logical expressions like and, not. The format of objects here needs a '?':1. (?boot.n.01 - boot.n.01) 2.(inside ?pebble.n.01_1 ?tank.n.02_1). The first type represents exist or ergodic, the second type represents the judgement of state.
Predicates:
Binary predicates (e.g., ontop, inside) specify spatial relationships or configurations.
Unary predicates (e.g., stained, cooked) describe individual object properties or states. 
Here are the predicates that are available,and some examples about how to use them.    
(cooked kabob.n.01_1)
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
(nextto ?firewood.n.01 ?fireplace.n.01_1)
(under ?gym_shoe.n.01 ?table.n.02_1)
(touching ?chair.n.01_1 ?bed.n.01_1)
(inside mug.n.04_1 cabinet.n.01_1)
(overlaid ?sheet.n.03_2 ?bed.n.01_1)
(attached light_bulb.n.01_1 table_lamp.n.01_1)
(draped ?long_trousers.n.01_1 ?shelf.n.01_1)
(insource vanilla__bottle.n.01_1 vanilla.n.02_1)
(inroom floor.n.01_1 living_room)
(broken mailbox.n.01_1)
                              

Here are the definitions of logical expression.
Conjunction (and): Logical "AND" operation, performing an "AND" operation on multiple subexpressions.
Disjunction (or): Logical "OR" operation, performing an "OR" operation on multiple subexpressions.
Negation (not): Logical "NOT" operation, performing a "NOT" operation on a single subexpression.
Implication (imply): Logical "IMPLY" operation, representing "if the premise holds, then the conclusion holds."
Universal (forall): Universal quantifier operation, representing that a condition holds for all objects within a given range.
Existential (exists): Existential quantifier operation, representing that a condition holds for some objects within a given range.
NQuantifier (forn): Represents "exactly N" elements that satisfy a given condition.
ForPairs (forpairs): Quantification over pairs of objects, representing that the condition holds for all object pairs.
ForNPairs (fornpairs): Represents that exactly N pairs of objects satisfy a given condition.

Here are a examples of bddl files.
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

Please divide this long-term task into the following four subtasks:
1. Navigation: Locate and reach the target object in the environment.
2. Grasp: Securely grasp the target object and maintain a stable hold.
3. Carry: Transport the object to the designated location (typically after grasping).
4. Put: Place the object in the specified position (typically after carrying).
Taking above preparing_clothes_for_the_next_day as an example, the task can be divied into following subtasks:

1. Find coat. (Navigation task)
2. Grasp coat. (Grasp task)
3. Carry the coad to the bed. (Carry task)
4. Put coat ontop of the bed. (Put task)
5. Find trouser. (Navigation task)
6. Grasp trouser. (Grasp task)
7. Carry the coad to the bed. (Carry task)
8. Put trouser ontop of the bed. (Put task)
9. Find left boot. (Navigation task)
10. Grasp left boot. (Grasp task)
11. Carry the left boot to childs' room. (Carry task)
12. Put left boot ontop of the floor in childs' room. (Put task)
13. Find right boot. (Navigation task)
14. Grasp right boot. (Grasp task)
15. Carry the right boot to find childs' room. (Carry task)
16. Put right boot ontop of the floor in childs' room. (Put task)

ATTENTION:The subtask should be as simple as you can. For example, "Water the plant" is not simple enough for the agent to train, so we need to describe it as "find watering can","grasp watering can","find plant","incline the watering can".
Then, you need to create bddl "goal" description for each subtasks. For example, the first two subtasks
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
    (:subgoal1 Navigation task:Find coat.
        (nextto ?coat.n.01_1 ?agent.n.01_1)
    (:subgoal2 Grasp task:Grasp coat.
        (attached ?coat.n.01_1 ?agent.n.01_1)
    ......
    (:subgoal15 Carry task:Carry right boot to floor in childs' room.
        (nextto ?floor.n.01_2 ?agent.n.01_1)
    (:subgoal16 Put task:Put right boot ontop of the floor in childs' room.
        (ontop ?boot.n.01_2 ?floor.n.01_2)
        
    The format for each subgoal should be:
    (:subgoal{n} {task type}:{instruction}. 
        ({relation} ?{object1} ?{object2})
    (Use States[cooked, frozen, open, folded,unfolded, toggled_on, hot, on_fire, future, real, saturated, covered, filled, contains,ontop, nextto, under, touching,inside,overlaid, attached,draped,insource.] to represent the subtask goal.)

To be note that if you want to put the object on table,you should carry it to the table,so the agent should next to the table.
Now we will give you some bddl files of which the subtasks are given, also we will give you a review of these split subtasks, you should rewrite the subtasks that are (Can't define) to Navigation,Grasp,Carry,Put(If you think they can be rewrited).For those tasks that are all defined, you don't have to change them, just output them. 
After the steps are some suggestion(if there aren't suggestions,just ignore this) about modifying the subtasks.
Remember, the relation between the two object can only be[cooked, frozen, open, folded,unfolded, toggled_on, hot, on_fire, future, real, saturated, covered, filled, contains,ontop, nextto, under, touching,inside,overlaid, attached,draped,insource.]
Here is the bddl files'''
+checked_files+'Here is the review of the subtasks:'+reply_from_gpt+
'''To be note that you shold output a complete file and only contains task definition without any annotations.'''},
            ],
            }
        ],
        temperature=0.7,
        max_tokens=10000,
    )
    response = completion.choices[0].message.content
    print(response)
    filename = f'{path}/txt/subtask_bddl_fixed.txt'
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, 'w', encoding='utf-8') as file:
        file.write(response)
    return response