COMMEND_GEN_PROMPT='''{social_description} You need to generate reasonable commands for the social members.The requirements for the commands are as follows:
1. The command must be broken down into the robot's subtasks as follows: pick up something, place something, open something, close something, toggle on appliance, toggle off appliance, go to room, put something on top of something, put something inside something, heat something, cook something, freeze something, unfreeze something, slice something, soak something, dry something, clean something. Make sure the commands can cover more subtasks like clean.
2. The objects and locations involved in the command should not exceed eight.
3. The commands should cover different needs of the four family members.
4. The completion criteria of the commands must be clear, such as ensuring an object is placed in a specific location or on something. Avoid commands where the result is unclear.
5. The commands should not exceed the mobile robot's capabilities. The robot can only: Move to a nearby location around an object, Turn by a specific angle, Pick up an object, Place an object, Move forward to a specific location, open something, close something, toggle on appliance, toggle off appliance, go to room, put something on top of something, put something inside something, heat something, cook something, freeze something, unfreeze something, slice something, soak something, dry something, clean something. The robot can't hang something or sort something or water plants, it can only perform simple tasks.
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
20. If you need to use objects to be a cleaner, you can use items in the list :{cleaner_object}.
21. If you need to use objects to be sliced, you can use items in the list :{cookable_object} and there must be a knife in the scene, which you need to add it in the objects section and init section.
22. The pick and place task don't need to consider whether the object is openable, toggleable, cookable, freezable or not. The robot can pick up and place any object. But the openable, toggleable, cookable, freezable task need to consider whether the object is openable, toggleable, cookable, freezable or not.
23. If you want to cook or freeze something, the scene needs to have a heatsource or a coldsource objects.
24. Each command needs to be diverse, involving operating different objects and rooms. But don't force to be diverse, for example, cook something or freeze something is not reasonable in a school gym scene while in a kitchen is reasonable to cook or freeze something.The commands must be complex enough.
25. The commands can use number to describe objects, like "clean all 3 plates on the table in the dining room_0" or "put all the four basketballs in the basket in the living room_0". This will increase the diversity of our task. Make sure to have more task in this format.

Please generate {amount} command. And make sure they are all reasonable fot the social members.Generate more tasks about slicing and cleaning, which involving states like sliced, soaked, dusty and stained.{extra_requirment}
Output in the format like:
1.Analyse the social_description, find out what kind of command is needed and reasonable. And what kinds of objects are reasonable to appear in this scene. Then how can these objects be operated by humans and robots. For example, we won't heat food in the bathroom, instead we will heat food in the kitchen. We won't put a bed in the kitchen, instead we will put a bed in the bedroom. We won't put a refrigerator in the living room, instead we will put a refrigerator in the kitchen. We won't put a desk_phone in the bathroom, instead we will put a desk_phone in the bedroom. Generate more task with clean actions.
2.Analyse the objects we may need for the commands, where they are supposed to be, and how to operate them. If you need to slice something, you need to have a knife in the scene.{knife_object}.
3.Generate the command, and make sure the command is reasonable and diverse.
ALL commands MUST be included by [[]] altogether, which will make it easier to parse. For example, [[
command1
command2
command3]]. all the commands you generated must be included in a single [[]].You should give a list of command which means no any extra punctuations and extra indexs.'''

SOCIAL_CHARACTER_GEN_PROMPT='''I need you to help me to generate a social description to a scene description, Here is an example. This is a description of a scene:"A multi-room office space featuring numerous cubicles, private offices, a conference hall, a meeting room, a lobby, and a copy room, equipped with various office furniture and equipment.". Generate a social description of this scene, such as:"In this scene, there is a family of four: a father, a mother, a 14-year-old girl, and an 8-year-old boy. They have a mobile robot (composed of a mobile platform and a simple gripper) as their household assistant. These five characters will give simple commands to the robot throughout the day." The social description should be related to the scene and there must be more than two humanbeings and only one robot to serve the humans.
                     Here is an other example:
                     scene description:A school scene featuring 7 rooms, including classrooms and corridors, with numerous desks, chairs, lockers, maps, and educational tools.
                     social description:In this scene, there is a dedicated teacher and a diligent school administrator, who work together to ensure the smooth operation of the school. They are assisted by a single educational robot that navigates through the classrooms and corridors. 
                     Here is a scene description:{scene_description}. This is the room list you can use :{room_description}, Please MARK:the generated social description can't include other rooms, only the rooms in the room list can be used to Generate the social description. Generate the social description.'''

SOCIAL_CHARACTER_GEN_PROMPT_IN_DETAIL='''I need you to help me generate a social description for a simulated scene.

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

                Your goal is to create a similar detailed description for this room list: {room_description}. Remember: each room must appear. Make it vivid, grounded, and specific. Only output the description. Do not include any explanation or list.'''

TASK_GENERATION_PROMPT='''The Behavior Domain Definition Language (BDDL) is a domain-specific language for defining activity scenarios in virtual environments, designed for the BEHAVIOR benchmark. Its syntax and structure include:
            1. Problem Definition:
            - Each activity is defined as a problem.
            - Includes references to the :domain it belongs to (e.g., omnigibson).
            2. Components:
            - :objects: Lists the objects involved, categorized by types (e.g., pool.n.01_1 - pool.n.01).task-relevant objects, where each line represents a WordNet synset of the object. For example, candle.n.01_1 candle.n.01_2 candle.n.01_3 candle.n.01_4 - candle.n.01 indicates that four objects that belong to the candoe.n.01 synset are needed for this task.The object you choose should not be abstract(eg:ground),it should be concrete(eg:floor).The name of the room should not be listed in the object.(eg:kitchen)
            - :init: Defines the initial state with ground literals, such as object locations or states (e.g., ontop, inroom).initial conditions of the task, where each line represents a ground predicate that holds at the beginning of the task. For example, (ontop candle.n.01_1 table.n.02_1) indicates that the first candle is on top of the first table when the task begins.
            - :goal: Specifies the conditions that need to be satisfied for success, usingS logical expressions like and, not.goal conditions of the task, where each line represents a ground predicate and each block represents a non-ground predicate (e.g. forall, forpairs, and, or, etc) that should hold for the task to be considered solved. For example, (inside ?candle.n.01 ?wicker_basket.n.01) indicates that the candle should be inside the wicker basket at the end of the task.The format of objects here needs a '?' when :1. (?boot.n.01 - boot.n.01) 2.(inside ?pebble.n.01_1 ?tank.n.02_1). The first type represents exist or ergodic, the second type represents the judgement of state.
            3. Predicates:
            - Binary predicates (e.g., ontop, inside) specify spatial relationships or configurations.
            - Unary predicates (e.g., stained, cooked) describe individual object properties or states.    - Here are the predicates that are available,and some examples about how to use them.    
            (cooked kabob.n.01_1)
            (frozen chicken.n.01_1)
            (open window.n.01_1)
            (unfolded dress.n.01_1)
            (toggled_on air_conditioner.n.01_1)
            (hot potato.n.01_1)
            (on_fire firewood.n.01)
            (real cheese_tart.n.01_1)
            (contains bowl.n.01_1 white_rice.n.01_1)
            (ontop agent.n.01_1 floor.n.01_1)
            (under gym_shoe.n.01 table.n.02_1)
            (inside mug.n.04_1 cabinet.n.01_1)
            (inroom floor.n.01_1 living_room)
            (soaked rag.n.01_1)
            (stained grill.n.01_1)
            (dusty sweater.n.01_1)
            PLEASE PAY ATTENTION:All the predicates you use must come from above.for example, you shouldn't use predicates like (closed ?closet.n.01_1),instead you can use the given predicates like (not\n (open ?closet.n.01_1)\n) to describe this kind of situation.    

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
                         
(define (problem cleaning_barbecue_grill_0)
    (:domain omnigibson)

    (:objects
        grill.n.02_1 - grill.n.02
        sink.n.01_1 - sink.n.01
        rag.n.01_1 - rag.n.01
        floor.n.01_1 - floor.n.01
        agent.n.01_1 - agent.n.01
        bucket.n.01_1 - bucket.n.01
        table.n.02_1 - table.n.02
    )
    
    (:init 
        (ontop grill.n.02_1 floor.n.01_1) 
        (ontop rag.n.01_1 table.n.02_1) 
        (ontop bucket.n.01_1 table.n.02_1)
        (stained grill.n.02_1) 
        (ontop agent.n.01_1 floor.n.01_1) 
        (dusty grill.n.02_1) 
        (inroom floor.n.01_1 garage_0)
        (inroom table.n.02_1 garage_0)
    )
    
    (:goal 
        (and 
            (not 
                (stained ?grill.n.02_1)
            ) 
            (not 
                (dusty ?grill.n.02_1)
            ) 
        )
    )
)
                         
(define (problem assembling_gift_baskets-0)
    (:domain omnigibson)

    (:objects
        basket.n.01_1 basket.n.01_2 basket.n.01_3 basket.n.01_4 - basket.n.01
        floor.n.01_1 - floor.n.01
        candle.n.01_1 candle.n.01_2 candle.n.01_3 candle.n.01_4 - candle.n.01
        cookie.n.01_1 cookie.n.01_2 cookie.n.01_3 cookie.n.01_4 - cookie.n.01
        cheese.n.01_1 cheese.n.01_2 cheese.n.01_3 cheese.n.01_4 - cheese.n.01
        bow.n.08_1 bow.n.08_2 bow.n.08_3 bow.n.08_4 - bow.n.08
        table.n.02_1 table.n.02_2 - table.n.02
        agent.n.01_1 - agent.n.01
    )

    (:init
        (ontop basket.n.01_1 floor.n.01_1)
        (ontop basket.n.01_2 floor.n.01_1)
        (ontop basket.n.01_3 floor.n.01_1)
        (ontop basket.n.01_4 floor.n.01_1)
        (ontop candle.n.01_1 table.n.02_1)
        (ontop candle.n.01_2 table.n.02_1)
        (ontop candle.n.01_3 table.n.02_1)
        (ontop candle.n.01_4 table.n.02_1)
        (ontop cookie.n.01_1 table.n.02_1)
        (ontop cookie.n.01_2 table.n.02_1)
        (ontop cookie.n.01_3 table.n.02_1)
        (ontop cookie.n.01_4 table.n.02_1)
        (ontop cheese.n.01_1 table.n.02_2)
        (ontop cheese.n.01_2 table.n.02_2)
        (ontop cheese.n.01_3 table.n.02_2)
        (ontop cheese.n.01_4 table.n.02_2)
        (ontop bow.n.08_1 table.n.02_2)
        (ontop bow.n.08_2 table.n.02_2)
        (ontop bow.n.08_3 table.n.02_2)
        (ontop bow.n.08_4 table.n.02_2)
        (ontop agent.n.01_1 floor.n.01_1)
        (inroom floor.n.01_1 living_room)
    )

    (:goal
        (and
            (forpairs (?basket.n.01 - basket.n.01) (?candle.n.01 - candle.n.01)
                (inside ?candle.n.01 ?basket.n.01)
            )
            (forpairs (?basket.n.01 - basket.n.01) (?cheese.n.01 - cheese.n.01)
                (inside ?cheese.n.01 ?basket.n.01)
            )
            (forpairs (?basket.n.01 - basket.n.01) (?cookie.n.01 - cookie.n.01)
                (inside ?cookie.n.01 ?basket.n.01)
            )
            (forpairs (?basket.n.01 - basket.n.01) (?bow.n.08 - bow.n.08)
                (inside ?bow.n.08 ?basket.n.01)
            )
        )
    )
)
                         
(define (problem cleaning_windows_with_rags-0)
    (:domain omnigibson)

    (:objects
        towel.n.01_1 towel.n.01_2 - towel.n.01
        cabinet.n.01_1 - cabinet.n.01
        rag.n.01_1 rag.n.01_2 - rag.n.01
        window.n.01_1 window.n.01_2 - window.n.01
        sink.n.01_1 - sink.n.01
        table.n.02_1 - table.n.02
        floor.n.01_1 floor.n.01_2 - floor.n.01
        agent.n.01_1 - agent.n.01
        
    )

    (:init
        (inside towel.n.01_1 cabinet.n.01_1)
        (inside towel.n.01_2 cabinet.n.01_1)
        (inside rag.n.01_1 cabinet.n.01_1)
        (inside rag.n.01_2 cabinet.n.01_1)
        (not (soaked rag.n.01_1))
        (not (soaked rag.n.01_2))
        (dusty window.n.01_1)
        (dusty window.n.01_2)
        (not (dusty sink.n.01_1))
        (ontop agent.n.01_1 floor.n.01_1)
        (inroom floor.n.01_1 kitchen)
    )

    (:goal
        (and
            (soaked rag.n.01_1)
            (soaked rag.n.01_2)
            (not (dusty window.n.01_1))
            (not (dusty window.n.01_2))
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
            10. If something is sliced, there must be a knife in the scene, which you need to add it in the objects section and init section. And check the final goal in the format of (sliced ?object.n.01_1) in the goal section. Not add a new object like sliced_tomato.n.01_1 in the goal section. Check the state of the sliced object, not create a new object.
            You can only use rooms as follows:{roomids}, the bddl file you generate can't exceed the range. {error_hint}             
            Now please output {input} for me. If the tasks don't emphasize the object is on the floor, then don't need to add the object floor, just use the inroom predicates to init the objects. However, the agent need to be init ontop of a floor, so you need at least one floor object to place the agent. Please remember the '?' in the goal section is a must when the object is a particular object instead of an object type. The task name is the command sentence which will give the full information of the task, don't simplify its content, just change the format
            Output in the format like:
            1.Analyse the task, find out the objects and the relationships between them. Make sure you know all the objects and their relationships in the task.
            2.Analyse the init states of the objects and the relationships between them. Make sure you know all the init states of the objects and their relationships in the task. The cleaner object like rag and brush don't need to be dusty or stained in the init section, you can just use the soaked predicate to describe the rag is soaked in the goal section. The dusty or stained object should be in the init section, and you need to check the dusty or stained predicate in the goal section. If the task is to clean the object, you need to add a sink.n.01 to the scene to make sure there is watersource.
            3.Analyse the goal states of the objects and the relationships between them. Make sure you know all the goal states of the objects and their relationships in the task. Don't add new objects in the goal section, just use the objects in the objects section. If you need to clean something, the cleaner need to be soaked in the goal section and the object to be cleaned need to be not stained or dusty in the goal section. Don't need to check whether the dirty object is soaked or not.
            4.Analyse the predicates of the objects and the relationships between them. Make sure you know all the predicates of the objects and their relationships in the task.
            5.Check whether the task involves cooking or freezing. If so, check the rule 9, which means if the cooked or frozen object is picked up, you need to check if the object is cooked or frozen in the goal section and do not check the heater or cooler is toggled on or the object is inside the heater or cooler.If the object is not picked up after cooked or frozen, you need to check the heater is toggled on and the object is inside the heater or cooler in the goal section.
            6.Generate the bddl file according to the above analysis. The bddl file should be in the format of the BDDL file, and the objects and their relationships should be in the format of the BDDL file. The predicates of the objects and their relationships should be in the format of the BDDL file.'''
                