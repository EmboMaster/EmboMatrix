import bddl_format_check
# 示例使用
file_content = """
```bddl
(define (problem bring_the_pillow_from_the_shelf_in_empty_room_0_to_the_bed_in_empty_room_0_and_place_the_blanket_from_the_bed_in_empty_room_0_on_the_shelf)
    (:domain omnigibson)    
    (:objects
        pillow.n.01_1 - pillow.n.01
        shelf.n.01_1 - shelf.n.01
        bed.n.01_1 - bed.n.01
        blanket.n.01_1 - blanket.n.01
        floor.n.01_1 - floor.n.01
        agent.n.01_1 - agent.n.01
    )

    (:init
        (ontop pillow.n.01_1 shelf.n.01_1)
        (ontop blanket.n.01_1 bed.n.01_1)
        (inroom shelf.n.01_1 empty_room_0)
        (inroom bed.n.01_1 empty_room_0)
        (ontop agent.n.01_1 floor.n.01_1)
        (inroom floor.n.01_1 empty_room_0)
    )

    (:goal
        (and
            (ontop ?pillow.n.01_1 ?bed.n.01_1)
            (ontop ?blanket.n.01_1 ?shelf.n.01_1)
        )
    )
)
```
"""

pure_content = bddl_format_check.remove_first_and_last_lines(file_content)
print(pure_content)