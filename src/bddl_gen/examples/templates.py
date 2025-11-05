BDDL_EXAMPLES='''(define (problem preparing_clothes_for_the_next_day-0)
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
)'''