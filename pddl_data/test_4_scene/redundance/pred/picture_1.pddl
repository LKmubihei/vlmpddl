(define (problem picture_1)
    (:domain ariac)
    
    (:objects
table pump_placement regulator_placement battery_placement buffer_placement - location
        blue_battery red_pump green_regulator red_battery - part
    )
    
    (:init
(robot_at table)
        (handempty)        
        (clear blue_battery)
        (clear red_pump)
        (clear green_regulator)
        (part_at blue_battery table)
        (part_at red_pump table)
        (part_at green_regulator table)
        (part_at red_battery table)
    )
    
    (:goal
(and
            (part_at blue_battery battery_placement)
            (part_at red_pump pump_placement)
            (part_at green_regulator regulator_placement)
        )
    )
)