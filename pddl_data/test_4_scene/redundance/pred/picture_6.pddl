(define (problem picture_6)
    (:domain ariac)
    
    (:objects
table pump_placement regulator_placement battery_placement buffer_placement - location
        blue_battery red_pump green_regulator blue_regulator green_regulator_2 - part
    )
    
    (:init
(robot_at table)
        (handempty)        
        (clear blue_battery)
        (clear green_regulator_2)
        (clear green_regulator)
        (clear blue_regulator)
        (on blue_battery red_pump)
        (part_at blue_regulator table)
        (part_at green_regulator table)
        (part_at green_regulator_2 table)
        (part_at red_pump table)
    )
    
    (:goal
(and
            (part_at blue_battery battery_placement)
            (part_at red_pump pump_placement)
            (part_at green_regulator regulator_placement)
        )
    )
)