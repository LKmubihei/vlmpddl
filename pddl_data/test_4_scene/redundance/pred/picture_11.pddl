(define (problem picture_11)
    (:domain ariac)
    
    (:objects
table pump_placement regulator_placement battery_placement buffer_placement - location
        green_battery red_battery blue_battery red_pump green_regulator green_regulator_2 - part
    )
    
    (:init
(robot_at table)
        (handempty)        
        (clear green_battery)
        (clear red_battery)
        (clear blue_battery)
        (clear green_regulator_2)
        (on green_regulator_2 red_pump)
        (part_at red_pump table)
        (part_at green_battery table)
        (part_at red_battery table)
        (part_at blue_battery battery_placement)
        (part_at green_regulator regulator_placement)
    )
    
    (:goal
(and
            (part_at red_pump pump_placement)
        )
    )
)