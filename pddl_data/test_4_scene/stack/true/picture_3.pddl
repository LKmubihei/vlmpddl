(define (problem picture_378)
    (:domain ariac)
    
    (:objects
table pump_placement regulator_placement battery_placement buffer_placement - location
        green_battery blue_regulator green_pump - part
    )
    
    (:init
(robot_at table)
        (handempty)        
        (clear green_battery)
        (on green_battery blue_regulator)
        (clear green_pump)
        (part_at green_pump table)
        (part_at blue_regulator table)
    )
    
    (:goal
(and
            (part_at green_battery battery_placement)
            (part_at green_pump pump_placement)
            (part_at blue_regulator regulator_placement)
        )
    )
)