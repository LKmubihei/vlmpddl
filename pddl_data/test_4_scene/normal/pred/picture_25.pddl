(define (problem picture_25)
    (:domain ariac)
    
    (:objects
table pump_placement regulator_placement battery_placement buffer_placement - location
        green_battery red_pump yellow_regulator - part
    )
    
    (:init
(robot_at table)
        (handempty)        
        (clear green_battery)
        (clear red_pump)
        (clear yellow_regulator)
        (part_at green_battery table)
        (part_at red_pump pump_placement)
        (part_at yellow_regulator regulator_placement)
    )
    
    (:goal
(and
            (part_at green_battery battery_placement)
        )
    )
)