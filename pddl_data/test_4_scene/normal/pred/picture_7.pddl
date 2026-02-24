(define (problem picture_7)
    (:domain ariac)
    
    (:objects
table pump_placement regulator_placement battery_placement buffer_placement - location
        red_pump yellow_battery red_regulator - part
    )
    
    (:init
(robot_at table)
        (handempty)        
        (clear red_pump)
        (clear yellow_battery)
        (clear red_regulator)
        (part_at red_pump table)
        (part_at yellow_battery battery_placement)
        (part_at red_regulator pump_placement)
    )
    
    (:goal
(and
            (part_at red_regulator regulator_placement)
        )
    )
)