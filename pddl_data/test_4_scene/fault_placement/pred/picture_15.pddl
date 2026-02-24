(define (problem picture_15)
    (:domain ariac)
    
    (:objects
table pump_placement regulator_placement battery_placement buffer_placement - location
        red_battery blue_battery red_regulator red_pump - part
    )
    
    (:init
(robot_at table)
        (handempty)        
        (clear red_battery)
        (clear blue_battery)
        (clear red_regulator)
        (clear red_pump)
        (part_at red_pump table)
        (part_at red_battery buffer_placement)
        (part_at blue_battery battery_placement)
        (part_at red_regulator table)
    )
    
    (:goal
(and
            (part_at red_pump pump_placement)
            (part_at red_regulator regulator_placement)
        )
    )
)