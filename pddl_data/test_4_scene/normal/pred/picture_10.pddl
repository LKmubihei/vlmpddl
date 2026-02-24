(define (problem picture_10)
    (:domain ariac)
    
    (:objects
table pump_placement regulator_placement battery_placement buffer_placement - location
        red_battery  yellow_regulator red_pump - part
    )
    
    (:init
(robot_at table)
        (handempty)        
        (clear red_battery)
        (clear yellow_regulator)
        (clear red_pump)
        (part_at red_pump table)
        (part_at red_battery buffer_placement)
        (part_at yellow_regulator regulator_placement)
    )
    
    (:goal
(and
            (part_at red_pump pump_placement)
        )
    )
)