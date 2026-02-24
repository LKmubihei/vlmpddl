(define (problem picture_2)
    (:domain ariac)
    
    (:objects
table pump_placement regulator_placement battery_placement buffer_placement - location
        red_battery green_battery red_pump green_regulator - part
    )
    
    (:init
(robot_at table)
        (handempty)        
        (clear red_battery)
        (clear green_battery)
        (clear red_pump)
        (clear green_regulator)
        (part_at red_battery table)
        (part_at green_battery buffer_placement)
        (part_at red_pump pump_placement)
        (part_at green_regulator table)
    )
    
    (:goal
(and
            (part_at red_battery battery_placement)
            (part_at red_pump pump_placement)
            (part_at green_regulator regulator_placement)
        )
    )
)