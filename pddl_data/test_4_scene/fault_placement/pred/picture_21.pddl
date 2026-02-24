(define (problem picture_21)
    (:domain ariac)
    
    (:objects
table pump_placement regulator_placement battery_placement buffer_placement - location
        red_pump green_regulator blue_battery orange_battery - part
    )
    
    (:init
(robot_at table)
        (handempty)        
        (clear green_regulator)
        (clear red_pump)
        (clear blue_battery)
        (clear orange_battery)
        (part_at green_regulator table)
        (part_at red_pump buffer_placement)
        (part_at blue_battery battery_placement)
        (part_at orange_battery battery_placement)
    )
    
    (:goal
(and
            (part_at red_pump pump_placement)
            (part_at green_regulator regulator_placement)
        )
    )
)