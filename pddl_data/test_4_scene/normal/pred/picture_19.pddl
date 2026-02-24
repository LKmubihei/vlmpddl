(define (problem picture_19)
    (:domain ariac)
    
    (:objects
table pump_placement regulator_placement battery_placement buffer_placement - location
        red_pump blue_pump green_regulator blue_battery - part
    )
    
    (:init
(robot_at table)
        (handempty)        
        (clear red_pump)
        (clear blue_pump)
        (clear blue_battery)
        (part_at red_pump table)
        (part_at blue_pump table)
        (part_at green_regulator buffer_placement)
        (part_at blue_battery battery_placement)
    )
    
    (:goal
(and
            (part_at red_pump pump_placement)
        )
    )
)