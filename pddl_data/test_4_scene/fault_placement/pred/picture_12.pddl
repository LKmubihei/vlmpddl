(define (problem picture_12)
    (:domain ariac)
    
    (:objects
table pump_placement regulator_placement battery_placement buffer_placement - location
        blue_battery red_pump green_pump green_regulator - part
    )
    
    (:init
(robot_at table)
        (handempty)        
        (clear green_pump)
        (clear blue_battery)
        (clear green_regulator)
        (clear red_pump)
        (part_at green_pump table)
        (part_at green_regulator table)
        (part_at blue_battery battery_placement)
        (part_at red_pump buffer_placement)
    )
    
    (:goal
(and
            (part_at green_pump pump_placement)
            (part_at green_regulator regulator_placement)
        )
    )
)