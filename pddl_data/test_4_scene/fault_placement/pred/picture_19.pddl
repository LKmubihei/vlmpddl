(define (problem picture_19)
    (:domain ariac)
    
    (:objects
table pump_placement regulator_placement battery_placement buffer_placement - location
        blue_battery red_pump blue_battery_2 red_battery - part
    )
    
    (:init
(robot_at table)
        (handempty)        
        (clear red_pump)
        (clear blue_battery)
        (clear blue_battery_2)
        (clear red_battery)
        (part_at red_pump table)
        (part_at blue_battery battery_placement)
        (part_at blue_battery_2 table)
        (part_at red_battery table)
    )
    
    (:goal
(and
            (part_at red_pump pump_placement)
        )
    )
)