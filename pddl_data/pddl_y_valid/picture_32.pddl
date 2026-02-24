(define (problem picture_32)
    (:domain ariac)
    
    (:objects
        table pump_placement regulator_placement battery_placement buffer_placement - location
        green_battery red_regulator blue_pump - part
    )
    
    (:init
        (robot_at table)
        (handempty)        
        (clear green_battery)
        (clear red_regulator)
        (clear blue_pump)
        (part_at red_regulator table)
        (on green_battery blue_pump)
        (part_at blue_pump table)
    )
    
    (:goal
        (and
            (part_at blue_pump pump_placement)
            (part_at red_regulator regulator_placement)
            (part_at green_battery battery_placement)
        )
    )
)
