(define (problem picture_13)
    (:domain ariac)
    
    (:objects
table pump_placement regulator_placement battery_placement buffer_placement - location
        green_regulator_1 green_pump_1 - part
    )
    
    (:init
(robot_at table)
        (handempty)        
        (clear green_pump_1)
        (clear green_regulator_1)
        (part_at green_pump_1 table)
        (part_at green_regulator_1 table)
    )
    
    (:goal
(and
            (part_at green_pump_1 pump_placement)
            (part_at green_regulator_1 regulator_placement)
        )
    )
)