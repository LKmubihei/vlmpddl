(define (problem picture_334)
    (:domain ariac)
    
    (:objects
table pump_placement regulator_placement battery_placement buffer_placement - location
         green_pump - part
    )
    
    (:init
(robot_at table)
        (handempty)        
        (clear green_pump)
        (part_at green_pump table)
    )
    
    (:goal
(and
            (part_at green_pump pump_placement)
        )
    )
)