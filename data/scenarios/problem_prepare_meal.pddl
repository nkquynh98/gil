(define (problem prepare_meal)
    (:domain meal)
    (:init
        (on disk shelf1)
        (on cup shelf2)
        (free robot)
        (avoid_human robot)
        (at robot shelf1)
    )
    (:goal (and
        (on disk table)
        (on cup table)
    ))
)