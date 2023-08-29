(define (problem tea-check-water-temp)
(:domain tea)

(:objects
  kettle - sth
  water - sth
  thermometer - sth
  right-hand - hand  
)

(:init
  (contains kettle water)
  (can-measure-temperature thermometer)
)

(:goal
  (
    (and
      (measured water)
      (used thermometer water)
    )
  )
)
)