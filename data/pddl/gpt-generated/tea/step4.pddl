(define (problem tea-pour-water)
(:domain tea)

(:objects
  kettle - sth
  water - sth
  mug - sth
  tea-bag - sth
  right-hand - hand 
)

(:init
  (contains kettle water)
  (contains mug tea-bag)
  (not (full mug))
)

(:goal
  (and
    (not (contains kettle water))
    (contains mug water)
  )  
)
)