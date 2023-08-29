(define (problem tea-steep-tea)  
(:domain tea)

(:objects
  mug - sth
  tea-bag - sth
  water - sth
)

(:init
  (contains mug tea-bag)
  (contains mug water)
)

(:goal
  (steeped tea-bag)
)
)