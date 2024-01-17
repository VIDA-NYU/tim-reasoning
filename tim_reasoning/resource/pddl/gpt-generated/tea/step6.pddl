(define (problem tea-discard-teabag)
(:domain tea)

(:objects
  mug - sth
  tea - sth
  tea-bag - sth
  right-hand - hand
)

(:init
  (contains mug tea)
  (steeped tea)
)
  
(:goal
  (and
    (not (contains mug tea-bag)) 
    (contains mug tea)
  )
)
)