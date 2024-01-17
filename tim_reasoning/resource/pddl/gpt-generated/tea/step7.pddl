(define (problem tea-add-honey)
(:domain tea)

(:objects
  mug - sth
  honey - sth
  spoon - sth
  tea - sth
  right-hand - hand
)

(:init
  (contains mug tea)
)

(:goal
  (contains mug honey)
)
)