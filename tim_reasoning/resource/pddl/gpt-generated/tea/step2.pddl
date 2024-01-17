(define (problem tea-place-teabag)
(:domain tea)

(:objects
  tea-bag - sth
  mug - sth
  right-hand - hand
)

(:init
  (is-container mug)
)

(:goal
  (contains mug tea-bag)
)
)