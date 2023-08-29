(define (problem tea-measure-water-kettle)
(:domain tea)
(:objects
  water - sth
  water-bottle tea-bag honey - sth
  measuring-cup kettle mug thermometer spoon - sth
  left-hand - hand  
)

(:init
  (is-container water-bottle)
  (is-container measuring-cup) 
  (is-container kettle)
  (is-container mug)

  (can-measure-quantity measuring-cup)
  (can-measure-temperature thermometer)
  (can-be-used-to-stir spoon)

  (contains water-bottle water)
  (is-liquid water)
)

(:goal
  (and
    (contains kettle water)
    (measured water)
  )
)
)