(define (domain tea)
  (:requirements :typing :strips :universal-preconditions)
  (:types sth hand state)
  (:predicates
    (visible ?a - sth)
    (near ?a - sth)
    (holding ?a - sth ?b - hand)
    (contains ?a - sth ?b - sth) ; a contains b
    (torn ?a - sth)
    (closed ?a - sth)
    (opened ?a - sth)
    (over ?a - sth ?b - sth)
    (measured ?a)
    (steeped ?tea - sth)
    (used ?a - sth ?b - sth)

    ; external knowledge


    ; capabilities
    (can-scoop-using ?a - sth)
    (can-scoop-from ?a - sth)
    (can-turn-on ?a - sth)
    (can-contain-liquid ?a - sth)
    (can-measure-temperature ?a - sth)
    (can-measure-quantity ?a - sth)
    (can-be-used-to-stir ?a - sth)
    (is-rigid ?a - sth)
    (is-liquid ?a - sth)

    (is-container ?a - sth)

    (full ?a - sth)
    (is-rotated ?a - sth)
  )

  ; basic

  ; I haven't seen an object yet, and I need it
  (:action locate
    :parameters (?a - sth)
    :precondition (not (visible ?a))
    :effect (and (visible ?a) (near ?a)) ; for simplicity, everything is near you
  )

  ; ; I'm not close to an object - this should probably be provided by 3d memory
  ; (:action approach
  ;   :parameters (?a - sth)
  ;   :precondition (and (visible ?a) (not (near ?a)))
  ;   :effect (near ?a)
  ; )

  ; not holding an object
  (:action pick-up
    :parameters (?a - sth ?hand - hand)
    :precondition (and
        (visible ?a)
        (near ?a)
        (forall (?o - sth) (not (holding ?o ?hand)))
        ;(not (holding ?a ?hand))
        (not (is-liquid ?a))
    )
    :effect (holding ?a ?hand)
  )

  ; finished holding an object
  (:action put-down
    :parameters (?a - sth ?hand - hand)
    :precondition (holding ?a ?hand)
    :effect (and
        (near ?a)
        (not (holding ?a ?hand))
    )
  )

  ; put object inside another object
  (:action put-inside
    :parameters (?a - sth ?b - sth ?hand - hand)
    :precondition (and
        (holding ?a ?hand)
        (is-container ?b)
        (not (contains ?b ?a))
        (near ?b)
        (not (closed ?b))
    )
    :effect (and
        (contains ?b ?a)
        (not (holding ?a ?hand))
    )
  )

  ; measure and pour water

  ; hold something over something else (lets the planner get to the pour-into precondition (over ?a ?b) state)
  ;(:action hold-over
  ;  :parameters (?a - sth ?b - sth ?hand - hand)
  ;  :precondition (and
  ;      (holding ?a ?hand)
  ;      (near ?b)
  ;  )
  ;  :effect (and
  ;      (position-relationship ?a ?b "over")
  ;  )
  ;)

  ; pour something into something else
  ; assumption: (not modeling pour quantity)
  (:action pour-into
    :parameters (?a - sth ?b - sth ?contents - sth ?hand - hand)
    :precondition (and
        (holding ?a ?hand)
        ;(position-relationship ?a ?b "over")
        (contains ?a ?contents)
        (not (full ?b))
        ;(is-rotated ?a)
    )
    :effect (and
        (contains ?b ?contents)
        (not (contains ?a ?contents))
    )
  )

  (:action measure
    :parameters (?a - sth ?contents - sth ?hand - hand)
    :precondition (and
        (holding ?a ?hand)
        (contains ?a ?contents)
        (can-measure-quantity ?a)
    )
    :effect (and
        (measured ?contents)
    )
  )

  (:action measure-temperature
    :parameters (?instrument - sth ?item - sth)
    :precondition (can-measure-temperature ?instrument)
    :effect (and 
      (measured ?item)
      (used ?instrument ?item)
    )
  )

  (:action steep
    :parameters (?container - sth ?tea - sth)
    :precondition (and 
      (contains ?container ?tea)
      (exists (?l - sth) (and
          (is-liquid ?l)
          (contains ?container ?l)
      ))
    )
    :effect (steeped ?tea)
  )

  ; assuming pickup and put-down can handle tea bag !! will "holding" be able to capture that?
)