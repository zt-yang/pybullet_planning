(define (domain arrange)
  (:predicates

    (Arrangeable ?o ?p ?r)
    (Arranged ?o)

  )

  (:action arrange
    :parameters (?a ?o ?r ?p ?g ?q ?t)
    :precondition (and (Kin ?a ?o ?p ?g ?q ?t) (Graspable ?o) (Arrangeable ?o ?p ?r)
                       (AtGrasp ?a ?o ?g) (AtBConf ?q)
                       (not (UnsafePose ?o ?p))
                       (not (UnsafeApproach ?o ?p ?g))
                       (not (CanMove))
                       (not (Placed ?o))
                       ; (not (UnsafeATraj ?t)) (not (UnsafeOTraj ?o ?g ?t))
                       )
    :effect (and (AtPose ?o ?p) (HandEmpty ?a) (CanMove)
                 (not (AtGrasp ?a ?o ?g)) (Placed ?o) (Arranged ?o)
                 ; (increase (total-cost) (PlaceCost))
                 (increase (total-cost) 1)
            )
  )

  (:derived (Arrangeable ?o ?p ?r)
    (or (Supported ?o ?p ?r) (Contained ?o ?p ?r))
  )


)