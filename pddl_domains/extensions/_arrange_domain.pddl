(define (domain arrange)
  (:predicates

    (Moved ?o)
    (Arrangeable ?o ?p ?r)
    (Stacked ?o ?r)

  )

  (:action arrange
    :parameters (?a ?o ?r ?p ?g ?q ?t)
    :precondition (and (Kin ?a ?o ?p ?g ?q ?t) (Graspable ?o) (Arrangeable ?o ?p ?r)
                       (AtGrasp ?a ?o ?g) (AtBConf ?q)
                       (not (UnsafePose ?o ?p))
                       (not (UnsafeApproach ?o ?p ?g))
                       (not (CanMove))
                       (not (Stacked ?o ?r))  ;; allow regrasping
                       ; (not (Placed ?o))  ;; allow regrasping
                       ; (not (UnsafeATraj ?t)) (not (UnsafeOTraj ?o ?g ?t))
                       )
    :effect (and (AtPose ?o ?p) (HandEmpty ?a) (CanPull ?a) (CanMove)
                 (not (AtGrasp ?a ?o ?g))
                 (Stacked ?o ?r) (Moved ?o)
                 (Placed ?o)
                 ; (increase (total-cost) (PlaceCost))
                 (increase (total-cost) 1)
            )
  )

  (:derived (Arrangeable ?o ?p ?r)
    (or (Supported ?o ?p ?r) (Contained ?o ?p ?r))
  )


)