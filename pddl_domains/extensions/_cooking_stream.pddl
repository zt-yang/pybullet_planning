(define (stream cooking)

  (:stream sample-pose-sprinkle
    :inputs (?o1 ?p1 ?o2)
    :domain (and (Region ?o1) (Pose ?o1 ?p1) (Graspable ?o2))
    :outputs (?p2)
    :certified (and (Pose ?o2 ?p2) (SprinklePose ?o1 ?p1 ?o2 ?p2))
  )

  (:stream test-cfree-pose-between
    :inputs (?o1 ?p1 ?o2 ?p2 ?o3 ?p3)
    :domain (and (Pose ?o1 ?p1) (Pose ?o2 ?p2) (Pose ?o3 ?p3))
    :certified (CFreePoseBetween ?o1 ?p1 ?o2 ?p2 ?o3 ?p3)
  )

)