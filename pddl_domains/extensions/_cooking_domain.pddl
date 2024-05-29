(define (domain cooking)
  (:predicates

    (Region ?o)
    (Sprinkler ?o)
    (Food ?o)
    (SprinklePose ?o1 ?p1 ?o2 ?p2)
    (SprinkledTo ?o1 ?o2)
    (UnsafePoseBetween ?o1 ?p1 ?o2 ?p2)
    (CFreePoseBetween ?o1 ?p1 ?o2 ?p2 ?o3 ?p3)

    (KinNudgeGrasp ?a ?o ?p ?g ?q ?aq ?t)
    (KinNudgeDoor ?a ?o ?p1 ?p2 ?g ?q1 ?q2 ?aq)
    (NudgeGrasp ?o ?g)
    (NudgedDoor ?o)
  )

  (:action sprinkle
    ;; move o1 from default grasping arm conf to p1, which is above o2 at p2
    :parameters (?a ?o1 ?p1 ?o2 ?p2 ?g ?q ?t)
    :precondition (and (Kin ?a ?o1 ?p1 ?g ?q ?t) (Sprinkler ?o1) (Region ?o2)
                       (AtPose ?o2 ?p2) (AtGrasp ?a ?o1 ?g) (SprinklePose ?o2 ?p2 ?o1 ?p1) (AtBConf ?q)
                       (not (UnsafePose ?o1 ?p1))
                       (not (UnsafePoseBetween ?o1 ?p1 ?o2 ?p2))
                       (not (CanMove))
                   )
    :effect (and (SprinkledTo ?o1 ?o2)
                 (increase (total-cost) 1)
            )
  )

    (:action nudge_door
      :parameters (?a ?o ?p1 ?p2 ?g ?q1 ?q2 ?aq ?at ?bt)
      :precondition (and (Door ?o) (AtBConf ?q1)
                         (KinNudgeGrasp ?a ?o ?p ?g ?q1 ?aq ?at)
                         (KinNudgeDoor ?a ?o ?p1 ?p2 ?g ?q1 ?q2 ?aq)
                         (AtPosition ?o ?p1) (HandEmpty ?a)
                         (not (NudgedDoor ?o))
                    )
      :effect (and (not (AtPosition ?o ?p1)) (AtPosition ?o ?p2)
                   (not (AtBConf ?q1)) (AtBConf ?q2)
                   (NudgedDoor ?o)
                   (increase (total-cost) 1)
              )
    )

  (:derived (UnsafePoseBetween ?o1 ?p1 ?o2 ?p2)
    (exists (?o3 ?p3) (and (Pose ?o1 ?p1) (Pose ?o2 ?p2) (AtPose ?o3 ?p3)
                           (not (= ?o3 ?o1)) (not (= ?o3 ?o2)) (not (Food ?o3))
                           (not (CFreePoseBetween ?o1 ?p1 ?o2 ?p2 ?o3 ?p3)) ))
  )

)