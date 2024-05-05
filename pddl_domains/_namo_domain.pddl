(define (domain namo)
  (:predicates
    (Location ?o)
    (Cart ?o)
    (Marker ?o)
    (Marked ?o ?o2)

    (BConfInLocation ?q ?r)
    (PoseInLocation ?o ?p ?r)
    (InRoom ?o ?r)
    (RobInRoom ?r)

    (MarkerGrasp ?o ?g)
    (AtMarkerGrasp ?a ?o ?g)
    (HoldingMarker ?a ?o)
    (PulledMarker ?o)
    (GraspedMarker ?o)
    (SavedMarker ?o)

    (KinGraspMarker ?a ?o ?p ?g ?q ?t)
    (KinUngraspMarker ?a ?o ?p ?g ?q ?t)
    (KinPullMarkerToPose ?a ?o ?p1 ?p2 ?g ?q1 ?q2 ?o2 ?p3 ?p4 ?t)
  )


    (:action grasp_marker
      :parameters (?a ?o ?o2 ?p ?g ?q ?t)
      :precondition (and (Cart ?o) (Marker ?o2) (Marked ?o ?o2)
                         (KinGraspMarker ?a ?o2 ?p ?g ?q ?t)
                         (AtPose ?o2 ?p) (HandEmpty ?a) (AtBConf ?q)
                    )
      :effect (and (AtMarkerGrasp ?a ?o ?g) (CanPull ?a)
                   (AtMarkerGrasp ?a ?o2 ?g)
                   (not (HandEmpty ?a)) (not (CanMove))
                   (not (CanUngrasp)) ;;
                   (increase (total-cost) (PickCost))
              )
    )

  (:action ungrasp_marker
    :parameters (?a ?o ?o2 ?p ?g ?q ?t)
    :precondition (and (Cart ?o) (Marker ?o2) (Marked ?o ?o2) (AtPose ?o2 ?p)
                       (CanUngrasp) ;;
                       (KinUngraspMarker ?a ?o2 ?p ?g ?q ?t)
                       (AtMarkerGrasp ?a ?o ?g)
                       (AtMarkerGrasp ?a ?o2 ?g) (AtBConf ?q))
    :effect (and (HandEmpty ?a) (CanMove)
                 (not (AtMarkerGrasp ?a ?o ?g))
                 (not (AtMarkerGrasp ?a ?o2 ?g))
                 (GraspedMarker ?o2) ;;
                 (increase (total-cost) (PlaceCost)))
  )

    ;; to a sampled base position
    (:action pull_marker_to_pose
      :parameters (?a ?o ?p1 ?p2 ?g ?q1 ?q2 ?o2 ?p3 ?p4 ?t)
      :precondition (and (not (CanMove)) (CanPull ?a) (not (= ?p1 ?p2))
                         (Marker ?o) (Cart ?o2) (Marked ?o2 ?o)
                         (KinPullMarkerToPose ?a ?o ?p1 ?p2 ?g ?q1 ?q2 ?o2 ?p3 ?p4 ?t)
                         (AtPose ?o ?p1) (AtPose ?o2 ?p3) (AtBConf ?q1)
                         (AtMarkerGrasp ?a ?o ?g)
                         ;(not (UnsafeBTrajWithMarker ?t ?o))
                    )
      :effect (and (not (AtPose ?o ?p1)) (AtPose ?o ?p2) (PulledMarker ?o)
                   (not (AtPose ?o2 ?p3)) (AtPose ?o2 ?p4)
                   (AtBConf ?q2) (not (AtBConf ?q1))
                   (not (CanPull ?a)) (CanUngrasp)
                   (increase (total-cost) (MoveCost ?t))
              )
    )

  (:action magic
    :parameters (?o ?o2 ?p1 ?p3)
    :precondition (and (Marker ?o) (Cart ?o2) (Marked ?o2 ?o)
                       (AtPose ?o ?p1) (AtPose ?o2 ?p3))
    :effect (and (not (AtPose ?o ?p1)) (not (AtPose ?o2 ?p3)))
  )

  (:derived (HoldingMarker ?a ?o)
    (exists (?g) (and (Arm ?a) (Marker ?o) (MarkerGrasp ?o ?g)
                      (AtMarkerGrasp ?a ?o ?g)))
  )

  (:derived (RobInRoom ?r)
    (exists (?q) (and (BConfInLocation ?q ?r) (AtBConf ?q)))
  )
  (:derived (InRoom ?o ?r)
    (exists (?p) (and (PoseInLocation ?o ?p ?r) (AtPose ?o ?p)))
  )

)