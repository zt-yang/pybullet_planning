(define (domain pull_decomposed)
  (:predicates

    (KinGraspHandle ?a ?o ?p ?g ?q ?aq ?t)  ;; grasp a handle
    (KinUngraspHandle ?a ?o ?p ?g ?q ?aq1 ?aq2 ?t)  ;; ungrasp a handle
    (KinPullDoorHandle ?a ?o ?p1 ?p2 ?g ?q1 ?q2 ?bt ?aq)  ;; pull the handle
    (KinPullDoorHandleWithLink ?a ?o ?p1 ?p2 ?g ?q1 ?q2 ?bt ?aq ?l ?lp1 ?lp2)  ;; pull the handle

  )

    (:action grasp_handle
      :parameters (?a ?o ?p ?g ?q ?aq1 ?aq2 ?t)
      :precondition (and (Joint ?o) (AConf ?a ?aq1) (CanGraspHandle) ; (CanUngrasp)
                         (AtPosition ?o ?p) (HandEmpty ?a)
                         (AtBConf ?q) (AtAConf ?a ?aq1)
                         (not (PulledOneAction ?o)) (not (Pulled ?o)) (CanPull ?a)
                         (KinGraspHandle ?a ?o ?p ?g ?q ?aq2 ?t)
                         (not (UnsafeATraj ?t))
                         ;(Enabled)
                    )
      :effect (and (AtHandleGrasp ?a ?o ?g) (not (HandEmpty ?a)) (not (CanPick))
                   (not (CanMove)) (CanPull ?a) (not (CanUngrasp)) (not (CanGraspHandle))
                   (not (AtAConf ?a ?aq1)) (AtAConf ?a ?aq2)
                   (Pulled ?o)
                   ;(increase (total-cost) (PickCost)) ; TODO: make one third of the cost
                   (increase (total-cost) 0)
              )
    )

    (:action ungrasp_handle
      :parameters (?a ?o ?p ?g ?q ?aq1 ?aq2 ?t)
      :precondition (and (Joint ?o) (AtPosition ?o ?p)
                         (KinUngraspHandle ?a ?o ?p ?g ?q ?aq1 ?aq2 ?t)
                         (AtHandleGrasp ?a ?o ?g) (CanUngrasp) (not (CanGraspHandle))
                         (AtBConf ?q) (UngraspBConf ?q) (AtAConf ?a ?aq1) ;; (DefaultAConf ?a ?aq2)
                         ;(Enabled)
                    )
      :effect (and (GraspedHandle ?o) (HandEmpty ?a) (CanMove) (CanPick) (CanGraspHandle)
                   (not (AtHandleGrasp ?a ?o ?g))
                   (not (AtAConf ?a ?aq1)) (AtAConf ?a ?aq2)
                   ;(increase (total-cost) (PlaceCost))
                   (increase (total-cost) 0)
              )
    )

    ;; from position ?p1 pull to the position ?p2
    (:action pull_handle
      :parameters (?a ?o ?p1 ?p2 ?g ?q1 ?q2 ?bt ?aq)
      :precondition (and (Joint ?o) (not (= ?p1 ?p2)) (CanPull ?a) ; (not (CanUngrasp))
                         (AtBConf ?q1) (AtAConf ?a ?aq)
                         (AtPosition ?o ?p1) (Position ?o ?p2) (AtHandleGrasp ?a ?o ?g)
                         (UnattachedJoint ?o)
                         (KinPullDoorHandle ?a ?o ?p1 ?p2 ?g ?q1 ?q2 ?bt ?aq)
                         ;(not (UnsafeApproach ?o ?p2 ?g))
                         ;(not (UnsafeATraj ?at))
                         ;(not (UnsafeBTraj ?bt))
                         ;(Enabled)
                    )
      :effect (and (not (CanPull ?a)) (CanUngrasp)
                  (AtPosition ?o ?p2) (not (AtPosition ?o ?p1))
                  (AtBConf ?q2) (not (AtBConf ?q1))
                  (increase (total-cost) 1)
              )
    )

    ;; from position ?p1 pull to the position ?p2, also affecting the pose of link attached to it
    (:action pull_handle_with_link
      :parameters (?a ?o ?p1 ?p2 ?g ?q1 ?q2 ?bt ?aq ?l ?lp1 ?lp2)
      :precondition (and (Joint ?o) (not (= ?p1 ?p2)) (CanPull ?a) ; (not (CanUngrasp))
                         (AtBConf ?q1) (AtAConf ?a ?aq)
                         (AtPosition ?o ?p1) (Position ?o ?p2) (AtHandleGrasp ?a ?o ?g)
                         (JointAffectLink ?o ?l) (AtPose ?l ?lp1) (Pose ?l ?lp2)
                         (KinPullDoorHandleWithLink ?a ?o ?p1 ?p2 ?g ?q1 ?q2 ?bt ?aq ?l ?lp1 ?lp2)
                         ; (not (UnsafeApproach ?o ?p2 ?g))
                         ; (not (UnsafeATraj ?at))
                         ; (not (UnsafeBTraj ?bt))
                         ; (Enabled)
                    )
      :effect (and (not (CanPull ?a)) (CanUngrasp)
                  (AtPosition ?o ?p2) (not (AtPosition ?o ?p1))
                  (AtBConf ?q2) (not (AtBConf ?q1))
                  (not (AtPose ?l ?lp1)) (AtPose ?l ?lp2)
                  (increase (total-cost) 1)
              )
    )

)