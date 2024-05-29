(define (stream rim-grasp)

    (:stream sample-rim-grasp
      :inputs (?o)
      :domain (Joint ?o)
      :outputs (?g)
      :certified (RimGrasp ?o ?g)
    )

    (:action grasp_handle
      :parameters (?a ?o ?p ?g ?q ?aq1 ?aq2 ?t)
      :precondition (and (Joint ?o) (AConf ?a ?aq1) (CanGraspHandle) ; (CanUngrasp)
                         (KinGraspHandle ?a ?o ?p ?g ?q ?aq2 ?t)
                         (AtPosition ?o ?p) (HandEmpty ?a)
                         (AtBConf ?q) (AtAConf ?a ?aq1)
                         ;(Enabled)
                    )
      :effect (and (AtHandleGrasp ?a ?o ?g) (not (HandEmpty ?a)) (not (CanPick))
                   (not (CanMove)) (CanPull ?a) (not (CanUngrasp)) (not (CanGraspHandle))
                   (not (AtAConf ?a ?aq1)) (AtAConf ?a ?aq2)
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
      :precondition (and (Joint ?o) (not (= ?p1 ?p2)) (CanPull ?a) (UnattachedJoint ?o) ; (not (CanUngrasp))
                         (AtPosition ?o ?p1) (Position ?o ?p2) (AtHandleGrasp ?a ?o ?g)
                         (KinPullDoorHandle ?a ?o ?p1 ?p2 ?g ?q1 ?q2 ?bt ?aq)
                         (AtBConf ?q1) (AtAConf ?a ?aq)
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

)