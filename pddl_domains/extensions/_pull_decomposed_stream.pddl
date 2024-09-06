(define (stream pull_decomposed)

    (:stream inverse-kinematics-grasp-handle
      :inputs (?a ?o ?p ?g)
      :domain (and (Controllable ?a) (Position ?o ?p) (HandleGrasp ?o ?g))
      :outputs (?q ?aq ?t)
      :certified (and (BConf ?q) (AConf ?a ?aq) (ATraj ?t)
                      (GraspHandle ?a ?o ?p ?g ?q ?aq)
                      (KinGraspHandle ?a ?o ?p ?g ?q ?aq ?t))
    )
    (:stream inverse-kinematics-ungrasp-handle
      :inputs (?a ?o ?p ?g ?q ?aq1)
      :domain (and (UngraspHandle ?a ?o ?p ?g ?q ?aq1))
      :outputs (?aq2 ?t)
      :certified (and (AConf ?a ?aq2) (ATraj ?t)
                      (KinUngraspHandle ?a ?o ?p ?g ?q ?aq1 ?aq2 ?t))
    )
    (:stream plan-base-pull-handle
      :inputs (?a ?o ?p1 ?p2 ?g ?q1 ?aq)
      :domain (and (GraspHandle ?a ?o ?p1 ?g ?q1 ?aq) (Position ?o ?p2) (IsSampledPosition ?o ?p1 ?p2)
                    (UnattachedJoint ?o))
      :outputs (?q2 ?bt)
      :certified (and (BConf ?q2) (UngraspBConf ?q2) (BTraj ?bt)
                      (UngraspHandle ?a ?o ?p2 ?g ?q2 ?aq)
                      (KinPullDoorHandle ?a ?o ?p1 ?p2 ?g ?q1 ?q2 ?bt ?aq))
    )
    (:stream plan-base-pull-handle-with-link
      :inputs (?a ?o ?p1 ?p2 ?g ?q1 ?aq ?l ?pl1)
      :domain (and (GraspHandle ?a ?o ?p1 ?g ?q1 ?aq) (Position ?o ?p2) (IsSampledPosition ?o ?p1 ?p2)
                    (JointAffectLink ?o ?l) (Pose ?l ?pl1))
      :outputs (?q2 ?bt ?pl2)
      :certified (and (BConf ?q2) (UngraspBConf ?q2) (BTraj ?bt)
                      (UngraspHandle ?a ?o ?p2 ?g ?q2 ?aq) (Pose ?l ?pl2)
                      (KinPullDoorHandleWithLink ?a ?o ?p1 ?p2 ?g ?q1 ?q2 ?bt ?aq ?l ?pl1 ?pl2))
    )


)