(define (stream pr2-tamp)
  (:stream sample-pose
    :inputs (?o ?r)
    :domain (Stackable ?o ?r)
    :outputs (?p)
    :certified (and (Pose ?o ?p) (Supported ?o ?p ?r))
  )
  (:stream sample-pose-inside
    :inputs (?o ?r)
    :domain (Containable ?o ?r)
    :outputs (?p)
    :certified (and (Pose ?o ?p) (Contained ?o ?p ?r))
  )

  (:stream sample-grasp
    :inputs (?o)
    :domain (Graspable ?o)
    :outputs (?g)
    :certified (Grasp ?o ?g)
  )

  (:stream inverse-reachability
    :inputs (?a ?o ?p ?g)
    :domain (and (Controllable ?a) (Pose ?o ?p) (Grasp ?o ?g))
    :outputs (?q)
    :certified (and (BConf ?q) (Reach ?a ?o ?p ?g ?q))
  )
  (:stream inverse-kinematics
    :inputs (?a ?o ?p ?g ?q)
    :domain (Reach ?a ?o ?p ?g ?q)
    :fluents (AtPose AtPosition)
    :outputs (?t)
    :certified (and (Kin ?a ?o ?p ?g ?q ?t))
  )

  (:stream plan-base-motion
    :inputs (?q1 ?q2)
    :domain (and (BConf ?q1) (BConf ?q2))
    :fluents (AtPose AtPosition AtAConf)
    :outputs (?t)
    :certified (and (BTraj ?t) (BaseMotion ?q1 ?t ?q2))
  )

  (:stream test-cfree-pose-pose
    :inputs (?o1 ?p1 ?o2 ?p2)
    :domain (and (Pose ?o1 ?p1) (Pose ?o2 ?p2))
    :certified (CFreePosePose ?o1 ?p1 ?o2 ?p2)
  )
  (:stream test-cfree-approach-pose
    :inputs (?o1 ?p1 ?g1 ?o2 ?p2)
    :domain (and (Pose ?o1 ?p1) (Grasp ?o1 ?g1) (Pose ?o2 ?p2))
    :certified (CFreeApproachPose ?o1 ?p1 ?g1 ?o2 ?p2)
  )
  (:stream get-joint-position-open
    :inputs (?o ?p1)
    :domain (and (Joint ?o) (Position ?o ?p1) (IsClosedPosition ?o ?p1))
    :outputs (?p2)
    :certified (and (Position ?o ?p2) (IsOpenedPosition ?o ?p2) (IsSampledPosition ?o ?p1 ?p2))
  )

    (:stream sample-handle-grasp
      :inputs (?o)
      :domain (Joint ?o)
      :outputs (?g)
      :certified (HandleGrasp ?o ?g)
    )
    (:stream inverse-kinematics-grasp-handle
      :inputs (?a ?o ?p ?g)
      :domain (and (Controllable ?a) (Position ?o ?p) (HandleGrasp ?o ?g) (IsClosedPosition ?o ?p))
      :outputs (?q ?aq ?t)
      :certified (and (BConf ?q) (AConf ?a ?aq) (ATraj ?t)
                      (GraspHandle ?a ?o ?p ?g ?q ?aq)
                      (KinGraspHandle ?a ?o ?p ?g ?q ?aq ?t))
    )
    (:stream inverse-kinematics-ungrasp-handle
      :inputs (?a ?o ?p ?g ?q ?aq1)
      :domain (and (UngraspHandle ?a ?o ?p ?g ?q ?aq1) (IsOpenedPosition ?o ?p))
      :outputs (?aq2 ?t)
      :certified (and (AConf ?a ?aq2) (ATraj ?t)
                      (KinUngraspHandle ?a ?o ?p ?g ?q ?aq1 ?aq2 ?t))
    )

    (:stream plan-base-pull-handle
      :inputs (?a ?o ?p1 ?p2 ?g ?q1 ?aq)
      :domain (and (GraspHandle ?a ?o ?p1 ?g ?q1 ?aq) (Position ?o ?p2) (IsSampledPosition ?o ?p1 ?p2))
      :outputs (?q2 ?bt)
      :certified (and (BConf ?q2) (UngraspBConf ?q2) (BTraj ?bt)
                      (UngraspHandle ?a ?o ?p2 ?g ?q2 ?aq)
                      (KinPullDoorHandle ?a ?o ?p1 ?p2 ?g ?q1 ?q2 ?bt ?aq))
    )
)
