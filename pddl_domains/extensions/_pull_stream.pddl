(define (stream pull_v2)

    (:stream inverse-kinematics-pull
      :inputs (?a ?o ?p1 ?p2 ?g)
      :domain (and (Controllable ?a) (Position ?o ?p1) (HandleGrasp ?o ?g) (Position ?o ?p2)
                    (IsSampledPosition ?o ?p1 ?p2) (UnattachedJoint ?o))
      :outputs (?q1 ?q2 ?aq ?at ?bt)
      :certified (and (BConf ?q1) (BConf ?q2) (AConf ?a ?aq) (ATraj ?at) (ATraj ?bt)
                      (KinPullOneAction ?a ?o ?p1 ?p2 ?g ?q1 ?q2 ?bt ?aq ?at))
    )

    (:stream inverse-kinematics-pull-with-link
      :inputs (?a ?o ?p1 ?p2 ?g ?l ?pl1)
      :domain (and (Controllable ?a) (Position ?o ?p1) (HandleGrasp ?o ?g) (Position ?o ?p2)
                    (IsSampledPosition ?o ?p1 ?p2) (JointAffectLink ?o ?l) (Pose ?l ?pl1) (StartPose ?l ?pl1))
      :outputs (?q1 ?q2 ?aq ?at ?bt ?pl2)
      :certified (and (BConf ?q1) (BConf ?q2) (AConf ?a ?aq) (ATraj ?at) (ATraj ?bt) (Pose ?l ?pl2)
                      (KinPullWithLinkOneAction ?a ?o ?p1 ?p2 ?g ?q1 ?q2 ?bt ?aq ?at ?l ?pl1 ?pl2))
    )

  (:stream test-cfree-traj-pose-at-bconf-at-joint-position
    :inputs (?t ?o2 ?p2 ?q ?o ?p1)
    :domain (and (ATraj ?t) (Pose ?o2 ?p2) (BConf ?q) (Position ?o ?p1))
    :certified (CFreeTrajPoseAtBConfAtJointPosition ?t ?o2 ?p2 ?q ?o ?p1)
  )

  (:stream test-cfree-traj-position-at-bconf-at-joint-position
    :inputs (?t ?o2 ?p2 ?q ?o ?p1)
    :domain (and (ATraj ?t) (Position ?o2 ?p2) (BConf ?q) (Position ?o ?p1))
    :certified (CFreeTrajPositionAtBConfAtJointPosition ?t ?o2 ?p2 ?q ?o ?p1)
  )

  (:stream test-cfree-traj-pose-at-bconf-at-joint-position-at-link-pose
    :inputs (?t ?o2 ?p2 ?q ?o ?p1 ?l ?lp)
    :domain (and (ATraj ?t) (Pose ?o2 ?p2) (BConf ?q) (Position ?o ?p1) (Pose ?l ?lp))
    :certified (CFreeTrajPoseAtBConfAtJointPositionAtLinkPose ?t ?o2 ?p2 ?q ?o ?p1 ?l ?lp)
  )

  (:stream test-cfree-traj-position-at-bconf-at-joint-position-at-link-pose
    :inputs (?t ?o2 ?p2 ?q ?o ?p1 ?l ?lp)
    :domain (and (ATraj ?t) (Position ?o2 ?p2) (BConf ?q) (Position ?o ?p1) (Pose ?l ?lp))
    :certified (CFreeTrajPositionAtBConfAtJointPositionAtLinkPose ?t ?o2 ?p2 ?q ?o ?p1 ?l ?lp)
  )

)