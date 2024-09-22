(define (domain pull_v2)
  (:predicates
     (KinPullOneAction ?a ?o ?p1 ?p2 ?g ?q1 ?q2 ?bt ?aq ?at)
     (KinPullWithLinkOneAction ?a ?o ?p1 ?p2 ?g ?q1 ?q2 ?bt ?aq2 ?at ?l ?lp1 ?lp2)
     (StartPose ?l ?lp)

     (UnsafeATrajToPosesAtBConfAtJointPosition ?at ?q1 ?o ?p1)
     (UnsafeATrajToPositionsAtBConfAtJointPosition ?at ?q1 ?o ?p1)
     (CFreeTrajPoseAtBConfAtJointPosition ?t ?o2 ?p2 ?q ?o ?p1)
     (CFreeTrajPositionAtBConfAtJointPosition ?t ?o2 ?p2 ?q ?o ?p1)

     (UnsafeATrajToPosesAtBConfAtJointPositionAtLinkPose ?at ?q1 ?o ?p1 ?l ?lp)
     (UnsafeATrajToPositionsAtBConfAtJointPositionAtLinkPose ?at ?q1 ?o ?p1 ?l ?lp)
     (CFreeTrajPoseAtBConfAtJointPositionAtLinkPose ?t ?o2 ?p2 ?q ?o ?p1 ?l ?lp)
     (CFreeTrajPositionAtBConfAtJointPositionAtLinkPose ?t ?o2 ?p2 ?q ?o ?p1 ?l ?lp)
  )

  ;; from position ?p1 pull to the position ?p2
  (:action grasp_pull_ungrasp_handle
   :parameters (?a ?o ?p1 ?p2 ?g ?q1 ?q2 ?bt ?aq1 ?aq2 ?at)
   :precondition (and (Joint ?o) (CanGraspHandle)
                      (not (PulledOneAction ?o)) (not (Pulled ?o))
                      (not (= ?p1 ?p2)) (CanPull ?a) (HandEmpty ?a)
                      (AtBConf ?q1) (AtAConf ?a ?aq1)
                      (AtPosition ?o ?p1) (Position ?o ?p2)

                      (UnattachedJoint ?o)
                      (KinPullOneAction ?a ?o ?p1 ?p2 ?g ?q1 ?q2 ?bt ?aq2 ?at)
                      (not (UnsafeATrajToPosesAtBConfAtJointPosition ?at ?q1 ?o ?p1))
                      (not (UnsafeATrajToPositionsAtBConfAtJointPosition ?at ?q1 ?o ?p1))
                      (not (UnsafeATrajToPosesAtBConfAtJointPosition ?at ?q2 ?o ?p1))
                      (not (UnsafeATrajToPositionsAtBConfAtJointPosition ?at ?q2 ?o ?p1))
                    )
    :effect (and (AtPosition ?o ?p2) (not (AtPosition ?o ?p1))
                 (PulledOneAction ?o) (GraspedHandle ?o) (CanMove)
                 (AtBConf ?q2) (not (AtBConf ?q1)) ; plan-base-pull-handle
            )
  )

  ;; from position ?p1 pull to the position ?p2, also affecting the pose of link attached to it
  (:action grasp_pull_ungrasp_handle_with_link
   :parameters (?a ?o ?p1 ?p2 ?g ?q1 ?q2 ?bt ?aq1 ?aq2 ?at ?l ?lp1 ?lp2)
   :precondition (and (Joint ?o) (CanGraspHandle)
                      ; (not (PulledOneAction ?o)) (not (Pulled ?o))  ;; so that drawer can be opened then closed
                      (not (= ?p1 ?p2)) (CanPull ?a) (HandEmpty ?a)
                      (AtBConf ?q1) (AtAConf ?a ?aq1)
                      (AtPosition ?o ?p1) (Position ?o ?p2)

                      (JointAffectLink ?o ?l) (AtPose ?l ?lp1) (StartPose ?l ?lp1) (Pose ?l ?lp2)
                      (KinPullWithLinkOneAction ?a ?o ?p1 ?p2 ?g ?q1 ?q2 ?bt ?aq2 ?at ?l ?lp1 ?lp2)
                      (not (UnsafeATrajToPosesAtBConfAtJointPositionAtLinkPose ?at ?q1 ?o ?p1 ?l ?lp1))
                      (not (UnsafeATrajToPositionsAtBConfAtJointPositionAtLinkPose ?at ?q1 ?o ?p1 ?l ?lp1))
                      (not (UnsafeATrajToPosesAtBConfAtJointPositionAtLinkPose ?at ?q2 ?o ?p1 ?l ?lp2))
                      (not (UnsafeATrajToPositionsAtBConfAtJointPositionAtLinkPose ?at ?q2 ?o ?p1 ?l ?lp2))
                    )
   :effect (and (AtPosition ?o ?p2) (not (AtPosition ?o ?p1))
                (PulledOneAction ?o) (GraspedHandle ?o) (CanMove)
                (AtBConf ?q2) (not (AtBConf ?q1)) ; plan-base-pull-handle
                (not (AtPose ?l ?lp1)) (AtPose ?l ?lp2)
           )
  )

  (:derived (UnsafeATrajToPosesAtBConfAtJointPosition ?t ?q ?o ?p1)
    (exists (?o2 ?p2) (and (ATraj ?t) (Pose ?o2 ?p2) (AtPose ?o2 ?p2) (BConf ?q) (Position ?o ?p1)
                            (not (CFreeTrajPoseAtBConfAtJointPosition ?t ?o2 ?p2 ?q ?o ?p1)) ))
  )

  (:derived (UnsafeATrajToPositionsAtBConfAtJointPosition ?t ?q ?o ?p1)
    (exists (?o2 ?p2) (and (ATraj ?t) (Position ?o2 ?p2) (AtPosition ?o2 ?p2) (BConf ?q) (Position ?o ?p1)
                            (not (CFreeTrajPositionAtBConfAtJointPosition ?t ?o2 ?p2 ?q ?o ?p1)) ))
  )

  (:derived (UnsafeATrajToPosesAtBConfAtJointPositionAtLinkPose ?t ?q ?o ?p1 ?l ?lp)
    (exists (?o2 ?p2) (and (ATraj ?t) (Pose ?o2 ?p2) (AtPose ?o2 ?p2) (BConf ?q) (Position ?o ?p1) (Pose ?l ?lp)
                            (not (CFreeTrajPoseAtBConfAtJointPositionAtLinkPose ?t ?o2 ?p2 ?q ?o ?p1 ?l ?lp)) ))
  )

  (:derived (UnsafeATrajToPositionsAtBConfAtJointPositionAtLinkPose ?t ?q ?o ?p1 ?l ?lp)
    (exists (?o2 ?p2) (and (ATraj ?t) (Position ?o2 ?p2) (AtPosition ?o2 ?p2) (BConf ?q) (Position ?o ?p1) (Pose ?l ?lp)
                            (not (CFreeTrajPositionAtBConfAtJointPositionAtLinkPose ?t ?o2 ?p2 ?q ?o ?p1 ?l ?lp)) ))
  )

)