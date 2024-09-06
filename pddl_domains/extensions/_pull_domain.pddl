(define (domain pull_v2)
  (:predicates
     (KinPullOneAction ?a ?o ?p1 ?p2 ?g ?q1 ?q2 ?bt ?aq ?at)
     (KinPullWithLinkOneAction ?a ?o ?p1 ?p2 ?g ?q1 ?q2 ?bt ?aq2 ?at ?l ?pl1 ?pl2)
     (StartPose ?l ?pl)
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
                    )
    :effect (and (AtPosition ?o ?p2) (not (AtPosition ?o ?p1)) (GraspedHandle ?o)
                 (PulledOneAction ?o) (GraspedHandle ?o) (CanMove)
                 (AtBConf ?q2) (not (AtBConf ?q1)) ; plan-base-pull-handle
            )
  )

  ;; from position ?p1 pull to the position ?p2, also affecting the pose of link attached to it
  (:action grasp_pull_ungrasp_handle_with_link
   :parameters (?a ?o ?p1 ?p2 ?g ?q1 ?q2 ?bt ?aq1 ?aq2 ?at ?l ?pl1 ?pl2)
   :precondition (and (Joint ?o) (CanGraspHandle)
                      (not (PulledOneAction ?o)) (not (Pulled ?o))
                      (not (= ?p1 ?p2)) (CanPull ?a) (HandEmpty ?a)
                      (AtBConf ?q1) (AtAConf ?a ?aq1)
                      (AtPosition ?o ?p1) (Position ?o ?p2)

                      (JointAffectLink ?o ?l) (AtPose ?l ?pl1) (StartPose ?l ?pl1) (Pose ?l ?pl2)
                      (KinPullWithLinkOneAction ?a ?o ?p1 ?p2 ?g ?q1 ?q2 ?bt ?aq2 ?at ?l ?pl1 ?pl2)
                    )
    :effect (and (AtPosition ?o ?p2) (not (AtPosition ?o ?p1)) (GraspedHandle ?o)
                 (PulledOneAction ?o) (GraspedHandle ?o) (CanMove)
                 (AtBConf ?q2) (not (AtBConf ?q1)) ; plan-base-pull-handle
                 (not (AtPose ?l ?pl1)) (AtPose ?l ?pl2)
            )
  )

)