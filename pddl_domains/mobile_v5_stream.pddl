(define (stream mobile-robot-tamp)
  (:stream sample-pose
    :inputs (?o ?r)
    :domain (and (Stackable ?o ?r) (StaticLink ?r))
    :outputs (?p)
    :certified (and (Pose ?o ?p) (Supported ?o ?p ?r))
  )
  (:stream sample-relpose
    :inputs (?o ?r)
    :domain (and (Stackable ?o ?r) (MovableLink ?r))
    :outputs (?p)
    :certified (and (RelPose ?o ?p ?r))
  )

  (:stream sample-pose-inside
    :inputs (?o ?r)
    :domain (and (Containable ?o ?r) (StaticLink ?r))
    :outputs (?p)
    :certified (and (Pose ?o ?p) (Contained ?o ?p ?r))
  )
  (:stream sample-relpose-inside
    :inputs (?o ?r)
    :domain (and (Containable ?o ?r) (MovableLink ?r))
    :outputs (?p)
    :certified (and (RelPose ?o ?p ?r))
  )

  (:stream sample-grasp
    :inputs (?o)
    :domain (Graspable ?o)
    :outputs (?g)
    :certified (Grasp ?o ?g)
  )

  ;; -------------------------------------------------------------------
  ;;  grasping planning step 1
  ;; -------------------------------------------------------------------
  (:stream test-inverse-reachability
    :inputs (?a ?o ?p ?g ?q)
    :domain (and (Controllable ?a) (Pose ?o ?p) (Grasp ?o ?g) (BConf ?q))
    :certified (Reach ?a ?o ?p ?g ?q)
  )
  (:stream inverse-reachability
    :inputs (?a ?o ?p ?g)
    :domain (and (Controllable ?a) (Pose ?o ?p) (Grasp ?o ?g))
    :outputs (?q)
    :certified (and (BConf ?q) (Reach ?a ?o ?p ?g ?q))
  )
  (:stream inverse-reachability-rel
    :inputs (?a ?o1 ?rp1 ?o2 ?p2 ?g)
    :domain (and (Controllable ?a) (RelPose ?o1 ?rp1 ?o2) (Pose ?o2 ?p2) (Grasp ?o1 ?g))
    :outputs (?q)
    :certified (and (BConf ?q) (ReachRel ?a ?o1 ?rp1 ?o2 ?p2 ?g ?q))
  )

  ;; -------------------------------------------------------------------
  ;;     case 1: when objects don't collide with the world during holding up
  ;; -------------------------------------------------------------------
  (:stream inverse-kinematics
    :inputs (?a ?o ?p ?g ?q)
    :domain (Reach ?a ?o ?p ?g ?q)
    :fluents (AtPose AtRelPose AtPosition)
    :outputs (?t)
    :certified (and (Kin ?a ?o ?p ?g ?q ?t))  ;; (ATraj ?t)
  )
  (:stream inverse-kinematics-rel
    :inputs (?a ?o1 ?rp1 ?o2 ?p2 ?g ?q)
    :domain (ReachRel ?a ?o1 ?rp1 ?o2 ?p2 ?g ?q)
    :fluents (AtPose AtRelPose AtPosition)
    :outputs (?t)
    :certified (and (KinRel ?a ?o1 ?rp1 ?o2 ?p2 ?g ?q ?t))
  )

  ;; -------------------------------------------------------------------
  ;;     case 2: when objects collide with the world during holding up
  ;; -------------------------------------------------------------------
  ;(:stream plan-arm-motion-grasp
  ;  :inputs (?a ?o ?p ?g ?bq)
  ;  :domain (Reach ?a ?o ?p ?g ?bq)
  ;  :outputs (?aq ?t)
  ;  :certified (and (AConf ?a ?aq) (UngraspAConf ?a ?aq) (ATraj ?t) (KinGrasp ?a ?o ?p ?g ?bq ?aq ?t))
  ;)
  ;(:stream plan-arm-motion-ungrasp
  ;  :inputs (?a ?o ?p ?g ?aq1 ?aq2)
  ;  :domain (and (Controllable ?a) (Pose ?o ?p) (Grasp ?o ?g) (UngraspAConf ?a ?aq1) (DefaultAConf ?a ?aq2))
  ;  :outputs (?bq ?t)
  ;  :certified (and (BConf ?bq) (ATraj ?t) (KinUngrasp ?a ?o ?p ?g ?bq ?aq1 ?aq2 ?t))
  ;)
  ;; -------------------------------------------------------------------

  (:stream plan-base-motion
    :inputs (?q1 ?q2)
    :domain (and (BConf ?q1) (BConf ?q2))
    :fluents (AtPose AtRelPose AtGrasp AtPosition AtAConf)
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

  (:stream test-cfree-rel-pose-pose
    :inputs (?o1 ?rp1 ?o2 ?p2 ?o3 ?p3)
    :domain (and (RelPose ?o1 ?rp1 ?o2) (Pose ?o2 ?p2) (Pose ?o3 ?p3))
    :certified (CFreeRelPosePose ?o1 ?rp1 ?o2 ?p2 ?o3 ?p3)
  )
  (:stream test-cfree-approach-rel-pose
    :inputs (?o1 ?rp1 ?o2 ?p2 ?g ?o3 ?p3)
    :domain (and (RelPose ?o1 ?rp1 ?o2) (Pose ?o2 ?p2) (Pose ?o3 ?p3) (Grasp ?o1 ?g))
    :certified (CFreeApproachRelPose ?o1 ?rp1 ?o2 ?p2 ?g ?o3 ?p3)
  )

  (:stream test-cfree-traj-pose
    :inputs (?t ?o2 ?p2)
    :domain (and (ATraj ?t) (Pose ?o2 ?p2))
    :certified (CFreeTrajPose ?t ?o2 ?p2)
  )
  (:stream test-cfree-traj-position
    :inputs (?t ?o2 ?p2)
    :domain (and (ATraj ?t) (Position ?o2 ?p2))
    :certified (CFreeTrajPosition ?t ?o2 ?p2)
  )
  (:stream test-bconf-close-to-surface
    :inputs (?q ?s)
    :domain (and (BConf ?q) (Surface ?s))
    :certified (BConfCloseToSurface ?q ?s)
  )

  ;(:stream test-cfree-btraj-pose
  ;  :inputs (?t ?o2 ?p2)
  ;  :domain (and (BTraj ?t) (Pose ?o2 ?p2))
  ;  :certified (CFreeBTrajPose ?t ?o2 ?p2)
  ;)

  ;(:stream test-pose-in-space
  ;  :inputs (?o ?p ?r)
  ;  :domain (and (Containable ?o ?r) (Pose ?o ?p))
  ;  :certified (and (Contained ?o ?p ?r))
  ;)

  (:stream get-joint-position-open
    :inputs (?o ?p1)
    :domain (and (Joint ?o) (Position ?o ?p1) (IsClosedPosition ?o ?p1))
    :outputs (?p2)
    :certified (and (Position ?o ?p2) (IsOpenedPosition ?o ?p2) (IsSampledPosition ?o ?p1 ?p2))
  )

  (:stream get-joint-position-closed
    :inputs (?o ?p1)
    :domain (and (Joint ?o) (Position ?o ?p1) (IsOpenedPosition ?o ?p1))
    :outputs (?p2)
    :certified (and (Position ?o ?p2) (IsClosedPosition ?o ?p2) (IsSampledPosition ?o ?p1 ?p2))
  )

  ;(:stream sample-joint-position-open
  ;  :inputs (?o)
  ;  :domain (and (Joint ?o))
  ;  :outputs (?p2)
  ;  :certified (and (Position ?o ?p2) (IsOpenedPosition ?o ?p2))
  ;)
  ;(:stream sample-joint-position-closed
  ;  :inputs (?o)
  ;  :domain (and (Joint ?o))
  ;  :outputs (?p2)
  ;  :certified (and (Position ?o ?p2) (IsClosedPosition ?o ?p2))
  ;)
  ;(:stream test-joint-position-open
  ;  :inputs (?o ?p)
  ;  :domain (and (Joint ?o) (Position ?o ?p))
  ;  :certified (IsOpenedPosition ?o ?p)
  ;)
  ;(:stream test-joint-position-closed
  ;  :inputs (?o ?p)
  ;  :domain (and (Joint ?o) (Position ?o ?p))
  ;  :certified (IsClosedPosition ?o ?p)
  ;)

  ;; -------------------------------------------------------------------
  ;; manipulate handles
  ;; -------------------------------------------------------------------
    (:stream sample-handle-grasp
      :inputs (?o)
      :domain (Joint ?o)
      :outputs (?g)
      :certified (HandleGrasp ?o ?g)
    )

  ;(:function (MoveCost ?t)
  ;  (and (BTraj ?t))
  ;)

  ;(:predicate (TrajPoseCollision ?t ?o2 ?p2)
  ;  (and (BTraj ?t) (Pose ?o2 ?p2))
  ;)
  ;(:predicate (TrajArmCollision ?t ?a ?q)
  ;  (and (BTraj ?t) (AConf ?a ?q))
  ;)
  ;(:predicate (TrajGraspCollision ?t ?a ?o ?g)
  ;  (and (BTraj ?t) (Arm ?a) (Grasp ?o ?g))
  ;)

  ;;----------------------------------------------------------------------
  ;;      extended streams from _cooking_stream.pddl
  ;;----------------------------------------------------------------------

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


  ;;----------------------------------------------------------------------
  ;;      extended streams from _arrange_stream.pddl
  ;;----------------------------------------------------------------------


  ;;----------------------------------------------------------------------
  ;;      extended streams from _pull_stream.pddl
  ;;----------------------------------------------------------------------

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