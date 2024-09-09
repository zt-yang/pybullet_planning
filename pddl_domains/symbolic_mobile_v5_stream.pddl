(define (stream mobile-robot-tamp)
  (:stream sample-pose
    :inputs (?o ?r)
    :domain (and (Stackable ?o ?r) (StaticLink ?r))
    :outputs ()
    :certified (and (Pose ?o) (Supported ?o ?r))
  )
  (:stream sample-relpose
    :inputs (?o ?r)
    :domain (and (Stackable ?o ?r) (MovableLink ?r))
    :outputs ()
    :certified (and (RelPose ?o ?r))
  )

  (:stream sample-pose-inside
    :inputs (?o ?r)
    :domain (and (Containable ?o ?r) (StaticLink ?r))
    :outputs ()
    :certified (and (Pose ?o) (Contained ?o ?r))
  )
  (:stream sample-relpose-inside
    :inputs (?o ?r)
    :domain (and (Containable ?o ?r) (MovableLink ?r))
    :outputs ()
    :certified (and (RelPose ?o ?r))
  )

  (:stream sample-grasp
    :inputs (?o)
    :domain (Graspable ?o)
    :outputs ()
    :certified (Grasp ?o)
  )

  ;; -------------------------------------------------------------------
  ;;  grasping planning step 1
  ;; -------------------------------------------------------------------
  (:stream test-inverse-reachability
    :inputs (?a ?o)
    :domain (and (Controllable ?a) (Pose ?o) (Grasp ?o) (BConf))
    :certified (Reach ?a ?o)
  )
  (:stream inverse-reachability
    :inputs (?a ?o)
    :domain (and (Controllable ?a) (Pose ?o) (Grasp ?o))
    :outputs ()
    :certified (and (BConf) (Reach ?a ?o))
  )
  (:stream inverse-reachability-rel
    :inputs (?a ?o1 ?o2)
    :domain (and (Controllable ?a) (RelPose ?o1 ?o2) (Pose ?o2) (Grasp ?o1))
    :outputs ()
    :certified (and (BConf) (ReachRel ?a ?o1 ?o2))
  )

  ;; -------------------------------------------------------------------
  ;;     case 1: when objects don't collide with the world during holding up
  ;; -------------------------------------------------------------------
  (:stream inverse-kinematics
    :inputs (?a ?o)
    :domain (Reach ?a ?o)
    :fluents (AtPose AtRelPose AtPosition)
    :outputs ()
    :certified (and (Kin ?a ?o))  ;; (ATraj)
  )
  (:stream inverse-kinematics-rel
    :inputs (?a ?o1 ?o2)
    :domain (ReachRel ?a ?o1 ?o2)
    :fluents (AtPose AtRelPose AtPosition)
    :outputs ()
    :certified (and (KinRel ?a ?o1 ?o2))
  )

  ;; -------------------------------------------------------------------
  ;;     case 2: when objects collide with the world during holding up
  ;; -------------------------------------------------------------------
  ;(:stream plan-arm-motion-grasp
  ;  :inputs (?a ?o)
  ;  :domain (Reach ?a ?o)
  ;  :outputs ()
  ;  :certified (and (AConf ?a) (UngraspAConf ?a) (ATraj) (KinGrasp ?a ?o))
  ;)
  ;(:stream plan-arm-motion-ungrasp
  ;  :inputs (?a ?o)
  ;  :domain (and (Controllable ?a) (Pose ?o) (Grasp ?o) (UngraspAConf ?a) (DefaultAConf ?a))
  ;  :outputs ()
  ;  :certified (and (BConf) (ATraj) (KinUngrasp ?a ?o))
  ;)
  ;; -------------------------------------------------------------------

  (:stream plan-base-motion
    :inputs ()
    :domain (and (BConf) (BConf))
    :fluents (AtPose AtRelPose AtGrasp AtPosition AtAConf)
    :outputs ()
    :certified (and (BTraj) (BaseMotion))
  )

  (:stream test-cfree-pose-pose
    :inputs (?o1 ?o2)
    :domain (and (Pose ?o1) (Pose ?o2))
    :certified (CFreePosePose ?o1 ?o2)
  )
  (:stream test-cfree-approach-pose
    :inputs (?o1 ?o2)
    :domain (and (Pose ?o1) (Grasp ?o1) (Pose ?o2))
    :certified (CFreeApproachPose ?o1 ?o2)
  )

  (:stream test-cfree-rel-pose-pose
    :inputs (?o1 ?o2 ?o3)
    :domain (and (RelPose ?o1 ?o2) (Pose ?o2) (Pose ?o3))
    :certified (CFreeRelPosePose ?o1 ?o2 ?o3)
  )
  (:stream test-cfree-approach-rel-pose
    :inputs (?o1 ?o2 ?o3)
    :domain (and (RelPose ?o1 ?o2) (Pose ?o2) (Pose ?o3) (Grasp ?o1))
    :certified (CFreeApproachRelPose ?o1 ?o2 ?o3)
  )

  (:stream test-cfree-traj-pose
    :inputs (?o2)
    :domain (and (ATraj) (Pose ?o2))
    :certified (CFreeTrajPose ?o2)
  )
  (:stream test-cfree-traj-position
    :inputs (?o2)
    :domain (and (ATraj) (Position ?o2))
    :certified (CFreeTrajPosition ?o2)
  )
  (:stream test-bconf-close-to-surface
    :inputs (?s)
    :domain (and (BConf) (Surface ?s))
    :certified (BConfCloseToSurface ?s)
  )

  ;(:stream test-cfree-btraj-pose
  ;  :inputs (?o2)
  ;  :domain (and (BTraj) (Pose ?o2))
  ;  :certified (CFreeBTrajPose ?o2)
  ;)

  ;(:stream test-pose-in-space
  ;  :inputs (?o ?r)
  ;  :domain (and (Containable ?o ?r) (Pose ?o))
  ;  :certified (and (Contained ?o ?r))
  ;)

  (:stream get-joint-position-open
    :inputs (?o)
    :domain (and (Joint ?o) (Position ?o) (IsClosedPosition ?o))
    :outputs ()
    :certified (and (Position ?o) (IsOpenedPosition ?o) (IsSampledPosition ?o))
  )

  (:stream get-joint-position-closed
    :inputs (?o)
    :domain (and (Joint ?o) (Position ?o) (IsOpenedPosition ?o))
    :outputs ()
    :certified (and (Position ?o) (IsClosedPosition ?o) (IsSampledPosition ?o))
  )

  ;(:stream sample-joint-position-open
  ;  :inputs (?o)
  ;  :domain (and (Joint ?o))
  ;  :outputs ()
  ;  :certified (and (Position ?o) (IsOpenedPosition ?o))
  ;)
  ;(:stream sample-joint-position-closed
  ;  :inputs (?o)
  ;  :domain (and (Joint ?o))
  ;  :outputs ()
  ;  :certified (and (Position ?o) (IsClosedPosition ?o))
  ;)
  ;(:stream test-joint-position-open
  ;  :inputs (?o)
  ;  :domain (and (Joint ?o) (Position ?o))
  ;  :certified (IsOpenedPosition ?o)
  ;)
  ;(:stream test-joint-position-closed
  ;  :inputs (?o)
  ;  :domain (and (Joint ?o) (Position ?o))
  ;  :certified (IsClosedPosition ?o)
  ;)

  ;; -------------------------------------------------------------------
  ;; manipulate handles
  ;; -------------------------------------------------------------------
    (:stream sample-handle-grasp
      :inputs (?o)
      :domain (Joint ?o)
      :outputs ()
      :certified (HandleGrasp ?o)
    )

  ;(:function (MoveCost)
  ;  (and (BTraj))
  ;)

  ;(:predicate (TrajPoseCollision ?o2)
  ;  (and (BTraj) (Pose ?o2))
  ;)
  ;(:predicate (TrajArmCollision ?a)
  ;  (and (BTraj) (AConf ?a))
  ;)
  ;(:predicate (TrajGraspCollision ?a ?o)
  ;  (and (BTraj) (Arm ?a) (Grasp ?o))
  ;)

  ;;----------------------------------------------------------------------
  ;;      extended streams from _cooking_stream.pddl
  ;;----------------------------------------------------------------------

  (:stream sample-pose-sprinkle
    :inputs (?o1 ?o2)
    :domain (and (Region ?o1) (Pose ?o1) (Graspable ?o2))
    :outputs ()
    :certified (and (Pose ?o2) (SprinklePose ?o1 ?o2))
  )

  (:stream test-cfree-pose-between
    :inputs (?o1 ?o2 ?o3)
    :domain (and (Pose ?o1) (Pose ?o2) (Pose ?o3))
    :certified (CFreePoseBetween ?o1 ?o2 ?o3)
  )


  ;;----------------------------------------------------------------------
  ;;      extended streams from _arrange_stream.pddl
  ;;----------------------------------------------------------------------


  ;;----------------------------------------------------------------------
  ;;      extended streams from _pull_stream.pddl
  ;;----------------------------------------------------------------------

    (:stream inverse-kinematics-pull
      :inputs (?a ?o)
      :domain (and (Controllable ?a) (Position ?o) (HandleGrasp ?o) (Position ?o)
                    (IsSampledPosition ?o) (UnattachedJoint ?o))
      :outputs ()
      :certified (and (BConf) (BConf) (AConf ?a) (ATraj) (ATraj)
                      (KinPullOneAction ?a ?o))
    )

    (:stream inverse-kinematics-pull-with-link
      :inputs (?a ?o ?l ?pl1)
      :domain (and (Controllable ?a) (Position ?o) (HandleGrasp ?o) (Position ?o)
                    (IsSampledPosition ?o) (JointAffectLink ?o ?l) (Pose ?l ?pl1) (StartPose ?l ?pl1))
      :outputs (?pl2)
      :certified (and (BConf) (BConf) (AConf ?a) (ATraj) (ATraj) (Pose ?l ?pl2)
                      (KinPullWithLinkOneAction ?a ?o ?l ?pl1 ?pl2))
    )

  (:stream test-cfree-traj-pose-at-bconf-at-joint-position
    :inputs (?o2 ?o)
    :domain (and (ATraj) (Pose ?o2) (BConf) (Position ?o))
    :certified (CFreeTrajPoseAtBConfAtJointPosition ?o2 ?o)
  )

  (:stream test-cfree-traj-position-at-bconf-at-joint-position
    :inputs (?o2 ?o)
    :domain (and (ATraj) (Position ?o2) (BConf) (Position ?o))
    :certified (CFreeTrajPositionAtBConfAtJointPosition ?o2 ?o)
  )

  (:stream test-cfree-traj-pose-at-bconf-at-joint-position-at-link-pose
    :inputs (?o2 ?o ?l)
    :domain (and (ATraj) (Pose ?o2) (BConf) (Position ?o) (Pose ?l))
    :certified (CFreeTrajPoseAtBConfAtJointPositionAtLinkPose ?o2 ?o ?l)
  )

  (:stream test-cfree-traj-position-at-bconf-at-joint-position-at-link-pose
    :inputs (?o2 ?o ?l)
    :domain (and (ATraj) (Position ?o2) (BConf) (Position ?o) (Pose ?l))
    :certified (CFreeTrajPositionAtBConfAtJointPositionAtLinkPose ?o2 ?o ?l)
  )


)
)