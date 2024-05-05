(define (domain mobile-robot-tamp)
  (:requirements :strips :equality)

  (:constants
    @movable @bottle @edible @medicine
  )


  (:predicates

    (Sink ?r)
    (Stove ?r)
    (Counter ?r)
    (Table ?r)
    (Salter ?o)
    (Egg ?o)
    (Veggie ?o)
    (Plate ?o)

    (Arm ?a)
    (Drawer ?o) ;;
    (Door ?o) ;;
    (Knob ?o) ;;
    (Joint ?o)
    (JointAffectLink ?j ?l)
    (UnattachedJoint ?o)

    (Edible ?o)
    (CleaningSurface ?s)
    (HeatingSurface ?s)
    (ControlledBy ?s ?n)

    (AConf ?a ?q)
    (DefaultAConf ?a ?q)
    (UngraspAConf ?a ?aq)  ;; for half-pick / place
    (BConf ?q)
    (UngraspBConf ?bq)

    (Pose ?o ?p)
    (RelPose ?o1 ?rp1 ?o2)
    (MagicPose ?o ?p)  ;; for teleport debugging
    (Position ?o ?p)  ;; joint position of a body
    (IsOpenedPosition ?o ?p)  ;;
    (IsClosedPosition ?o ?p)  ;; assume things start out closed
    (IsSampledPosition ?o ?p1 ?p2)  ;; to make planning more efficient

    (Grasp ?o ?g)
    (HandleGrasp ?o ?g)

    (Controllable ?o)
    (Graspable ?o)
    (MovableLink ?o)
    (StaticLink ?o)
    (Stackable ?o ?r)
    (Containable ?o ?r)

    (Kin ?a ?o ?p ?g ?q ?t)
    (KinRel ?a ?o1 ?rp1 ?o2 ?p2 ?g ?q ?t)
    (KinGrasp ?a ?o ?p ?g ?bq ?aq ?t)  ;; for half-pick / place
    (KinUngrasp ?a ?o ?p ?g ?bq ?aq1 ?aq2 ?t)
    (KinGraspHandle ?a ?o ?p ?g ?q ?aq ?t)  ;; grasp a handle
    (KinUngraspHandle ?a ?o ?p ?g ?q ?aq1 ?aq2 ?t)  ;; ungrasp a handle
    (KinPullDrawerHandle ?a ?o ?p1 ?p2 ?g ?q1 ?q2 ?t)  ;; pull the handle
    (KinPullDoorHandle ?a ?o ?p1 ?p2 ?g ?q1 ?q2 ?bt ?aq)  ;; pull the handle
    (KinPullDoorHandleWithLink ?a ?o ?p1 ?p2 ?g ?q1 ?q2 ?bt ?aq ?l ?pl1 ?pl2)  ;; pull the handle
    (KinTurnKnob ?a ?o ?p1 ?p2 ?g ?q ?aq1 ?aq2 ?at)

    (Reach ?a ?o ?p ?g ?bq)
    (ReachRel ?a ?o1 ?rp1 ?o2 ?p2 ?g ?bq)

    (BConfCloseToSurface ?q ?s)

    (BaseMotion ?q1 ?t ?q2)
    (BaseMotionWithObj ?q1 ?t ?a ?o ?g ?q2)
    (ArmMotion ?a ?q1 ?t ?q2)
    (Supported ?o ?p ?r)
    (Contained ?o ?p ?s) ;; aabb contains
    (BTraj ?t)
    (ATraj ?t)

    (TrajPoseCollision ?t ?o ?p)
    (TrajArmCollision ?t ?a ?q)
    (TrajGraspCollision ?t ?a ?o ?g)
    (CFreePosePose ?o ?p ?o2 ?p2)
    (CFreeApproachPose ?o ?p ?g ?o2 ?p2)
    (CFreeRelPosePose ?o1 ?rp1 ?o2 ?p2 ?o3 ?p3)
    (CFreeApproachRelPose ?o1 ?rp1 ?o2 ?p2 ?g ?o3 ?p3)
    (CFreeTrajPose ?t ?o2 ?p2)
    (CFreeTrajPosition ?t ?o2 ?p2)

    (CFreeBTrajPose ?t ?o2 ?p2)

    (AtPose ?o ?p)
    (AtRelPose ?o1 ?rp1 ?o2)
    (AtPosition ?o ?p)  ;; joint position of a body

    (AtGrasp ?a ?o ?g)
    (AtGraspHalf ?a ?o ?g)
    (AtHandleGrasp ?a ?o ?g)  ;; holding the handle
    (HandleGrasped ?a ?o)  ;; holding the handle
    (KnobTurned ?a ?o)  ;; holding the knob
    (HandEmpty ?a)
    (AtBConf ?q)
    (AtAConf ?a ?q)

    (CanMoveBase)
    (CanMove)
    (CanPull ?a)
    (CanUngrasp)
    (CanPick)
    (CanGraspHandle)

    (Cleaned ?o)
    (Cooked ?o)
    (Seasoned ?o)
    (Served ?o ?o2)
    (EnableOmelette ?egg1 ?veggie1 ?plate1)
    (ExistOmelette ?env1)

    (OpenedJoint ?o) ;;
    (ClosedJoint ?o) ;;
    (GraspedHandle ?o) ;;

    (On ?o ?r)
    (In ?o ?r) ;;
    (Holding ?a ?o)
    (OfType ?o ?t)
    (StoredInSpace ?t ?r)
    (Space ?r)

    (UnsafePose ?o ?p)
    (UnsafeApproach ?o ?p ?g)
    (UnsafePoseRel ?o1 ?rp1 ?o2 ?p2)
    (UnsafeApproachRel ?o1 ?rp1 ?o2 ?p2 ?g)
    (UnsafeOTraj ?t)
    (UnsafeATraj ?t)
    (UnsafeBTraj ?t)
    (PoseObstacle ?o ?p ?o2)
    (ApproachObstacle ?o ?p ?g ?o2)
    (ATrajObstacle ?t ?o)

    (Debug1)
    (Debug2)
    (Debug3)

    (Identical ?v1 ?v2)

    (Picked ?o)
    (Placed ?o)
    (Pulled ?o)
    (Enabled)
    (Disabled)

  ;;----------------------------------------------------------------------
  ;;      extended predicates from _namo_domain.pddl
  ;;----------------------------------------------------------------------

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



  (:functions
    (MoveCost ?t)
    (PickCost)
    (PlaceCost)
  )

  (:action move_base
    :parameters (?q1 ?q2 ?t)
    :precondition (and (CanMoveBase) (CanMove) (BaseMotion ?q1 ?t ?q2) ; (HandEmpty ?a)
                       (not (Identical ?q1 ?q2))
                       (AtBConf ?q1))
    :effect (and (AtBConf ?q2)
                 (not (AtBConf ?q1)) (not (CanMove))
                 ;(increase (total-cost) (MoveCost ?t))
                 (increase (total-cost) 1)
            )
  )

  ;(:action move_arm
  ;  :parameters (?q1 ?q2 ?t)
  ;  :precondition (and (ArmMotion ?a ?q1 ?t ?q2)
  ;                     (AtAConf ?a ?q1))
  ;  :effect (and (AtAConf ?a ?q2)
  ;               (not (AtAConf ?a ?q1)))
  ;)

  ;(:action pick_half
  ;  :parameters (?a ?o ?p ?g ?bq ?aq1 ?aq2 ?t)
  ;  :precondition (and (KinGrasp ?a ?o ?p ?g ?bq ?aq2 ?t) (Enabled)
  ;                     (AtPose ?o ?p) (HandEmpty ?a)
  ;                     (AtBConf ?bq) (AtAConf ?a ?aq1) (DefaultAConf ?a ?aq1)
  ;                     (not (UnsafeApproach ?o ?p ?g))
  ;                     (not (UnsafeATraj ?t))
  ;                )
  ;  :effect (and (AtGrasp ?a ?o ?g) (CanMove)
  ;               (AtAConf ?a ?aq2) (not (AtAConf ?a ?aq1))
  ;               (not (AtPose ?o ?p)) (not (HandEmpty ?a))
  ;               (increase (total-cost) 1)
  ;          )
  ;)

  ;(:action place_half
  ;  :parameters (?a ?o ?p ?g ?bq ?aq1 ?aq2 ?t)
  ;  :precondition (and (KinUngrasp ?a ?o ?p ?g ?bq ?aq1 ?aq2 ?t) (Enabled)
  ;                     (AtGrasp ?a ?o ?g) (AtBConf ?bq)
  ;                     (AtAConf ?a ?aq1) (DefaultAConf ?a ?aq2)
  ;                     (not (UnsafePose ?o ?p))
  ;                     (not (UnsafeApproach ?o ?p ?g))
  ;                     (not (UnsafeATraj ?t))
  ;                     )
  ;  :effect (and (AtPose ?o ?p) (HandEmpty ?a) (CanMove)
  ;               (AtAConf ?a ?aq2) (not (AtAConf ?a ?aq1))
  ;               (not (AtGrasp ?a ?o ?g))
  ;               (increase (total-cost) 1)
  ;          )
  ;)

  ;; pick from drawer or movable object
  (:action pick_from_supporter
    :parameters (?a ?o1 ?rp1 ?o2 ?p2 ?g ?q ?t)
    :precondition (and (KinRel ?a ?o1 ?rp1 ?o2 ?p2 ?g ?q ?t) (Graspable ?o1) (MovableLink ?o2) (CanPick)
                       (AtRelPose ?o1 ?rp1 ?o2) (AtPose ?o2 ?p2) (HandEmpty ?a) (AtBConf ?q)
                       (not (UnsafeApproachRel ?o1 ?rp1 ?o2 ?p2 ?g))
                       )
    :effect (and (AtGrasp ?a ?o1 ?g) (CanMove)
                 (not (AtRelPose ?o1 ?rp1 ?o2)) (not (HandEmpty ?a))
                 (increase (total-cost) 1)
            )
  )

  ;; place onto drawer or movable object
  (:action place_to_supporter
    :parameters (?a ?o1 ?rp1 ?o2 ?p2 ?g ?q ?t)
    :precondition (and (KinRel ?a ?o1 ?rp1 ?o2 ?p2 ?g ?q ?t) (Graspable ?o1) (MovableLink ?o2)
                       (AtGrasp ?a ?o1 ?g) (AtBConf ?q) (AtPose ?o2 ?p2)
                       (not (UnsafePoseRel ?o1 ?rp1 ?o2 ?p2))
                       (not (UnsafeApproachRel ?o1 ?rp1 ?o2 ?p2 ?g))
                       (not (CanMove))
                       )
    :effect (and (AtRelPose ?o1 ?rp1 ?o2) (HandEmpty ?a) (CanMove)
                 (not (AtGrasp ?a ?o1 ?g))
                 (increase (total-cost) 1)
            )
  )

  (:action pick
    :parameters (?a ?o ?p ?g ?q ?t)
    :precondition (and (Kin ?a ?o ?p ?g ?q ?t) (Graspable ?o) (CanPick)
                       (AtPose ?o ?p) (HandEmpty ?a) (AtBConf ?q)
                       (not (UnsafeApproach ?o ?p ?g))
                       ; (not (UnsafeATraj ?t)) (not (UnsafeOTraj ?o ?g ?t)) (not (CanMove)) (not (Picked ?o))
                       )
    :effect (and (AtGrasp ?a ?o ?g) (CanMove) ; (Picked ?o)
                 (not (AtPose ?o ?p)) (not (HandEmpty ?a))
                 ; (increase (total-cost) (PickCost))
                 (increase (total-cost) 1)
            )
  )

  (:action place
    :parameters (?a ?o ?p ?g ?q ?t)
    :precondition (and (Kin ?a ?o ?p ?g ?q ?t) (Graspable ?o)
                       (AtGrasp ?a ?o ?g) (AtBConf ?q)
                       (not (UnsafePose ?o ?p))
                       (not (UnsafeApproach ?o ?p ?g))
                       (not (CanMove))
                       ; (not (UnsafeATraj ?t)) (not (UnsafeOTraj ?o ?g ?t))
                       ; (not (Placed ?o))
                       )
    :effect (and (AtPose ?o ?p) (HandEmpty ?a) (CanMove)
                 (not (AtGrasp ?a ?o ?g))
                 ; (increase (total-cost) (PlaceCost))
                 (increase (total-cost) 1)
            )
  )

    ;(:action declare_store_in_space
    ;  :parameters (?t ?r)
    ;  :precondition (and (Space ?r)
    ;                     (forall (?o) (imply (OfType ?o ?t) (In ?o ?r)))
    ;                )
    ;  :effect (and (StoredInSpace ?t ?r))
    ;)

  (:action clean
    :parameters (?o ?r)
    :precondition (and (Stackable ?o ?r) (Sink ?r) (On ?o ?r))
    :effect (and (Cleaned ?o))
  )
  (:action cook
    :parameters (?o ?r)
    :precondition (and (Stackable ?o ?r) (Stove ?r) (On ?o ?r)
                       (Cleaned ?o)
                       )
    :effect (and (Cooked ?o)
                 (not (Cleaned ?o))
                 )
  )
  (:action season
    :parameters (?o ?r ?o2)
    :precondition (and (Stackable ?o ?r) (Counter ?r)
                       (On ?o ?r) (Cooked ?o)
                       (Stackable ?o2 ?r) (Salter ?o2)
                       (On ?o2 ?r))
    :effect (and (Seasoned ?o))
  )
  (:action serve
    :parameters (?o ?r ?o2)
    :precondition (and (Stackable ?o ?r) (Table ?r)
                       (On ?o ?r) (Seasoned ?o)
                       (Stackable ?o2 ?r) (Plate ?o2)
                       (On ?o2 ?r))
    :effect (and (Served ?o ?o2))
  )


  (:action just-clean
    :parameters (?a ?o ?s)
    :precondition (and (Controllable ?a) (HandEmpty ?a) ;; (CanMove)
                       (Stackable ?o ?s) (On ?o ?s)
                       (CleaningSurface ?s)
                       ;; (AtBConf ?q) (BConfCloseToSurface ?q ?s)
                       )
    :effect (and (Cleaned ?o) ) ;(not (Picked ?o))
  )

  (:action just-cook
    :parameters (?a ?o ?s)
    :precondition (and (Controllable ?a) (HandEmpty ?a) ;; (CanMove)
                       (Stackable ?o ?s) (On ?o ?s)
                       (HeatingSurface ?s) (Cleaned ?o)
                       ;; (AtBConf ?q) (BConfCloseToSurface ?q ?s)
                       )
    :effect (and (Cooked ?o)) ; (not (Picked ?o))
  )

  (:action just-serve
    :parameters (?a ?o ?r)
    :precondition (and (Controllable ?a) (HandEmpty ?a) ;; (CanMove) ;; (Enabled)
                       (Stackable ?o ?r) (On ?o ?r) (Plate ?r)
                       ;; (AtBConf ?q) (BConfCloseToSurface ?q ?r)
                       (Cleaned ?o)
                       )
    :effect (and (Served ?o ?r))
  )


  (:action wait-clean
    :parameters (?o ?s ?n)
    :precondition (and (Edible ?o) (CleaningSurface ?s) (ControlledBy ?s ?n)
                       (On ?o ?s) (GraspedHandle ?n)
                       )
    :effect (and (Cleaned ?o))
  )
  (:action wait-cook
    :parameters (?o ?s ?n)
    :precondition (and (Edible ?o) (HeatingSurface ?s) (ControlledBy ?s ?n)
                       (On ?o ?s) (GraspedHandle ?n)
                       (Cleaned ?o)
                       )
    :effect (and (Cooked ?o)
                 (not (Cleaned ?o)))
  )

    (:action grasp_handle
      :parameters (?a ?o ?p ?g ?q ?aq1 ?aq2 ?t)
      :precondition (and (Joint ?o) (AConf ?a ?aq1) (CanGraspHandle)
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
      :precondition (and (Joint ?o) (not (= ?p1 ?p2)) (CanPull ?a) (UnattachedJoint ?o)
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

    ;; from position ?p1 pull to the position ?p2, also affecting the pose of link attached to it
    (:action pull_handle_with_link
      :parameters (?a ?o ?p1 ?p2 ?g ?q1 ?q2 ?bt ?aq ?l ?pl1 ?pl2)
      :precondition (and (Joint ?o) (not (= ?p1 ?p2)) (CanPull ?a)
                         (JointAffectLink ?o ?l) (AtPose ?l ?pl1) (Pose ?l ?pl2)
                         (AtPosition ?o ?p1) (Position ?o ?p2) (AtHandleGrasp ?a ?o ?g)
                         (KinPullDoorHandleWithLink ?a ?o ?p1 ?p2 ?g ?q1 ?q2 ?bt ?aq ?l ?pl1 ?pl2)
                         (AtBConf ?q1) (AtAConf ?a ?aq)
                         ; (not (UnsafeApproach ?o ?p2 ?g))
                         ; (not (UnsafeATraj ?at))
                         ; (not (UnsafeBTraj ?bt))
                         ; (Enabled)
                    )
      :effect (and (not (CanPull ?a)) (CanUngrasp)
                  (not (AtPose ?l ?pl1)) (AtPose ?l ?pl2)
                  (AtPosition ?o ?p2) (not (AtPosition ?o ?p1))
                  (AtBConf ?q2) (not (AtBConf ?q1))
                  (increase (total-cost) 1)
              )
    )

  ;(:action pull_door
  ; :parameters (?a ?o ?p1 ?p2 ?g ?q1 ?q2 ?bt ?aq1 ?aq2 ?at ?t1 ?aq3 ?t2)
  ; :precondition (and ; (Door ?o) (not (= ?p1 ?p2)) (CanPull ?a)
  ;                    (AtPosition ?o ?p1) ; (AtHandleGrasp ?a ?o ?g)
  ;                    (KinPullDoorHandle ?a ?o ?p1 ?p2 ?g ?q1 ?q2 ?bt ?aq1 ?aq2 ?at)
  ;                    (KinGraspHandle ?a ?o ?p1 ?g ?q1 ?aq1 ?t1)
  ;                    (KinUngraspHandle ?a ?o ?p2 ?g ?q2 ?aq2 ?aq3 ?t2)
  ;                    (AtBConf ?q1) ; (AtAConf ?a ?aq1)
  ;                    (HandEmpty ?a)
  ;                    ;(not (UnsafeApproach ?o ?p2 ?g))
  ;                    ;(not (UnsafeATraj ?at))
  ;                    ;(not (UnsafeBTraj ?bt))
  ;                    ;(not (Pulled ?o))
  ;                    (Enabled)
  ;                  )
  ;  :effect (and (AtPosition ?o ?p2) (not (AtPosition ?o ?p1)) (Pulled ?o) (CanMove)
  ;               (AtBConf ?q2) (not (AtBConf ?q1))
  ;               ;(AtAConf ?a ?aq3) (not (AtAConf ?a ?aq1))
  ;          )
  ;)

    ;; from fully closed position ?p1 pull to the fully open position ?p2
    (:action turn_knob
      :parameters (?a ?o ?p1 ?p2 ?g ?q ?aq1 ?aq2 ?at)
      :precondition (and (Knob ?o) (not (= ?p1 ?p2)) (CanPull ?a)
                         (AtPosition ?o ?p1) (Position ?o ?p2) (AtHandleGrasp ?a ?o ?g)
                         (KinTurnKnob ?a ?o ?p1 ?p2 ?g ?q ?aq1 ?aq2 ?at)
                         (AtBConf ?q) (AtAConf ?a ?aq1)
                         ;(not (UnsafeApproach ?o ?p2 ?g))
                         ;(not (UnsafeATraj ?at))
                         ;(not (UnsafeBTraj ?bt))
                    )
      :effect (and (not (CanPull ?a)) (CanUngrasp) (KnobTurned ?a ?o)
                  (AtPosition ?o ?p2) (not (AtPosition ?o ?p1))
                  (UngraspBConf ?q)
                  (AtAConf ?a ?aq2) (not (AtAConf ?a ?aq1))
              )
    )

  (:derived (On ?o ?r)
    (or
        (exists (?p) (and (Supported ?o ?p ?r) (AtPose ?o ?p)))
        (exists (?p) (and (AtRelPose ?o ?p ?r)))
    )
  )
  (:derived (In ?o ?r)
    (or
        (exists (?p) (and (Contained ?o ?p ?r) (AtPose ?o ?p)))
        (exists (?p) (and (AtRelPose ?o ?p ?r)))
    )
  )
  (:derived (Holding ?a ?o)
    (or
        (exists (?g) (and (Arm ?a) (Grasp ?o ?g)
                      (AtGrasp ?a ?o ?g)))
        (exists (?g) (and (Arm ?a) (Grasp ?o ?g)
                      (AtGraspHalf ?a ?o ?g)))
    )
  )

  (:derived (OpenedJoint ?o)
    (exists (?pstn) (and (Joint ?o) (Position ?o ?pstn) (AtPosition ?o ?pstn)
                      (IsOpenedPosition ?o ?pstn)))
  )
  (:derived (ClosedJoint ?o)
    (exists (?pstn) (and (Joint ?o) (Position ?o ?pstn) (AtPosition ?o ?pstn)
                      (IsClosedPosition ?o ?pstn)))
  )

    (:derived (HandleGrasped ?a ?o)
      (exists (?hg) (and (Arm ?a) (Joint ?o) (HandleGrasp ?o ?hg)
                        (AtHandleGrasp ?a ?o ?hg)))
    )

  (:derived (UnsafePose ?o ?p)
    (exists (?o2 ?p2) (and (Pose ?o ?p) (Pose ?o2 ?p2) (not (= ?o ?o2))
                           (not (CFreePosePose ?o ?p ?o2 ?p2))
                           (AtPose ?o2 ?p2)))
  )
  (:derived (UnsafeApproach ?o ?p ?g)
    (exists (?o2 ?p2) (and (Pose ?o ?p) (Grasp ?o ?g) (Pose ?o2 ?p2) (not (= ?o ?o2))
                           (not (CFreeApproachPose ?o ?p ?g ?o2 ?p2))
                           (AtPose ?o2 ?p2)))
  )

  (:derived (UnsafePoseRel ?o1 ?rp1 ?o2 ?p2)
    (exists (?o3 ?p3) (and (RelPose ?o1 ?rp1 ?o2) (Pose ?o2 ?p2) (Pose ?o3 ?p3) (not (= ?o1 ?o3)) (Graspable ?o3)
                           (not (CFreeRelPosePose ?o1 ?rp1 ?o2 ?p2 ?o3 ?p3))
                           (AtPose ?o3 ?p3)))
  )
  (:derived (UnsafeApproachRel ?o1 ?rp1 ?o2 ?p2 ?g)
    (exists (?o3 ?p3) (and (RelPose ?o1 ?rp1 ?o2) (Pose ?o2 ?p2) (Pose ?o3 ?p3)
                           (not (= ?o1 ?o3)) (not (= ?o2 ?o3)) (Graspable ?o3)
                           (not (CFreeApproachRelPose ?o1 ?rp1 ?o2 ?p2 ?g ?o3 ?p3))
                           (AtPose ?o3 ?p3)))
  )

  ;(:derived (UnsafeATraj ?t)
  ;  (or
  ;      (exists (?o2 ?p2) (and (ATraj ?t) (Pose ?o2 ?p2)
  ;                             (not (CFreeTrajPose ?t ?o2 ?p2))
  ;                             (AtPose ?o2 ?p2)))
  ;      (exists (?o2 ?p2) (and (ATraj ?t) (Position ?o2 ?p2)
  ;                             (not (CFreeTrajPosition ?t ?o2 ?p2))
  ;                             (AtPosition ?o2 ?p2)))
  ;  )
  ;)

  ;(:derived (UnsafeBTraj ?t)
  ;  (exists (?o2 ?p2) (and (BTraj ?t) (Pose ?o2 ?p2) (AtPose ?o2 ?p2)
  ;                         (not (CFreeBTrajPose ?t ?o2 ?p2))))
  ;)

    ;(:derived (PoseObstacle ?o ?p ?o2)
    ;  (exists (?p2)
    ;     (and (Pose ?o ?p) (Pose ?o2 ?p2) (not (= ?o ?o2))
    ;           (not (CFreePosePose ?o ?p ?o2 ?p2))
    ;           (AtPose ?o2 ?p2)))
    ;)
    ;(:derived (ApproachObstacle ?o ?p ?g ?o2)
    ;  (exists (?p2)
    ;     (and (Pose ?o ?p) (Grasp ?o ?g) (Pose ?o2 ?p2) (not (= ?o ?o2))
    ;          (not (CFreeApproachPose ?o ?p ?g ?o2 ?p2))
    ;          (AtPose ?o2 ?p2)))
    ;)
    ;(:derived (ATrajObstacle ?t ?o2)
    ;  (exists (?p2)
    ;     (and (ATraj ?t) (Pose ?o2 ?p2)
    ;          (not (CFreeTrajPose ?t ?o2 ?p2))
    ;          (AtPose ?o2 ?p2)))
    ;)

  ;(:derived (UnsafeBTraj ?t) (or
  ;  (exists (?o2 ?p2) (and (TrajPoseCollision ?t ?o2 ?p2)
  ;                         (AtPose ?o2 ?p2)))
  ;  (exists (?a ?q) (and (TrajArmCollision ?t ?a ?q)
  ;                       (AtAConf ?a ?q)))
  ;  (exists (?a ?o ?g) (and (TrajGraspCollision ?t ?a ?o ?g)
  ;                          (AtGrasp ?a ?o ?g)))
  ;))

  ;;----------------------------------------------------------------------
  ;;      extended operators & axioms from _namo_domain.pddl
  ;;----------------------------------------------------------------------

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