(define (domain spot-tamp)
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

    (Edible ?o)
    (CleaningSurface ?s)
    (HeatingSurface ?s)
    (ControlledBy ?s ?n)

    (AConf ?a ?q)
    (DefaultAConf ?a ?q)
    (UngraspAConf ?a ?aq)
    (BConf ?q)
    (UngraspBConf ?bq)
    (Pose ?o ?p)
    (MagicPose ?o ?p)  ;; for teleport debugging
    (LinkPose ?o ?p)
    (Position ?o ?p)  ;; joint position of a body
    (IsOpenedPosition ?o ?p)  ;;
    (IsClosedPosition ?o ?p)  ;; assume things start out closed
    (Grasp ?o ?g)
    (HandleGrasp ?o ?g)

    (Controllable ?o)
    (Graspable ?o)
    (Stackable ?o ?r)
    (Containable ?o ?r)  ;;

    (Kin ?a ?o ?p ?g ?q ?t)
    (KinGrasp ?a ?o ?p ?g ?bq ?aq ?t)
    (KinUngrasp ?a ?o ?p ?g ?bq ?aq1 ?aq2 ?t)
    (KinGraspHandle ?a ?o ?p ?g ?q ?aq ?t)  ;; grasp a handle
    (KinUngraspHandle ?a ?o ?p ?g ?q ?aq1 ?aq2 ?t)  ;; ungrasp a handle
    (KinPullDrawerHandle ?a ?o ?p1 ?p2 ?g ?q1 ?q2 ?t)  ;; pull the handle
    (KinPullDoorHandle ?a ?o ?p1 ?p2 ?g ?q1 ?q2 ?bt ?aq1 ?aq2 ?at)  ;; pull the handle
    (KinTurnKnob ?a ?o ?p1 ?p2 ?g ?q ?aq1 ?aq2 ?at)

    (Reach ?a ?o ?p ?g ?bq)

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
    (CFreeTrajPose ?t ?o2 ?p2)
    (CFreeTrajPosition ?t ?o2 ?p2)
    (IsClosedPosition ?o ?p)  ;;
    (IsOpenPosition ?o ?p)  ;;

    (CFreeBTrajPose ?t ?o2 ?p2)

    (AtPose ?o ?p)
    (AtLinkPose ?o ?p)
    (AtPosition ?o ?p)  ;; joint position of a body
    (OpenPosition ?o ?p)  ;; joint position of a body
    (ClosedPosition ?o ?p)  ;; joint position of a body
    (AtGrasp ?a ?o ?g)
    (AtGraspHalf ?a ?o ?g)
    (AtHandleGrasp ?a ?o ?g)  ;; holding the handle
    (HandleGrasped ?a ?o)  ;; holding the handle
    (KnobTurned ?a ?o)  ;; holding the knob
    (HandEmpty ?a)
    (AtBConf ?q)
    (AtAConf ?a ?q)

    (CanMove)
    (CanPull)
    (CanUngrasp)

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
  )

  (:functions
    (MoveCost ?t)
    (PickCost)
    (PlaceCost)
  )

  (:action move_base
    :parameters (?q1 ?q2 ?t)
    :precondition (and (CanMove) (BaseMotion ?q1 ?t ?q2) ; (HandEmpty ?a)
                       (not (Identical ?q1 ?q2))
                       (AtBConf ?q1))
    :effect (and (AtBConf ?q2)
                 (not (AtBConf ?q1)) (not (CanMove))
                 ;(increase (total-cost) (MoveCost ?t))
                 (increase (total-cost) 1)
            )
  )

  (:action pick
    :parameters (?a ?o ?p ?g ?q ?t)
    :precondition (and (Kin ?a ?o ?p ?g ?q ?t) ;(Enabled)
                       (AtPose ?o ?p) (HandEmpty ?a)
                       (AtBConf ?q)
                       (not (UnsafeApproach ?o ?p ?g))
                       ;(not (UnsafeATraj ?t))
                       ;(not (UnsafeOTraj ?o ?g ?t))
                       ; (not (CanMove))
                       ; (not (Picked ?o))
                       )
    :effect (and (AtGrasp ?a ?o ?g) (CanMove) ; (Picked ?o)
                 (not (AtPose ?o ?p)) (not (HandEmpty ?a))
                 ;(increase (total-cost) (PickCost))
                 (increase (total-cost) 1)
            )
  )

  (:action place
    :parameters (?a ?o ?p ?g ?q ?t)
    :precondition (and (Kin ?a ?o ?p ?g ?q ?t) ;(Enabled)
                       (AtGrasp ?a ?o ?g) (AtBConf ?q)
                       (not (UnsafePose ?o ?p))
                       (not (UnsafeApproach ?o ?p ?g))
                       ;(not (UnsafeATraj ?t))
                       ;(not (UnsafeOTraj ?o ?g ?t))
                       ; (not (CanMove))
                       ; (not (Placed ?o))
                       )
    :effect (and (AtPose ?o ?p) (HandEmpty ?a) (CanMove) ;(Placed ?o)
                 (not (AtGrasp ?a ?o ?g))
                 ;(increase (total-cost) (PlaceCost))
                 (increase (total-cost) 1)
            )
  )

    (:action declare_store_in_space
      :parameters (?t ?r)
      :precondition (and (Space ?r)
                         (forall (?o) (imply (OfType ?o ?t) (In ?o ?r)))
                    )
      :effect (and (StoredInSpace ?t ?r))
    )

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

  (:action just-serve
    :parameters (?o ?r)
    :precondition (and (Stackable ?o ?r) (Plate ?r)
                       (On ?o ?r))
    :effect (and (Served ?o ?r))
  )

  (:action just-clean
    :parameters (?o ?s)
    :precondition (and (Edible ?o) (CleaningSurface ?s) (On ?o ?s)
                       )
    :effect (and (Cleaned ?o))
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
      :precondition (and (Joint ?o) (AConf ?a ?aq1)
                         (KinGraspHandle ?a ?o ?p ?g ?q ?aq2 ?t)
                         (AtPosition ?o ?p) (HandEmpty ?a)
                         (AtBConf ?q) (AtAConf ?a ?aq1)
                         ;(Enabled)
                    )
      :effect (and (AtHandleGrasp ?a ?o ?g) (not (HandEmpty ?a))
                   (not (CanMove)) (CanPull) (not (CanUngrasp))
                   (not (AtAConf ?a ?aq1)) (AtAConf ?a ?aq2)
                   ;(increase (total-cost) (PickCost)) ; TODO: make one third of the cost
                   (increase (total-cost) 0)
              )
    )
    (:action ungrasp_handle
      :parameters (?a ?o ?p ?g ?q ?aq1 ?aq2 ?t)
      :precondition (and (Joint ?o) (AtPosition ?o ?p)
                         (KinUngraspHandle ?a ?o ?p ?g ?q ?aq1 ?aq2 ?t)
                         (AtHandleGrasp ?a ?o ?g) (CanUngrasp)
                         (AtBConf ?q) (UngraspBConf ?q) (AtAConf ?a ?aq1) ;; (DefaultAConf ?a ?aq2)
                         ;(Enabled)
                    )
      :effect (and (GraspedHandle ?o) (HandEmpty ?a) (CanMove)
                   (not (AtHandleGrasp ?a ?o ?g))
                   (not (AtAConf ?a ?aq1)) (AtAConf ?a ?aq2)
                   ;(increase (total-cost) (PlaceCost))
                   (increase (total-cost) 0)
              )
    )

    ;; from fully closed position ?p1 pull to the fully open position ?p2
    ;(:action pull_drawer_handle
    ;  :parameters (?a ?o ?p1 ?p2 ?g ?q1 ?q2 ?bt)
    ;  :precondition (and (Drawer ?o) (not (= ?p1 ?p2)) (CanPull)
    ;                     (AtPosition ?o ?p1) (Position ?o ?p2)
    ;                     (KinPullDrawerHandle ?a ?o ?p1 ?p2 ?g ?q1 ?q2 ?bt)
    ;                     (AtHandleGrasp ?a ?o ?g) (AtBConf ?q1)
    ;                     ;(not (UnsafeBTraj ?bt))
    ;                )
    ;  :effect (and (not (CanPull)) (CanUngrasp)
    ;              (AtPosition ?o ?p2) (not (AtPosition ?o ?p1))
    ;              (AtBConf ?q2) (not (AtBConf ?q1))
    ;          )
    ;)

    ;; from fully closed position ?p1 pull to the fully open position ?p2
    (:action pull_door_handle
      :parameters (?a ?o ?p1 ?p2 ?g ?q1 ?q2 ?bt ?aq1 ?aq2 ?at)
      :precondition (and (Joint ?o) (not (= ?p1 ?p2)) (CanPull)
                         (AtPosition ?o ?p1) (Position ?o ?p2) (AtHandleGrasp ?a ?o ?g)
                         (KinPullDoorHandle ?a ?o ?p1 ?p2 ?g ?q1 ?q2 ?bt ?aq1 ?aq2 ?at)
                         (AtBConf ?q1) (AtAConf ?a ?aq1)
                         ;(not (UnsafeApproach ?o ?p2 ?g))
                         ;(not (UnsafeATraj ?at))
                         ;(not (UnsafeBTraj ?bt))
                         ;(Enabled)
                    )
      :effect (and (not (CanPull)) (CanUngrasp)
                  (AtPosition ?o ?p2) (not (AtPosition ?o ?p1))
                  (AtBConf ?q2) (not (AtBConf ?q1))
                  (AtAConf ?a ?aq2) (not (AtAConf ?a ?aq1))
                  (increase (total-cost) 1)
              )
    )

  ;(:action pull_door
  ; :parameters (?a ?o ?p1 ?p2 ?g ?q1 ?q2 ?bt ?aq1 ?aq2 ?at ?t1 ?aq3 ?t2)
  ; :precondition (and ; (Door ?o) (not (= ?p1 ?p2)) (CanPull)
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
    ;(:action turn_knob
    ;  :parameters (?a ?o ?p1 ?p2 ?g ?q ?aq1 ?aq2 ?at)
    ;  :precondition (and (Knob ?o) (not (= ?p1 ?p2)) (CanPull)
    ;                     (AtPosition ?o ?p1) (Position ?o ?p2) (AtHandleGrasp ?a ?o ?g)
    ;                     (KinTurnKnob ?a ?o ?p1 ?p2 ?g ?q ?aq1 ?aq2 ?at)
    ;                     (AtBConf ?q) (AtAConf ?a ?aq1)
    ;                     ;(not (UnsafeApproach ?o ?p2 ?g))
    ;                     ;(not (UnsafeATraj ?at))
    ;                     ;(not (UnsafeBTraj ?bt))
    ;                )
    ;  :effect (and (not (CanPull)) (CanUngrasp) (KnobTurned ?a ?o)
    ;              (AtPosition ?o ?p2) (not (AtPosition ?o ?p1))
    ;              (UngraspBConf ?q)
    ;              (AtAConf ?a ?aq2) (not (AtAConf ?a ?aq1))
    ;          )
    ;)

  (:derived (On ?o ?r)
    (exists (?p) (and (Supported ?o ?p ?r) (AtPose ?o ?p)))
  )
  (:derived (In ?o ?r)
    (exists (?p) (and (Contained ?o ?p ?r) (AtPose ?o ?p)))
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
  (:derived (UnsafeATraj ?t)
    (or
        (exists (?o2 ?p2) (and (ATraj ?t) (Pose ?o2 ?p2)
                               (not (CFreeTrajPose ?t ?o2 ?p2))
                               (AtPose ?o2 ?p2)))
        (exists (?o2 ?p2) (and (ATraj ?t) (Position ?o2 ?p2)
                               (not (CFreeTrajPosition ?t ?o2 ?p2))
                               (AtPosition ?o2 ?p2)))
    )
  )

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
)
