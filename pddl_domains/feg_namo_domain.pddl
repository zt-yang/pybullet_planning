(define (domain fe-gripper-tamp)
  (:requirements :strips :equality)

  (:constants
    @movable @bottle @edible @medicine
  )


  (:predicates

    (Drawer ?o)
    (Door ?o)
    (Knob ?o)
    (Joint ?o)

    (Edible ?o)
    (CleaningSurface ?s)
    (HeatingSurface ?s)
    (ControlledBy ?s ?n)

    (Controllable ?a)
    (HandEmpty ?a)
    (SEConf ?q)
    (Pose ?o ?p)
    (Position ?o ?p)
    (IsOpenedPosition ?o ?p)
    (IsClosedPosition ?o ?p)
    (Grasp ?o ?g)
    (HandleGrasp ?o ?g)

    (Graspable ?o)
    (Stackable ?o ?r)
    (Containable ?o ?r)

    (OriginalSEConf ?q)

    (Kin ?a ?o ?p ?g ?q ?t)
    (KinHandle ?a ?o ?p ?g ?q1)
    (KinGraspHandle ?a ?o ?p ?g ?q1 ?t)
    (KinPullDoorHandle ?a ?o ?p1 ?p2 ?g ?q1 ?q2 ?t)

    (FreeMotion ?p1 ?t ?p2)
    (Supported ?o ?p ?r)
    (Contained ?o ?p ?s)
    (Traj ?t)

    (TrajPoseCollision ?t ?o ?p)
    (CFreePosePose ?o ?p ?o2 ?p2)
    (CFreeApproachPose ?o ?p ?g ?o2 ?p2)
    (CFreeTrajPose ?t ?o2 ?p2)

    (AtSEConf ?q)
    (AtPose ?o ?p)
    (AtPosition ?o ?p)
    (OpenPosition ?o ?p)
    (ClosedPosition ?o ?p)

    (AtGrasp ?a ?o ?g)
    (AtHandleGrasp ?a ?o ?g)
    (HandleGrasped ?a ?o)

    (CanMove)
    (CanPull ?a)
    (CanUngrasp)
    (Cleaned ?o)
    (Cooked ?o)
    (OpenedJoint ?o)
    (ClosedJoint ?o)
    (GraspedHandle ?o)

    (On ?o ?r)
    (In ?o ?r) ;;
    (Holding ?a ?o)

    (UnsafePose ?o ?p)
    (UnsafeApproach ?o ?p ?g)
    (UnsafeTraj ?t)

    (PoseObstacle ?o ?p ?o2)
    (ApproachObstacle ?o ?p ?g ?o2)

    (Debug1)
    (Debug2)
    (Debug3)
    (Debug4)

    (OfType ?o ?t)
    (StoredInSpace ?t ?r)
    (Space ?r)
    (ContainObj ?o)
    (AtAttachment ?o ?j)
    (NewPoseFromAttachment ?o ?p)

    (Cleaned ?o)
    (Cooked ?o)

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
    ; (MoveCost ?t)
    (PickCost)
    (PlaceCost)
  )

  (:action move_cartesian
    :parameters (?q1 ?q2 ?t)
    :precondition (and (CanMove) (AtSEConf ?q1)
                       (FreeMotion ?q1 ?t ?q2)
                       ; (not (UnsafeTraj ?t))
                   )
    :effect (and (AtSEConf ?q2) (not (AtSEConf ?q1))
                 (not (CanMove))
                 ; (increase (total-cost) (MoveCost ?t))
                 (increase (total-cost) 1)
                 )
  )

  (:action pick_hand
    :parameters (?a ?o ?p ?g ?q ?t)
    :precondition (and (Kin ?a ?o ?p ?g ?q ?t) (HandEmpty ?a)
                       (AtPose ?o ?p)
                       (AtSEConf ?q)
                       ; (not (CanMove))
                       ; (not (UnsafeTraj ?t))
                       (not (UnsafeApproach ?o ?p ?g))
                       )
    :effect (and (AtGrasp ?a ?o ?g) (CanMove)
                 (not (AtPose ?o ?p)) (not (HandEmpty ?a))
                 ; (increase (total-cost) (PickCost))
                 (increase (total-cost) 1)
                 )
  )

  (:action place_hand
    :parameters (?a ?o ?p ?g ?q ?t)
    :precondition (and (Kin ?a ?o ?p ?g ?q ?t)
                       (AtGrasp ?a ?o ?g)
                       (AtSEConf ?q)
                       ; (not (CanMove))
                       ; (not (UnsafeTraj ?t))
                       (not (UnsafePose ?o ?p))
                       (not (UnsafeApproach ?o ?p ?g))
                       )
    :effect (and (AtPose ?o ?p) (CanMove) (HandEmpty ?a)
                 (not (AtGrasp ?a ?o ?g))
                 ; (increase (total-cost) (PlaceCost))
                 (increase (total-cost) 1)
                 )
  )

    ;; including grasping and pulling, q1 is approach grasp for p1, q2 is approach grasp for p2
    (:action grasp_pull_handle
      :parameters (?a ?o ?p1 ?p2 ?g ?q1 ?q2 ?t1 ?t2 ?t3)
      :precondition (and (Joint ?o) (HandEmpty ?a) ; (CanPull ?a)
                         (AtSEConf ?q1)
                         (AtPosition ?o ?p1) (Position ?o ?p2) (not (= ?p1 ?p2))
                         (KinGraspHandle ?a ?o ?p1 ?g ?q1 ?t1)
                         (KinPullDoorHandle ?a ?o ?p1 ?p2 ?g ?q1 ?q2 ?t2)
                         (KinGraspHandle ?a ?o ?p2 ?g ?q2 ?t3)
                         ; (not (UnsafeTraj ?t))
                         ; (not (UnsafePose ?o ?p))
                         ; (not (UnsafeApproach ?o ?p ?g))
                    )
      :effect (and (GraspedHandle ?o)
                  (CanMove) ; (not (CanPull ?a))
                  (AtSEConf ?q2) (not (AtSEConf ?q1))
                  (AtPosition ?o ?p2) (not (AtPosition ?o ?p1))
                  (increase (total-cost) 1)
              )
    )

    ;; with attachment
    (:action pull_articulated_handle_attachment
      :parameters (?a ?o ?p1 ?p2 ?g ?q1 ?q2 ?t ?o3 ?p3 ?p4)
      :precondition (and (Joint ?o) (not (= ?p1 ?p2)) (CanPull ?a)
                         (AtPosition ?o ?p1) (Position ?o ?p2) (AtHandleGrasp ?a ?o ?g)
                         (KinPullDoorHandle ?a ?o ?p1 ?p2 ?g ?q1 ?q2 ?t)
                         (ContainObj ?o3) (AtPose ?o3 ?p3) (Pose ?o3 ?p4)
                         (AtAttachment ?o3 ?o) (NewPoseFromAttachment ?o3 ?p4)
                    )
      :effect (and (not (CanPull ?a)) (CanUngrasp)
                  (AtPosition ?o ?p2) (not (AtPosition ?o ?p1))
                  (not (AtPose ?o3 ?p3)) (AtPose ?o3 ?p4)
              )
    )

    (:action declare_store_in_space
      :parameters (?t ?r)
      :precondition (and (Space ?r)
                         (forall (?o) (imply (OfType ?o ?t) (In ?o ?r)))
                    )
      :effect (and (StoredInSpace ?t ?r))
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

  (:derived (On ?o ?r)
    (exists (?p) (and (Supported ?o ?p ?r) (AtPose ?o ?p)))
  )
  (:derived (In ?o ?r)
    (exists (?p) (and (Contained ?o ?p ?r) (AtPose ?o ?p)))
  )
  (:derived (Holding ?a ?o)
    (exists (?g) (and (Grasp ?o ?g) (AtGrasp ?a ?o ?g)))
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
    (exists (?hg) (and (Joint ?o) (HandleGrasp ?o ?hg)
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

  ;; in order to use fluent streams
  ;(:derived (UnsafeTraj ?t)
  ;  (exists (?o2 ?p2) (and (Traj ?t) (Pose ?o2 ?p2) (AtPose ?o2 ?p2)
  ;                         (not (CFreeTrajPose ?t ?o2 ?p2))))
  ;)

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