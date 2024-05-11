(define (stream fe-gripper-tamp)

  (:stream sample-pose-on
    :inputs (?o ?r)
    :domain (Stackable ?o ?r)
    :outputs (?p)
    :certified (and (Pose ?o ?p) (Supported ?o ?p ?r))
  )
  (:stream sample-pose-in
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
  (:stream inverse-kinematics-hand
    :inputs (?a ?o ?p ?g)
    :domain (and (Controllable ?a) (Pose ?o ?p) (Grasp ?o ?g))
    :outputs (?q ?t)
    :certified (and (SEConf ?q) (Traj ?t) (Kin ?a ?o ?p ?g ?q ?t))
  )
  (:stream plan-free-motion-hand
    :inputs (?q1 ?q2)
    :domain (and (SEConf ?q1) (SEConf ?q2))
    :fluents (AtPose AtGrasp AtSEConf AtPosition)
    :outputs (?t)
    :certified (and (Traj ?t) (FreeMotion ?q1 ?t ?q2))
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

  ;(:stream test-cfree-traj-pose
  ;  :inputs (?t ?o2 ?p2)
  ;  :domain (and (Traj ?t) (Pose ?o2 ?p2))
  ;  :certified (CFreeTrajPose ?t ?o2 ?p2)
  ;)

  (:stream get-joint-position-open
    :inputs (?o ?p1)
    :domain (and (Joint ?o) (Position ?o ?p1) (IsClosedPosition ?o ?p1))
    :outputs (?p2)
    :certified (and (Position ?o ?p2) (IsOpenedPosition ?o ?p2))
  )
    (:stream sample-handle-grasp
      :inputs (?o)
      :domain (Joint ?o)
      :outputs (?g)
      :certified (HandleGrasp ?o ?g)
    )
    (:stream inverse-kinematics-grasp-handle
      :inputs (?a ?o ?p ?g)
      :domain (and (Controllable ?a) (Position ?o ?p) (HandleGrasp ?o ?g))
      :outputs (?q1 ?t)
      :certified (and (SEConf ?q1) (Traj ?t) (KinGraspHandle ?a ?o ?p ?g ?q1 ?t) (KinHandle ?a ?o ?p ?g ?q1))
    )
    (:stream plan-grasp-pull-handle
      :inputs (?a ?o ?pst1 ?pst2 ?g ?q1 ?q2)
      :domain (and (Controllable ?a) (Joint ?o) (Position ?o ?pst1) (Position ?o ?pst2)
                   (HandleGrasp ?o ?g) (SEConf ?q1) (SEConf ?q2)
                   (KinHandle ?a ?o ?pst1 ?g ?q1) (KinHandle ?a ?o ?pst2 ?g ?q2)
                   (IsClosedPosition ?o ?pst1) (IsOpenedPosition ?o ?pst2))
      :outputs (?t)
      :certified (and (Traj ?t) (KinPullDoorHandle ?a ?o ?pst1 ?pst2 ?g ?q1 ?q2 ?t))
    )

  ;; -------- already put those possible NewPoseFromAttachment in init -----------
  ;(:stream get-pose-from-attachment
  ;  :inputs (?o)
  ;  :domain (and (Graspable ?o))
  ;  :outputs (?p)
  ;  :certified (and (Pose ?o ?p) (NewPoseFromAttachment ?o ?p))
  ;)
  ;; ----------------------------------------------------------

  ;(:function (MoveCost ?t)
  ;  (and (Traj ?t))
  ;)

  ;;----------------------------------------------------------------------
  ;;      extended streams from _namo_stream.pddl
  ;;----------------------------------------------------------------------

    (:stream sample-marker-grasp
      :inputs (?o)
      :domain (and (Marker ?o))
      :outputs (?g)
      :certified (MarkerGrasp ?o ?g)
    )
    (:stream sample-marker-pose
      :inputs (?o ?p1)
      :domain (and (Marker ?o) (Pose ?o ?p1))
      :outputs (?p2)
      :certified (Pose ?o ?p2)
    )

    (:stream inverse-kinematics-grasp-marker
      :inputs (?a ?o ?p ?g)
      :domain (and (Controllable ?a) (Pose ?o ?p) (MarkerGrasp ?o ?g))
      :outputs (?q ?t)
      :certified (and (BConf ?q) (ATraj ?t) (KinGraspMarker ?a ?o ?p ?g ?q ?t))
    )
    (:stream inverse-kinematics-ungrasp-marker
      :inputs (?a ?o ?p ?g ?q)
      :domain (and (Controllable ?a) (Pose ?o ?p) (MarkerGrasp ?o ?g) (BConf ?q))
      :outputs (?t)
      :certified (and (ATraj ?t) (KinUngraspMarker ?a ?o ?p ?g ?q ?t))
    )
    (:stream plan-base-pull-marker-to-pose
      :inputs (?a ?o ?p1 ?p2 ?g ?q1 ?o2 ?p3)
      :domain (and
        (Controllable ?a) (Marker ?o) (Pose ?o ?p1) (Pose ?o ?p2) (MarkerGrasp ?o ?g)
        (BConf ?q1) (Cart ?o2) (Pose ?o2 ?p3)
      )
      :outputs (?q2 ?p4 ?t)
      :certified (and (BConf ?q2) (Pose ?o2 ?p4) (BTraj ?t) (KinPullMarkerToPose ?a ?o ?p1 ?p2 ?g ?q1 ?q2 ?o2 ?p3 ?p4 ?t))
    )

  (:stream sample-bconf-in-location
    :inputs (?r)
    :domain (and (Location ?r))
    :outputs (?q)
    :certified (and (BConf ?q) (BConfInLocation ?q ?r))
  )
  (:stream sample-pose-in-location
    :inputs (?o ?r)
    :domain (and (Moveable ?o) (Location ?r))
    :outputs (?p)
    :certified (and (Pose ?o ?p) (PoseInLocation ?o ?p ?r))
  )

  ;(:stream test-bconf-in-location
  ;  :inputs (?q ?r)
  ;  :domain (and (BConf ?q) (Location ?r))
  ;  :certified (BConfInLocation ?q ?r)
  ;)
  (:stream test-pose-in-location
    :inputs (?o ?p ?r)
    :domain (and (Pose ?o ?p) (Location ?r))
    :certified (PoseInLocation ?o ?p ?r)
  )


)