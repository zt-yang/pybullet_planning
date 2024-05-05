(define (stream namo)

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