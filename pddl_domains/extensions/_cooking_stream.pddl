(define (stream cooking)

  (:stream sample-pose-sprinkle
    :inputs (?o1 ?p1 ?o2)
    :domain (and (Region ?o1) (Pose ?o1 ?p1) (Graspable ?o2))
    :outputs (?p2)
    :certified (and (Pose ?o2 ?p2) (SprinklePose ?o1 ?p1 ?o2 ?p2))
  )

  (:stream sample-nudge-grasp
    :inputs (?o)
    :domain (Door ?o)
    :outputs (?g)
    :certified (NudgeGrasp ?o ?g)
  )

  (:stream get-joint-position-nudged-open
    :inputs (?o ?p1)
    :domain (and (Door ?o) (Position ?o ?p1) (IsOpenedPosition ?o ?p1))
    :outputs (?p2)
    :certified (and (Position ?o ?p2) (IsNudgedPosition ?o ?p2) (IsSampledNudgedPosition ?o ?p1 ?p2))
  )
  ;(:stream get-joint-position-nudged-closed
  ;  :inputs (?o ?p1)
  ;  :domain (and (Door ?o) (Position ?o ?p1) (IsNudgedPosition ?o ?p1))
  ;  :outputs (?p2)
  ;  :certified (and (Position ?o ?p2) (IsOpenedPosition ?o ?p2) (IsSampledNudgedPosition ?o ?p1 ?p2))
  ;)

    (:stream inverse-kinematics-nudge-door
      :inputs (?a ?o ?p ?g)
      :domain (and (Controllable ?a) (IsOpenedPosition ?o ?p) (NudgeGrasp ?o ?g))
      :outputs (?q ?aq ?t)
      :certified (and (BConf ?q) (AConf ?a ?aq) (ATraj ?t)
                      (NudgeConf ?a ?o ?p ?g ?q ?aq)
                      (KinNudgeGrasp ?a ?o ?p ?g ?q ?aq ?t))
    )

    (:stream plan-base-nudge-door
      :inputs (?a ?o ?p1 ?p2 ?g ?q1 ?aq)
      :domain (and (NudgeConf ?a ?o ?p1 ?g ?q1 ?aq) (IsSampledNudgedPosition ?o ?p1 ?p2))
      :outputs (?q2 ?bt)
      :certified (and (BConf ?q2) (BTraj ?bt) (KinNudgeDoor ?a ?o ?p1 ?p2 ?g ?q1 ?q2 ?aq))
    )

)