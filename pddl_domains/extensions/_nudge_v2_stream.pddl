(define (stream nudge_v2)

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
    :certified (and (Position ?o ?p2) (IsNudgedPosition ?o ?p2)
                    (IsSampledNudgedPosition ?o ?p1 ?p2))
  )

    (:stream inverse-reachability-nudge-door
      :inputs (?a ?o ?p ?g)
      :domain (and (Controllable ?a) (IsOpenedPosition ?o ?p) (NudgeGrasp ?o ?g))
      :outputs (?q)
      :certified (and (BConf ?q) (ReachPosition ?a ?o ?p ?g ?q))
    )

    (:stream inverse-kinematics-nudge-door
      :inputs (?a ?o ?p ?g ?q)
      :domain (ReachPosition ?a ?o ?p ?g ?q)
      :fluents (AtPosition)
      :outputs (?aq ?t)
      :certified (and (AConf ?a ?aq) (NudgeConf ?a ?o ?p ?g ?q ?aq)
                      (KinNudgeGrasp ?a ?o ?p ?g ?q ?aq ?t))
    )

    (:stream plan-base-nudge-door
      :inputs (?a ?o ?p1 ?p2 ?g ?q1 ?aq)
      :fluents (AtPosition)
      :domain (and (NudgeConf ?a ?o ?p1 ?g ?q1 ?aq) (IsSampledNudgedPosition ?o ?p1 ?p2))
      :outputs (?q2 ?bt)
      :certified (and (BConf ?q2) (BTraj ?bt) (KinNudgeDoor ?a ?o ?p1 ?p2 ?g ?q1 ?q2 ?aq ?bt))
    )

)