(define (stream nudge_v1)

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

    (:stream inverse-kinematics-nudge-door
      :inputs (?a ?o ?p ?g)
      :domain (and (Controllable ?a) (IsOpenedPosition ?o ?p) (NudgeGrasp ?o ?g))
      :outputs (?q ?aq ?t)
      :certified (and (BConf ?q) (AConf ?a ?aq)
                      (NudgeConf ?a ?o ?p ?g ?q ?aq)
                      (KinNudgeGrasp ?a ?o ?p ?g ?q ?aq ?t))
    )

    (:stream plan-base-nudge-door
      :inputs (?a ?o ?p1 ?p2 ?g ?q1 ?aq)
      :domain (and (NudgeConf ?a ?o ?p1 ?g ?q1 ?aq) (IsSampledNudgedPosition ?o ?p1 ?p2))
      ;:fluents (AtPosition)
      :outputs (?q2 ?bt)
      :certified (and (BConf ?q2) (BTraj ?bt) (KinNudgeDoor ?a ?o ?p1 ?p2 ?g ?q1 ?q2 ?aq ?bt))
    )

  ;(:stream sample-nudge-back-grasp
  ;  :inputs (?o)
  ;  :domain (Door ?o)
  ;  :outputs (?g)
  ;  :certified (NudgeBackGrasp ?o ?g)
  ;)

    ;(:stream inverse-kinematics-nudge-door-back
    ;  :inputs (?a ?o ?p ?g)
    ;  :domain (and (Controllable ?a) (IsNudgedPosition ?o ?p) (NudgeBackGrasp ?o ?g))
    ;  :outputs (?q ?aq ?t)
    ;  :certified (and (BConf ?q) (AConf ?a ?aq) (ATraj ?t)
    ;                  (NudgeBackConf ?a ?o ?p ?g ?q ?aq)
    ;                  (KinNudgeBackGrasp ?a ?o ?p ?g ?q ?aq ?t))
    ;)

    ;(:stream plan-base-nudge-door-back
    ;  :inputs (?a ?o ?p1 ?p2 ?g ?q1 ?aq)
    ;  :domain (and (NudgeBackConf ?a ?o ?p1 ?g ?q1 ?aq) (IsSampledNudgedPosition ?o ?p2 ?p1))
    ;  :outputs (?q2 ?bt)
    ;  :certified (and (BConf ?q2) (BTraj ?bt) (KinNudgeBackDoor ?a ?o ?p1 ?p2 ?g ?q1 ?q2 ?aq ?bt))
    ;)

)