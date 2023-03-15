from pybullet_tools.utils import LockRenderer, HideOutput, load_model, link_from_name, \
    get_aabb, get_aabb_center, draw_aabb

SPOT_URDF = "models/spot_description/model.urdf"
SPOT_TOOL_LINK = "arm0.link_wr1"
SPOT_JOINT_GROUPS = {
    'base': ['x', 'y', 'theta', 'torso_lift_joint'],
    'leg': ['fl.hy', 'fl.kn', 'fr.hy', 'fr.kn', 'hl.hy', 'hl.kn', 'hr.hy', 'hr.kn']
}


def load_spot():
    with LockRenderer():
        with HideOutput():
            pr2 = load_model(SPOT_URDF, fixed_base=False)
    return pr2


def compute_link_lengths(body):
    """
    -- o O -----------  body of robot (zO)
       |\
     aA  \ lA
          o A   ------  knee of robot
         / \
     lB / aB
      o B   ----------  toe of robot
    """

    hip_link = link_from_name(body, "fl.hip")
    hip_aabb = get_aabb(body, link=hip_link)
    zO = get_aabb_center(hip_aabb)[2]

    toe_link = link_from_name(body, "fl.toe")
    zB = get_aabb_center(get_aabb(body, link=toe_link))[2]

    upper_leg_link = link_from_name(body, "fl.uleg")
    upper_leg_aabb = get_aabb(body, link=upper_leg_link)
    zA1 = upper_leg_aabb.lower[2]

    lower_leg_link = link_from_name(body, "fl.lleg")
    lower_leg_aabb = get_aabb(body, link=lower_leg_link)
    zA2 = lower_leg_aabb.upper[2]

    zA = (zA1 + zA2) / 2
    lA = zO - zA
    lB = zA - zB
    return lA, lB