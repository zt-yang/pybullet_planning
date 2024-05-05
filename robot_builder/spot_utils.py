from pybullet_tools.utils import LockRenderer, HideOutput, load_model, link_from_name, \
    get_aabb, get_aabb_center, draw_aabb, joint_from_name, PI
from pybullet_tools.bullet_utils import load_robot_urdf
from pybullet_tools.pr2_primitives import Conf

from robot_builder.robot_utils import get_robot_group_joints

import time
import math

SPOT_URDF = "models/spot_description/model.urdf"
SPOT_GRIPPER_ROOT = "arm0.link_wr1"
SPOT_TOOL_LINK = "arm0.link_wr1"
SPOT_FINGER_LINK = "arm0.link_fngr"
SPOT_JOINT_GROUPS = {
    'base-torso': ['x', 'y', 'torso_lift_joint', 'theta'],
    'arm': ['arm0.sh0', 'arm0.sh1', 'arm0.hr0', 'arm0.el0', 'arm0.el1', 'arm0.wr0', 'arm0.wr1'],
    'leg': ['fl.hy', 'fl.kn', 'fr.hy', 'fr.kn', 'hl.hy', 'hl.kn', 'hr.hy', 'hr.kn'],
    'gripper': ['arm0.f1x']
}
SPOT_REST_ARM_CONF = (0, -PI, 0, PI, 0, 0, 0)
SPOT_CARRY_ARM_CONF = (0, -PI*0.75, 0, 1.83, 0, 1.83, 0)


def load_spot():
    return load_robot_urdf(SPOT_URDF)


def solve_leg_conf(body, torso_lift_value, verbose=True):
    from sympy import Symbol, Eq, solve, cos, sin

    zO = 0.739675
    lA = 0.3058449991941452
    lB = 0.3626550008058548  ## 0.3826550008058548

    aA = Symbol('aA', real=True)
    # aB = Symbol('aB', real=True)
    aBB = Symbol('aBB', real=True)  ## pi = (pi/2 - aA) + aBB - aB
    e1 = Eq(lA * cos(aA) + lB * sin(aBB), zO + torso_lift_value)
    e2 = Eq(lA * sin(aA), lB * cos(aBB))

    start = time.time()
    solutions = solve([e1, e2], aA, aBB)  ## , e2, e3, e4, e5
    if verbose:
        print(f'solve_leg_conf in : {round(time.time() - start, 2)} sec')

    solutions = [r for r in solutions if r[0] > 0 and r[1] > 0]
    if len(solutions) == 0:
        return None

    aA, aBB = solutions[0]
    aB = - math.pi/2 - aA + aBB
    joint_values = [aA, aB] * 4
    joint_names = get_robot_group_joints(body, 'leg', SPOT_JOINT_GROUPS)
    joints = [joint_from_name(body, name) for name in joint_names]
    conf = Conf(body, joints, joint_values)
    return conf


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