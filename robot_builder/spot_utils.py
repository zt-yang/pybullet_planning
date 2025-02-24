from pybullet_tools.utils import LockRenderer, HideOutput, load_model, link_from_name, \
    get_aabb, get_aabb_center, draw_aabb, joint_from_name, PI
from pybullet_tools.bullet_utils import load_robot_urdf
from pybullet_tools.pr2_primitives import Conf

from robot_builder.robot_utils import get_robot_group_joints, solve_leg_conf

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


def solve_spot_leg_conf(body, torso_lift_value, **kwargs):
    zO = 0.739675
    lA = 0.3058449991941452
    lB = 0.3626550008058548  ## 0.3826550008058548

    def fn(aA, aB):
        return [aA, aB] * 4
    # zO, lA, lB = compute_link_lengths(body, "fl.hip", "fl.toe", "fl.uleg", "fl.lleg")
    return solve_leg_conf(body, torso_lift_value, zO, lA, lB, SPOT_JOINT_GROUPS,
                          get_leg_positions_from_hip_knee_angles=fn, **kwargs)
