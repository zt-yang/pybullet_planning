from pybullet_tools.spot_utils import SPOT_JOINT_GROUPS
from pybullet_tools.pr2_primitives import Conf
from pybullet_tools.utils import joint_from_name
import time
import math

from sympy import *


def solve_leg_conf(body, torso_lift_value, verbose=True):
    zO = 0.739675
    lA = 0.3058449991941452
    lB = 0.3626550008058548  ## 0.3826550008058548
    # 'fl.toe' 'fl.hy'

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
    joint_names = SPOT_JOINT_GROUPS['leg']
    joints = [joint_from_name(body, name) for name in joint_names]
    conf = Conf(body, joints, joint_values)
    return conf