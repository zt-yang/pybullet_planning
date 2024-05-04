import random

from pybullet_tools.pr2_primitives import Conf, get_group_joints
from pybullet_tools.utils import invert, get_name, pairwise_collision, \
    get_link_pose, get_pose, set_pose, wait_if_gui
from pybullet_tools.pose_utils import sample_pose, xyzyaw_to_pose
from pybullet_tools.bullet_utils import nice, collided, equal

from world_builder.world_utils import sort_body_indices
from world_builder.loaders import *


def load_one_office(world):
    asset_renaming = {'Chair': 'OfficeChair'}
    world.set_skip_joints()
    floor = load_floor_plan(world, plan_name='office_1.svg', asset_renaming=asset_renaming,
                            debug=True, spaces=None, surfaces=None, random_instance=False, verbose=True)
    world.remove_object(floor)
    wait_if_gui()
    return [], []

