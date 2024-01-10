from world_builder.builders import *
from world_builder.loaders import *
from world_builder.robot_builders import build_table_domain_robot, build_robot_from_args, \
    build_fridge_domain_robot
from world_builder.utils import load_asset, FLOOR_HEIGHT, WALL_HEIGHT, visualize_point
from world_builder.world_generator import to_lisdf
from world_builder.paths import KITCHEN_WORLD

from os.path import join, abspath
import numpy as np
import sys
import math
import random

from problem_utils import create_world, pddlstream_from_state_goal, save_to_kitchen_worlds, \
    test_template


def test_pick_low(args, **kwargs):
    def loader_fn(world, **world_builder_args):
        xy = (2, 2)
        arm = world.robot.arms[0]
        table = create_table(world, xy=xy, h=0.6)
        cabbage = create_movable(world, supporter=table, xy=xy)
        set_camera_target_body(table, dx=1.5, dy=1.5, dz=1.5)

        # goals = [('AtBConf', Conf(robot, get_group_joints(robot, 'base'), (2, 7, 0)))]
        # goals = ("test_grasps", cabbage)
        goals = [("Holding", arm, cabbage)]

        return goals
    return test_simple_table_domain(args, loader_fn, **kwargs)
