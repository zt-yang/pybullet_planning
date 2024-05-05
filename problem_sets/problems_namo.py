from pybullet_tools.pr2_streams import get_bconf_in_region_gen

from world_builder.loaders_namo import load_rooms, load_cart
from world_builder.world_utils import visualize_point

from robot_builder.robot_builders import build_namo_domain_robot

from problem_sets.problem_utils import *


def visualize_sampled_bconf(state, area):
    funk = get_bconf_in_region_gen(state)(area)
    for i in range(10):
        bq = next(funk)[0]
        print(i, bq)
        if bq is not None:
            visualize_point(bq.values[:2], state.world)


#######################################################


def test_navigation(args, **kwargs):

    def loader_fn(world, **world_builder_args):
        load_rooms(world)

        location = world.name_to_body('kitchen')
        # visualize_sampled_bconf(state, location)

        goals = [('RobInRoom', location)]

        return goals

    return problem_template(args, robot_builder_fn=build_namo_domain_robot,
                            world_loader_fn=loader_fn, **kwargs)


def test_cart_pull(args, **kwargs):

    def loader_fn(world, **world_builder_args):
        load_rooms(world)

        cart, marker = load_cart(world)

        goals = ('test_marker_grasp', marker)

        # goals = [('AtBConf', Conf(robot, get_group_joints(robot, 'base'), (1.204, 0.653, -2.424)))]
        # goals = [('HoldingMarker', 'left', marker)]

        ## --- test `plan-base-pull-marker-random`
        # goals = [("PulledMarker", marker)]

        ## --- test `plan-base-pull-marker-to-bconf`
        # goals = [('HoldingMarker', 'left', marker), ('RobInRoom', world.name_to_body('laundry_room'))]

        ## --- test `grasp` + `pull` + `ungrasp`
        # goals = [('GraspedMarker', marker)]

        ## --- test `plan-base-pull-marker-to-pose`
        # goals = [('InRoom', marker, world.name_to_body('laundry_room'))]

        return goals

    return problem_template(args, robot_builder_fn=build_namo_domain_robot,
                            world_loader_fn=loader_fn, **kwargs)
