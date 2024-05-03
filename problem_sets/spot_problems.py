from pybullet_tools.pr2_primitives import Conf

from world_builder.loaders import *

from robot_builder.robot_builders import build_table_domain_robot

from problem_sets.problem_utils import problem_template


def test_pick_low(args, **kwargs):
    def loader_fn(world, **world_builder_args):
        robot = world.robot

        xy = (2, 2)
        table = create_table(world, xy=xy, h=0.6)
        cabbage = create_movable(world, supporter=table, xy=xy)
        set_camera_target_body(table, dx=1.5, dy=1.5, dz=1.5)

        arm = robot.arms[0]
        goals = [('AtBConf', Conf(robot, robot.get_base_joints(), (-2, -2, 0.5, 0)))]
        # goals = ("test_grasps", cabbage)
        # goals = [("Holding", arm, cabbage)]

        return {'goals': goals}

    return problem_template(args, robot_builder_fn=build_table_domain_robot,
                            world_loader_fn=loader_fn, **kwargs)


def test_office_chairs(args, **kwargs):
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

    return problem_template(args, robot_builder_fn=build_table_domain_robot,
                            world_loader_fn=loader_fn, **kwargs)
