from pybullet_tools.pr2_primitives import Conf

from world_builder.loaders import *
from world_builder.loaders_office import load_one_office

from robot_builder.robot_builders import build_table_domain_robot

from problem_sets.problem_utils import problem_template


def test_spot_pick(args, **kwargs):
    def loader_fn(world, **world_builder_args):
        robot = world.robot

        xy = (2, 2)
        table = create_table(world, xy=xy, h=0.6)
        cabbage = create_movable(world, supporter=table, xy=xy)
        set_camera_target_body(table, dx=1.5, dy=1.5, dz=1.5)

        arm = robot.arms[0]
        goals = [('AtBConf', Conf(robot, robot.get_base_joints(), (2, 0, 0, 0)))]
        # goals = ("test_object_grasps", cabbage)
        # goals = [("Holding", arm, cabbage)]

        return {'goals': goals}

    return problem_template(args, robot_builder_fn=build_table_domain_robot,
                            world_loader_fn=loader_fn, **kwargs)


def test_office_chairs(args, **kwargs):
    def loader_fn(world, **world_builder_args):
        chairs, tables = load_one_office(world)
        set_camera_target_body(chairs[0], dx=1.5, dy=1.5, dz=1.5)

        goals = []

        return goals

    return problem_template(args, robot_builder_fn=build_table_domain_robot,
                            world_loader_fn=loader_fn, **kwargs)
