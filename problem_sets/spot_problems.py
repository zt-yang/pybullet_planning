from world_builder.loaders import *


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
