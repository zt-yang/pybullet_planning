from world_builder.loaders import *

from robot_builder.robot_builders import build_table_domain_robot

from problem_sets.problem_utils import problem_template


####################################################


def test_pick(args, robot_builder_args=dict(), **kwargs):
    def loader_fn(world, **world_builder_args):
        test_object_grasps_fn = world_builder_args['test_grasp_fn'] if 'test_grasp_fn' else "test_object_grasps"

        xy = world_builder_args['xy'] if 'xy' in world_builder_args else (2, 2)
        w = world_builder_args['w'] if 'w' in world_builder_args else 0.25
        h = world_builder_args['h'] if 'h' in world_builder_args else 0.9
        movable_category = world_builder_args['movable_category'] \
            if 'movable_category' in world_builder_args else 'VeggieCabbage'

        arm = world.robot.arms[0]
        floor = create_floor_covering_base_limits(world)
        table = create_table(world, xy=xy, w=w, h=h)
        cabbage = create_movable(world, supporter=table, xy=xy, movable_category=movable_category)
        set_camera_target_body(table, dx=1.5, dy=1.5, dz=1.5)

        # goals = [('AtBConf', Conf(robot, get_group_joints(robot, 'base'), (2, 7, 0)))]
        goals = (test_object_grasps_fn, cabbage)
        # goals = [("Holding", arm, cabbage)]

        # ## debug
        # if not world.robot.move_base:
        #     goals = ("test_reachable_bconf", (arm, cabbage))

        return goals

    if 'base_q' in robot_builder_args:
        robot_builder_args['base_q'][:2] = [0, 0]
        if 'move_base' in robot_builder_args and not robot_builder_args['move_base']:
            robot_builder_args['base_q'][:2] = [2, 1]
            robot_builder_args['base_q'][-1:] = [PI/2]

    return problem_template(args, robot_builder_fn=build_table_domain_robot,
                            robot_builder_args=robot_builder_args,
                            world_loader_fn=loader_fn, **kwargs)


def test_small_sink(args, **kwargs):
    def loader_fn(world, **world_builder_args):
        table = create_table(world, xy=(4, 6))
        cabbage = create_movable(world, supporter=table, xy=(4, 6))
        sink = create_table(world, xy=(2, 5), color=(.25, .25, .75, 1), w=0.125, name='sink')
        turkey = create_movable(world, supporter=sink, xy=(2, 5), movable_category='MeatTurkeyLeg', name='turkey')

        goals = ("test_object_grasps", cabbage)
        goals = [('Holding', 'left', cabbage)]
        goals = [('Holding', 'left', turkey)]
        goals = [('On', cabbage, sink)]
        return goals
    return problem_template(args, robot_builder_fn=build_table_domain_robot, world_loader_fn=loader_fn, **kwargs)


def test_plated_food(args, **kwargs):
    def loader_fn(world, **world_builder_args):
        table = create_table(world, xy=(2, 1))
        set_camera_target_body(table, dx=1.5, dy=1.5, dz=1.5)
        sink = create_table(world, xy=(2, -2), color=(.25, .25, .75, 1), w=0.125, name='sink')
        plate = create_movable(world, supporter=sink, xy=(2, -2), movable_category='Plate', name='plate')
        sink.attach_obj(plate, world=world)
        # cabbage = create_movable(world, supporter=plate, xy=(2, 5))
        # plate.attach_obj(cabbage)

        # goals = [('Debug1', cabbage, plate)]
        # goals = ('test_pose_kin', (cabbage, plate))
        # goals = [("Holding", 'left', cabbage)]
        goals = ("test_object_grasps", plate)
        # goals = [("Holding", 'left', plate)]
        # goals = [("On", cabbage, table)]
        # goals = [("On", plate, table)]

        return goals
    return problem_template(args, robot_builder_fn=build_table_domain_robot, world_loader_fn=loader_fn, **kwargs)


####################################################


def test_five_tables_domain(args, world_loader_fn, **kwargs):
    if 'feg' in args.domain_pddl:
        robot_builder_args = {'robot_name': 'feg'}
    else:
        robot_builder_args = {'robot_name': 'pr2'}
    return problem_template(args, robot_builder_fn=build_table_domain_robot, robot_builder_args=robot_builder_args,
                            world_loader_fn=world_loader_fn, **kwargs)


def test_five_tables(args, **kwargs):
    def loader_fn(world, **world_builder_args):
        arm = world.robot.arms[0]
        cabbage, egg, plate, salter, sink, stove, counter, table = load_five_table_scene(world)

        ## suddenly stopped working
        # goals = [('Holding', arm, plate)]
        # goals = [('On', plate, stove)]
        # goals = [('On', cabbage, stove), ('On', plate, table)]

        goals = ("test_object_grasps", plate)
        goals = [('Holding', arm, cabbage)]
        goals = [('On', cabbage, stove)]
        goals = [('On', cabbage, stove), ('On', salter, table)]
        # goals = [('Cleaned', cabbage)]
        # goals = [('Cooked', cabbage)]
        # goals = [('Seasoned', cabbage)]
        # goals = [('ExistOmelette', table)]

        # world.remove_bodies_from_planning(goals)
        return goals

    return test_five_tables_domain(args, loader_fn, **kwargs)


def test_five_tables_small_sink(args, **kwargs):
    def loader_fn(world, **world_builder_args):
        arm = world.robot.arms[0]
        cabbage, egg, plate, salter, sink, stove, counter, table = load_five_table_scene(world)

        world.remove_object(plate)
        egg.adjust_pose(x=0, y=2)

        goals = [('Holding', arm, egg)]
        goals = [('On', cabbage, sink)]
        goals = [('On', cabbage, sink), ('On', salter, table)]

        # world.remove_bodies_from_planning(goals)
        return goals

    return test_five_tables_domain(args, loader_fn, **kwargs)


################################################################
