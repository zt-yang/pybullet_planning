from pybullet_tools.pr2_primitives import get_group_joints, Conf
from pybullet_tools.pr2_streams import sample_points_along_line, get_bconf_in_region_gen, get_parent_new_pose, get_bqs_given_p2

from world_builder.builders import *
from world_builder.loaders import *
from world_builder.loaders_nvidia_kitchen import load_kitchen_floor_plan
from world_builder.world_utils import load_asset, FLOOR_HEIGHT, visualize_point

from robot_builder.robot_builders import build_fridge_domain_robot
import random

from problem_sets.problem_utils import create_world, pddlstream_from_state_goal, save_to_kitchen_worlds


#######################################################


def test_bucket_lift(args, domain='pr2_eggs.pddl', w=.5, h=.9, mass=1):
    world = create_world(args)

    floor = world.add_object(
        Floor(create_box(w=10, l=10, h=FLOOR_HEIGHT, color=BLACK, collision=True)),
        Pose(point=Point(x=0, y=0, z=-2 * FLOOR_HEIGHT)))

    bucket = world.add_object(Object(load_asset('Bucket', floor=floor), category='bucket'))
    find_points_for_dual_grasp(bucket, world)

    robot = create_pr2_robot(world, base_q=(0, 2, -PI / 2), dual_arm=True)
    world.remove_object(floor)

    set_all_static()
    state = State(world)
    exogenous = []

    name_to_body = state.world.name_to_body
    goals = [('On', name_to_body('cabbage'), name_to_body('sink'))]

    pddlstream_problem = pddlstream_from_state_goal(state, goals, custom_limits=args.base_limits,
                                                    domain_pddl=args.domain_pddl, stream_pddl=args.stream_pddl)
    return state, exogenous, goals, pddlstream_problem


def find_points_for_dual_grasp(body, world):
    (x_min, y_min, z_min), (x_max, y_max, z_max) = get_aabb(body)
    x_c = (x_max + x_min) / 2
    y_c = (y_max + y_min) / 2
    pts = [(x_c, y_min, z_max), (x_c, y_max, z_max), (x_min, y_c, z_max), (x_max, y_c, z_max)]
    for (x,y,z) in pts:
        world.add_object(
            Movable(create_box(.05, .05, .05, mass=1, color=(0, 1, 0, 1)), category='marker'),
            Pose(point=Point(x, y, z)))


#######################################################


# def test_cart_pull(args, w=.5, h=.9, mass=1):
#     world = create_world(args)
#
#     kitchen = load_rooms(world)
#
#     cart, marker = load_cart(world)
#
#     robot = create_pr2_robot(world, base_q=(2, 0, -PI))  ## (-0.214, -1.923, 0)) ##
#
#     set_all_static()
#     state = State(world)
#     exogenous = []
#
#     # grasps = test_marker_pull_grasps(state, robot, cart, marker, visualize=False)
#     # goals = [("AtMarkerGrasp", 'left', marker, grasps[-1][0])]
#
#     # goals = [('AtBConf', Conf(robot, get_group_joints(robot, 'base'), (1.204, 0.653, -2.424)))]
#     # goals = [('HoldingMarker', 'left', marker)]
#
#     ## --- test `plan-base-pull-marker-random`
#     # goals = [("PulledMarker", marker)]
#
#     ## --- test `plan-base-pull-marker-to-bconf`
#     # goals = [('HoldingMarker', 'left', marker), ('RobInRoom', world.name_to_body('laundry_room'))]
#
#     ## --- test `grasp` + `pull` + `ungrasp`
#     goals = [('GraspedMarker', marker)]
#
#     ## --- test `plan-base-pull-marker-to-pose`
#     # goals = [('InRoom', marker, world.name_to_body('laundry_room'))]
#     ## --------------------------------------------
#
#     pddlstream_problem = pddlstream_from_state_goal(state, goals, custom_limits=args.base_limits,
#                                                     domain_pddl=args.domain_pddl, stream_pddl=args.stream_pddl)
#     return state, exogenous, goals, pddlstream_problem

#######################################################


def test_cart_obstacle_unsafebtraj(args, domain='pr2_eggs.pddl', stream='pr2_stream_carts.pddl'):
    world = create_world(args)

    cabbage, kitchen, laundry_room, storage, cart, marker = load_cart_regions(world)

    base_q = (2, 0, -PI)
    # base_q = shift_objects(base_q, world) ## debugging second move base
    robot = create_pr2_robot(world, base_q=base_q)  ## (-0.214, -1.923, 0)) ##

    set_all_static()
    state = State(world)
    exogenous = []
    # grasps = test_marker_pull_grasps(state, marker, visualize=True)

    # visualize_sampled_pose(world)

    ## move base with world config as argument
    goals = [('PulledMarker', marker)]
    goals = [('InRoom', marker, laundry_room)]
    goals = [('GraspedMarker', marker)]

    ## --- best using pull_to_pose
    # goals = [('GraspedMarker', marker), ('InRoom', marker, laundry_room)]
    # ## goals = [('SavedMarker', marker)]
    #
    # goals = [('InRoom', marker, laundry_room), ('RobInRoom', storage)]
    #
    # goals = [('InRoom', cart, laundry_room)]
    # goals = [('InRoom', marker, laundry_room), ('RobInRoom', kitchen)]
    #
    # goals = [('InRoom', marker, laundry_room), ('Holding', 'left', cabbage)]
    goals = [('Holding', 'left', cabbage)]
    ## --------------------------------------------

    pddlstream_problem = pddlstream_from_state_goal(state, goals, custom_limits=args.base_limits,
                                                    domain_pddl=args.domain_pddl, stream_pddl=args.stream_pddl)
    return state, exogenous, goals, pddlstream_problem


def visualize_sampled_pose(world):
    for i in range(20):
        pose2 = sample_points_along_line(world.name_to_body('cart'),
                  world.name_to_body('marker'), limit=(4,4.5), learned=True)
        visualize_point(pose2[0], world)
        print(pose2)


def shift_objects(base_q, world):
    marker = world.name_to_body('marker')
    cart = world.name_to_body('cart')
    p_marker = sample_points_along_line(cart, marker, limit=(4, 4.5), learned=True)
    p_cart = get_parent_new_pose(get_pose(marker), p_marker, get_pose(cart))
    base_q = get_bqs_given_p2(marker, cart, base_q, p_marker, 1)[0]
    world.BODY_TO_OBJECT[marker].set_pose(p_marker)
    world.BODY_TO_OBJECT[cart].set_pose(p_cart)
    world.remove_object(world.BODY_TO_OBJECT[marker])
    world.remove_object(world.BODY_TO_OBJECT[cart])
    return base_q

#######################################################


def test_cart_obstacle_wconf(args, domain='pr2_rearrange.pddl', stream='pr2_stream_carts.pddl'):
    world = create_world(args)

    cabbage, kitchen, laundry_room, storage, cart, marker = load_cart_regions(world)

    base_q = (2, 0, -PI)
    # base_q = shift_objects(base_q, world) ## debugging second move base
    robot = create_pr2_robot(world, base_q=base_q)  ## (-0.214, -1.923, 0)) ##

    set_all_static()
    state = State(world)
    exogenous = []

    # visualize_sampled_pose(world)

    goals = [('AtBConf', Conf(robot, get_group_joints(robot, 'base'), (2, 1, -PI)))]
    goals = [('WConfChanged',)]
    goals = [('InRoom', marker, laundry_room)]
    goals = [('GraspedMarker', marker)]
    ## goals = [('InRoom', marker, laundry_room), ('RobInRoom', world.name_to_body('doorway'))]
    goals = [('InRoom', marker, laundry_room), ('RobInRoom', storage)]
    ## goals = [('InRoom', marker, laundry_room), ('RobInRoom', kitchen)]
    goals = [('InRoom', marker, laundry_room), ('Holding', 'left', cabbage)]
    goals = [('Holding', 'left', cabbage)]
    ## --------------------------------------------

    pddlstream_problem = pddlstream_from_state_goal(state, goals, custom_limits=args.base_limits,
                                                    domain_pddl=args.domain_pddl, stream_pddl=args.stream_pddl)
    return state, exogenous, goals, pddlstream_problem

#######################################################


def test_cart_obstacle(args, domain='pr2_rearrange.pddl', stream='pr2_stream_carts.pddl'):
    world = create_world(args)

    cabbage, cart, marker = load_blocked_kitchen(world)
    robot = create_pr2_robot(world, base_q=(2, 0, -PI))

    set_all_static()
    state = State(world)
    exogenous = []

    goals = [('Holding', 'left', cabbage)]

    pddlstream_problem = pddlstream_from_state_goal(state, goals, custom_limits=args.base_limits,
                                                    domain_pddl=args.domain_pddl, stream_pddl=args.stream_pddl)
    return state, exogenous, goals, pddlstream_problem

#######################################################


def test_moving_carts(args):
    world = create_world(args)

    cabbage, cart, marker = load_blocked_kitchen(world)
    sink, cart2, marker2 = load_blocked_sink(world)
    robot = create_pr2_robot(world, base_q=(2, -1.5, -PI))

    laundry_room = world.add_object(
        Location(create_box(w=2, l=6, h=FLOOR_HEIGHT, color=YELLOW, collision=True),
                 name='laundry_room'),
        Pose(point=Point(x=4, y=-1.5, z=-2 * FLOOR_HEIGHT)))

    ## --- test basic grasping and pulling
    goals = [('Holding', 'left', cabbage)]
    # goals = [('InRoom', marker, laundry_room)]
    # goals = [('InRoom', marker2, laundry_room)]
    # goals = [('InRoom', marker, laundry_room), ('InRoom', marker2, laundry_room)]
    #
    # ## -- test basic picking and placing
    # # world.remove_object(cart)
    # # world.remove_object(marker)
    # # world.remove_object(cart2)
    # # world.remove_object(marker2)
    # goals = [('On', cabbage, sink)]
    # goals = [('InRoom', marker, laundry_room), ('InRoom', marker2, laundry_room), ('On', cabbage, sink)]
    #
    # goals = [('On', cabbage, sink)]

    set_all_static()
    state = State(world)
    exogenous = []

    pddlstream_problem = pddlstream_from_state_goal(state, goals, custom_limits=args.base_limits,
                                                    domain_pddl=args.domain_pddl, stream_pddl=args.stream_pddl)
    return state, exogenous, goals, pddlstream_problem


def test_three_moving_carts(args, domain='pr2_rearrange.pddl', stream='pr2_stream.pddl'):
    world = create_world(args)

    cabbage, cart, marker = load_blocked_kitchen(world)
    sink, cart2, marker2 = load_blocked_sink(world)
    stove, cart3, marker3 = load_blocked_stove(world)
    robot = create_pr2_robot(world, base_q=(2, 0, -PI))

    ## --- test basic grasping and pulling
    # world.remove_object(cart2)
    # world.remove_object(marker2)
    # world.remove_object(cart3)
    # world.remove_object(marker3)
    goals = [('Cooked', cabbage)]

    set_all_static()
    state = State(world)
    exogenous = []

    pddlstream_problem = pddlstream_from_state_goal(state, goals, custom_limits=args.base_limits,
                                                    domain_pddl=args.domain_pddl, stream_pddl=args.stream_pddl)
    return state, exogenous, goals, pddlstream_problem


#######################################################


def test_kitchen_joints(args):
    world = create_world(args)

    floor = load_kitchen_floor_plan(world, plan_name='kitchen.svg') ## studio0, studio0
    world.remove_object(floor)
    robot = create_pr2_robot(world, base_q=(1.79, 6, PI / 2 + PI / 2))

    name_to_body = world.name_to_body

    set_camera_pose((3, 7.5, 2), (1, 7.5, 1))
    import time
    import random
    goals = []
    joints = world.cat_to_bodies('drawer') + world.cat_to_bodies('door')
    for k in range(50):
        j = random.choice(joints)
        world.toggle_joint_by_name(world.BODY_TO_OBJECT[j].shorter_name)
        time.sleep(0.2)
        # wait_if_gui()

    set_all_static()
    state = State(world)
    exogenous = []

    pddlstream_problem = pddlstream_from_state_goal(state, goals, custom_limits=args.base_limits,
                                                    domain_pddl=args.domain_pddl, stream_pddl=args.stream_pddl)
    return state, exogenous, goals, pddlstream_problem


#######################################################


def test_fridge_pose(args, w=.5, h=.9, wb=.07, hb=.1):
    world = create_world(args)

    floor = load_floor_plan(world, plan_name='fridge.svg')
    world.remove_object(floor)
    robot = create_pr2_robot(world, base_q=(1.79, 6, PI / 2 + PI / 2))

    name_to_body = world.name_to_body
    name_to_object = world.name_to_object

    # ## ---- for testing grasping from fridge
    # egg = load_experiment_objects(world, CABBAGE_ONLY=True, name='eggblock', color=TAN)
    # world.put_on_surface(egg, 'shelf_bottom')
    #
    # ## ---- basic pulling door open
    # door = 'fridge_door'
    # door = name_to_body(door)
    # goals = ('test_handle_grasps', door)
    # goals = [("HandleGrasped", 'left', door)]
    # goals = [("OpenedJoint", door)]
    # goals = [("GraspedHandle", door)]
    #
    # ## -- just pick get the block
    # world.open_joint_by_name('fridge_door')
    # set_camera_pose(camera_point=[3, 6.5, 2], target_point=[1, 4, 1])
    # goals = [("Holding", "left", egg)]

    # ## ---- with domain='pr2_food.pddl'
    # world.close_joint_by_name('fridge_door')
    # goals = [("HandleGrasped", 'left', door)]
    # goals = [("GraspedHandle", door)]
    # goals = [("GraspedHandle", door), ("Holding", "left", egg)]
    # goals = [("Holding", "left", egg)]

    ## -- test crowded fridge
    # world.remove_object(egg)
    # table = world.add_object(
    #     Object(create_box(w, w, h, color=(.75, .75, .75, 1)), category='supporter', name='table'),
    #     Pose(point=Point(x=2.5, y=4.3, z=h / 2)))
    set_camera_target_body(name_to_body('fridge'), dx=0, dy=0.5, dz=0.3)
    # set_camera_target_body(name_to_body('fridge'), dx=3, dy=2, dz=1)
    # set_camera_target_body(name_to_body('fridge'), dx=1.5, dy=1.5, dz=1)
    world.open_joint_by_name('fridge_door')
    name_to_object('meatchicken').set_pose(((1.728, 4.227, 0.914), (0, 0, 0, 1)))
    name_to_object('veggiepotato').set_pose(((1.428, 4.447, 0.914), (1, 1, 0, 1)))
    name_to_object('veggiecabbage').set_pose(((1.628, 4.447, 0.914), (0, 0, 0, 1)))
    name_to_object('veggiecauliflower').set_pose(((1.828, 4.447, 0.914), (0, 0, 0, 1)))
    name_to_object('veggieartichoke').set_pose(((1.538, 4.247, 0.914), (1, 1, 0, 0)))
    name_to_object('veggiezucchini').set_pose(((1.728, 4.347, 0.914), (0, 0, 0, 1)))
    name_to_object('veggietomato').set_pose(((1.438, 4.287, 0.914), (0, 0, 0, 1)))
    name_to_object('veggiegreenpepper').set_pose(((1.6, 4.6, 0.948), (0, 0, 0, 1)))
    name_to_object('veggiegreenpepper').set_pose(((1.69, 4.55, 0.948), (0, 0, 0, 1)))
    # objects = world.name_to_object('shelf_bottom').supported_objects
    # for o in objects:
    #     # draw_aabb(get_aabb(o.body))
    #     for oo in objects:
    #         if oo == o: continue
    #         if pairwise_collision(oo.body, o.body):
    #             print(f'collision between {o.name} and {oo.name}')
    goals = [("Holding", "left", name_to_body('veggiegreenpepper'))]
    goals = [("Holding", "left", name_to_body('veggiecabbage'))]
    goals = [("Holding", "left", name_to_body('veggiezucchini'))]
    # goals = [("On", name_to_body('veggiegreenpepper'), table)]
    # goals = [("Holding", "left", name_to_body('veggiecabbage'))]

    set_all_static()
    state = State(world)
    exogenous = []

    pddlstream_problem = pddlstream_from_state_goal(state, goals, custom_limits=args.base_limits,
                                                    domain_pddl=args.domain_pddl, stream_pddl=args.stream_pddl)

    return state, exogenous, goals, pddlstream_problem

#######################################################


def test_pr2_counter_minifridge(args, SAMPLED=True, robot_builder_args=None, **kwargs):
    world = create_world(args)
    # custom_limits = ((0, 0, 0.1), (6, 6, 1.5))
    custom_limits = {0: (0, 6), 1: (0, 6), 2: (0.1, 1.5)}

    robot_name = 'feg' if 'feg' in args.domain_pddl else 'pr2'
    exp_name = "one_fridge_{robot}".format(robot=robot_name)
    robot = build_fridge_domain_robot(world, robot_name, custom_limits=custom_limits)

    if SAMPLED:
        movable_category = 'food'
        case = random.choice([0, 1, 2, 2])
        # case = random.choice([0, 1])
        # case = 0
        case = 0
        if case == 0:
            from world_builder.builders import test_one_fridge as test_scene
        elif case == 1:
            from world_builder.builders import test_fridge_table as test_scene
        elif case == 2:
            from world_builder.builders import test_fridges_tables as test_scene
        elif case == 22:
            from world_builder.builders import test_fridges_tables as test_scene
            movable_category = 'stapler'
        elif case == 28:
            from world_builder.builders import test_fridges_tables_conjunctive as test_scene
        elif case == 3:
            from world_builder.builders import test_three_fridges_tables as test_scene
        # from world_builder.builders import test_three_fridges_tables as test_scene

        goals = test_scene(world, movable_category, SAMPLING=False)
        # goals = ('test_handle_grasps', world.cat_to_bodies('door')[0])
        # goals = [('AtBConf', Conf(robot, robot.get_base_joints(), (5, 5, 0.5, 0)))]
        # world.open_all_doors_drawers(extent=1)

        # """ ============== [Output] Save depth image ==================== """
        # ## you may purturb the camera pose ((point), (quaternian))
        # camera_pose = ((3.7, 8, 1.3), (0.5, 0.5, -0.5, -0.5))
        # world.add_camera(camera_pose)

        robot = world.robot
        custom_limits = robot.custom_limits

        set_renderer(True)
        set_renderer(True)
        fridge = world.name_to_body('minifridge::storage')
        set_camera_target_body(fridge[0], dx=2, dy=0, dz=2)

        set_renderer(False)
        exp_name += f"_{get_datetime(seconds=True)}"

        door = world.name_to_object('minifridge').doors[0]
        # goals += [("OpenedJoint", door)]

    else:

        world.set_skip_joints()
        minifridge_doors = load_random_mini_kitchen_counter(world, table_only=False)
        # world.remove_object(world.name_to_object('counter'))

        door = minifridge_doors[0]
        # door = random.choice(minifridge_doors)

        cabbage = world.name_to_body('cabbage')
        counter = world.name_to_body('counter')
        fridge = world.name_to_body('minifridge::storage')
        set_camera_target_body(fridge[0], dx=2, dy=0, dz=2)

        arm = robot.arms[0]
        goals = [('on', cabbage, counter)]
        goals = ('test_handle_grasps', door)
        goals = [("HandleGrasped", arm, door)]
        goals = [("OpenedJoint", door)]
        goals = [('holding', arm, cabbage)]
        # goals = [('in', cabbage, fridge)]

    set_renderer(True)
    set_renderer(False)
    set_all_static()
    state = State(world, grasp_types=robot.grasp_types)

    pddlstream_problem = pddlstream_from_state_goal(state, goals, args, custom_limits, **kwargs)
    save_to_kitchen_worlds(state, pddlstream_problem, exp_name=exp_name, world_name=exp_name,
                           exit=False, DEPTH_IMAGES=False)
    return state, [], goals, pddlstream_problem


def test_pick_ir_ik(args, w=.15, TEST=True, **kwargs):
    world = create_world(args)
    exp_name = 'reachability_pick'

    robot = create_pr2_robot(world, custom_limits=args.base_limits, base_q=(2, 5, 0), use_torso=True)
    set_camera_pose((5, 3, 1), (0, 3, 1))

    x = 2
    y = 1
    if TEST:
        h = random.uniform(0.1, 1.5)
        table = world.add_box(
            Object(create_box(w, w, h, color=(.75, .75, .75, 1)), category='supporter', name=f'table_{h}'),
            Pose(point=Point(x=x, y=y, z=h / 2)))
        cabbage = world.add_object(
            Movable(load_asset('VeggieCabbage', x=x, y=y, yaw=0, floor=table), name=f'cabbage_{h}'))
        goals = [("Holding", 'left', cabbage)]
        exp_name += f"_{get_datetime(seconds=True)}"

    else:
        cabbages = {}
        for h in [.2, .5, .9, 1.2]:
            table = world.add_box(
                Object(create_box(w, w, h, color=(.75, .75, .75, 1)), category='supporter', name=f'table_{h}'),
                Pose(point=Point(x=x, y=y, z=h / 2)))
            cabbages[h] = world.add_object(
                Movable(load_asset('VeggieCabbage', x=x, y=y, yaw=0, floor=table), name=f'cabbage_{h}'))
            y += 1
        goals = [("Holding", 'left', cabbages[0.5])]

    set_all_static()
    state = State(world)
    exogenous = []

    pddlstream_problem = pddlstream_from_state_goal(state, goals, domain_pddl=args.domain_pddl,
                                                    custom_limits=args.base_limits, stream_pddl=args.stream_pddl)
    save_to_kitchen_worlds(state, pddlstream_problem, exp_name=exp_name, world_name=exp_name)
    return state, exogenous, goals, pddlstream_problem

