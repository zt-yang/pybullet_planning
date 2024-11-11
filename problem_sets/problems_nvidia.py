import random

from pybullet_tools.utils import set_camera_pose
from pybullet_tools.pose_utils import sample_obj_in_body_link_space
from pybullet_tools.bullet_utils import close_joint

from world_builder.loaders_partnet_kitchen import put_lid_on_braiser
from world_builder.loaders_nvidia_kitchen import *
from world_builder.world_utils import select_door_closer_to_body

from robot_builder.robot_builders import build_robot_from_args

from problem_sets.problem_utils import create_world, pddlstream_from_state_goal, save_to_kitchen_worlds, \
    problem_template
from world_builder.actions import pull_actions, pick_place_actions, pull_with_link_actions, \
    pick_sprinkle_actions, pick_place_rel_actions, pick_arrange_actions


#######################################################


def test_kitchen_fridge(args, **kwargs):
    def loader_fn(world, difficulty=0, use_vlmtamp_kitchen=False, **world_builder_args):
        if use_vlmtamp_kitchen:
            surfaces = {
                'Fridge': { 'shelf_top': [], 'shelf_bottom': []},
                'counter': { 'front_right_stove': ['BraiserBody'] }
            }
            load_full_kitchen(world, surfaces=surfaces, spaces={}, load_cabbage=False)
            load_nvidia_kitchen_movables(world, open_doors_for=['chicken-leg'])
            body = world.name_to_body('chicken-leg')
        else:
            floor = load_kitchen_floor_plan(world, plan_name='kitchen_v2.svg')
            world.remove_object(floor)
            # floor = load_floor_plan(world, plan_name='fridge_v2.svg')
            body = load_experiment_objects(world, CABBAGE_ONLY=True, name='eggblock', color=TAN)

        set_camera_pose(camera_point=[3, 5, 3], target_point=[0, 6, 1])
        robot = world.robot

        door = world.add_joints_by_keyword('fridge', 'fridge_door')[0]

        skeleton = []
        ## -- open the door, requires kitchen_v2 floorplan
        if difficulty == 0:
            goals = ('test_handle_grasps', door)
            goals = [("HandleGrasped", 'left', door)]
            # goals = [("OpenedJoint", door)]
            goals = [("GraspedHandle", door)]
            #
            # ## -- just pick get the block
            # world.open_joint_by_name('fridge_door')
            # world.put_on_surface(egg, 'shelf_bottom')
            # goals = [("Holding", "left", body)]

        ## --- using three actions or one action
        if difficulty == 2:
            goals = [("Pulled", door)]  ## mobile_v4_domain.pddl
            goals = [("Pulled", door), ("CanUngrasp",)]  ## mobile_v4_domain.pddl
            goals = [("GraspedHandle", door)]  ## mobile_v4_domain.pddl
            # goals = [("PulledOneAction", door)]  ## mobile_v5_domain.pddl

        if difficulty in [1, 11]:
            if difficulty == 1:
                world.close_joint_by_name('fridge_door')
            goals = [('AtBConf', Conf(robot, get_group_joints(robot, 'base'), (2, 1, -PI)))]
            goals = ("test_object_grasps", body)
            goals = [("Holding", "left", body)]

        if difficulty in [3, 31, 32, 4]:
            """ rummy needs to change grasps from side to top """
            shelf_bottom = world.name_to_body('shelf_bottom')
            # world.BODY_TO_OBJECT[body].change_pose_interactive() find pose for saved_poses

            table = world.add_object(
                Supporter(load_asset('DinerTable'), name='table'),
                pose=((0.5, 4.25, 0.59), (0.0, 0.0, 0.0, 1.0)),
            )  ## .change_pose_interactive()
            close_joint(table, 1)  ## instance originally has opened drawer

            pot = world.name_to_object('braiserbody')
            world.add_object(Surface(pot, link_from_name(pot, 'braiser_bottom')))
            world.put_on_surface(pot, 'table')
            goal_location = world.name_to_body('braiser_bottom')

            if difficulty == 3:
                goals = [("Holding", "left", body)]
                skeleton += [('pick', "left", body), ('arrange', "left", body, shelf_bottom), ('pick', "left", body)]

            elif difficulty == 31:
                goals = [("On", body, shelf_bottom)]
                skeleton += [('pick', "left", body), ('arrange', "left", body, shelf_bottom)]

            elif difficulty == 32:
                ## problem with pddlstream stream plan
                if world.robot.name == 'rummyrobot':
                    world.name_to_object('chicken-leg').set_pose((0.654, 4.846, 0.794, 0.0, 0.0, -0.166))
                    world.robot.set_base_conf((1.732, 4.323, 0.539, 2.034))
                goals = [("On", body, goal_location)]
                skeleton += [('pick', "left", body), ('arrange', "left", body, goal_location)]

            elif difficulty == 4:
                ## problem with pddlstream stream plan
                goals = [("On", body, goal_location)]
                if world.robot.name == 'rummyrobot':
                    skeleton += [('pick', "left", body), ('arrange', "left", body, shelf_bottom)]
                skeleton += [('pick', "left", body), ('place', "left", body)]
                # skeleton += [('pick', "left", body), ('arrange', "left", body, goal_location)]

        world.remove_bodies_from_planning(goals, skeleton=skeleton)
        return {'goals': goals, 'skeleton': skeleton}

    return test_nvidia_kitchen_domain(args, loader_fn, initial_xy=(2, 5), **kwargs)


def test_kitchen_oven(args, **kwargs):
    world = create_world(args)
    cabbage = load_full_kitchen(world)

    name_to_body = world.name_to_body

    ## -- test the position of stove
    # world.put_on_surface(egg, 'front_right_stove')
    goals = [("Holding", "left", cabbage)]

    ## -- prepare the pot
    oven = world.name_to_object('oven')
    pot = world.name_to_object('braiserbody')
    lid = world.name_to_body('braiserlid')
    world.put_on_surface(pot, 'front_right_stove')
    set_camera_target_body(oven, dx=1, dy=0, dz=1)
    bottom = world.add_object(Surface(pot, link_from_name(pot, 'braiser_bottom')))
    world.put_on_surface(cabbage, 'indigo_tmp')
    # world.put_on_surface(cabbage, 'braiser_bottom')

    # world.remove_object(oven)
    # world.remove_object(pot)

    goals = ("test_object_grasps", cabbage)
    goals = [("Holding", "left", cabbage)]
    goals = [("Holding", "left", lid)]
    # goals = [("On", cabbage, bottom)]
    # goals = [("On", cabbage, world.name_to_body('hitman_countertop'))]

    set_all_static()
    state = State(world)
    exogenous = []

    pddlstream_problem = pddlstream_from_state_goal(state, goals, args, **kwargs)
    return state, exogenous, goals, pddlstream_problem


#######################################################

def test_oven_egg(args, domain='pr2_food.pddl', stream='pr2_stream.pddl'):
    world = create_world(args)
    world.set_skip_joints()

    floor = load_floor_plan(world, plan_name='counter.svg')
    egg = load_experiment_objects(world, CABBAGE_ONLY=True, name='eggblock', color=TAN)
    world.remove_object(floor)
    robot = create_pr2_robot(world, base_q=(1.79, 6, PI / 2 + PI / 2))

    name_to_body = world.name_to_body
    surface = name_to_body('indigo_tmp')
    world.put_on_surface(egg, 'indigo_tmp')

    ## -- add pot
    pot = world.name_to_body('braiserbody')
    world.put_on_surface(pot, 'front_right_stove')
    # oven = world.name_to_object('oven')
    # set_camera_target_body(oven, dx=1, dy=0, dz=1)
    bottom = world.add_object(Surface(pot, link_from_name(pot, 'braiser_bottom')))
    bottom = world.name_to_body('braiser_bottom')

    ## -- test that pick-place is working
    cabbage = load_experiment_objects(world, CABBAGE_ONLY=False, EXIST_PLATE=False)
    sink = name_to_body('sink')
    goals = [("On", cabbage, sink)]  ## success
    goals = [("On", egg, sink)]  ## success
    goals = [("On", cabbage, bottom)]  ## fail
    goals = ('test_pose_gen', (cabbage, sink))  ## success
    goals = ('test_pose_gen', (cabbage, bottom))  ## success
    goals = ('test_pose_gen', (egg, bottom))  ## success

    ## -- start on surface to braiser
    goals = [("On", egg, bottom)]  ## success

    set_all_static()
    state = State(world)
    exogenous = []

    pddlstream_problem = pddlstream_from_state_goal(state, goals, custom_limits=args.base_limits,
                                                    domain_pddl=args.domain_pddl, stream_pddl=args.stream_pddl)
    return state, exogenous, goals, pddlstream_problem

#######################################################


def test_braiser_lid(args, domain='pr2_food.pddl', stream='pr2_stream.pddl'):
    world = create_world(args)
    world.set_skip_joints()

    floor = load_floor_plan(world, plan_name='counter.svg')
    egg = load_experiment_objects(world, CABBAGE_ONLY=True, name='eggblock',
                                  color=TAN, wb=.03, hb =.04) ##
    world.remove_object(floor)
    robot = create_pr2_robot(world, base_q=(1.79, 6, PI / 2 + PI / 2))

    name_to_body = world.name_to_body
    name_to_object = world.name_to_object

    world.put_on_surface(egg, 'indigo_tmp')
    print('egg original', name_to_object('egg').get_pose())
    oven = world.name_to_object('oven')
    # set_camera_target_body(oven, dx=1, dy=0, dz=1)
    set_camera_target_body(oven, dx=0.7, dy=0, dz=1.5)

    ## -- add pot
    pot = name_to_body('braiserbody')
    world.put_on_surface(pot, 'front_right_stove')

    ## ------ put in egg without lid
    world.add_object(Surface(pot, link_from_name(pot, 'braiser_bottom')))
    bottom = name_to_body('braiser_bottom')
    goals = ('test_pose_gen', (egg, bottom))  ## succeed
    goals = [("On", egg, bottom)]  ## succeed

    ## ------ remove lid
    lid = name_to_body('braiserlid')
    world.put_on_surface(lid, 'braiserbody')
    surface = name_to_body('indigo_tmp')
    goals = ("test_object_grasps", lid)
    goals = ('test_pose_gen', (lid, surface))  ## success
    goals = [('AtBConf', Conf(robot, get_group_joints(robot, 'base'), (1.239, 8.136, 3.113)))]
    goals = [("On", lid, surface)]
    # world.remove_object(name_to_object('braiserlid'))
    world.add_to_cat(lid, 'movable')

    ## ------ remove lid in order to put in egg
    goals = ('test_pose_gen', (egg, bottom))  ## success
    goals = [("On", lid, surface)]  ## success
    # goals = [("On", egg, bottom)]  ## fail

    ## ======== debug option 1 =========
    # uncomment test_grasp_ik(state, state.get_facts()+preimage, name='eggblock') in stream_agent.py
    ## ------- test IK successful
    # world.put_on_surface(egg, 'braiser_bottom')
    # world.remove_object(name_to_object('braiserlid'))
    # goals = [("On", egg, world.name_to_body('front_left_stove'))]

    ## ------- test IK successful - so wconf is correct
    # world.put_on_surface(egg, 'braiser_bottom')
    # goals = [("On", lid, surface)]

    ## ======== debug option 2 =========
    goals = [("Debug1",)]  ## succeed
    goals = [("Debug2",)]  ## succeed
    world.put_on_surface(egg, 'braiser_bottom')
    goals = [("Debug2",)]  ## succeed, it can pick the egg after removing lid!
    # goals = [("On", egg, world.name_to_body('front_left_stove')), ("Debug3",)]  ## fail
    # goals = [("Debug3",)]  ## fail
    # goals = [("On", lid, surface), ("On", egg, bottom), ("Debug3",)]  ## fail

    ## just give it more time!
    goals = [("Holding", 'left', egg)]  ## successful
    goals = [("On", egg, world.name_to_body('front_left_stove'))]  ## successful

    world.remove_body_from_planning(name_to_body('hitman_countertop'))
    set_all_static()
    state = State(world)
    exogenous = []

    pddlstream_problem = pddlstream_from_state_goal(state, goals, custom_limits=args.base_limits,
                                                    domain_pddl=args.domain_pddl, stream_pddl=args.stream_pddl)
    return state, exogenous, goals, pddlstream_problem

#######################################################


def make_base_limits(x_min, x_max, y_min, y_max):
    zero_limits = 0 * np.ones(2)
    half_limits = 12 * np.ones(2)
    BASE_LIMITS = (-half_limits, +half_limits)  ## (zero_limits, +half_limits) ##
    return (np.asarray([x_min, y_min]), np.asarray([x_max, y_max]))


def test_egg_movements(args):
    world = create_world(args)
    world.set_skip_joints()

    floor = load_kitchen_floor_plan(world, plan_name='kitchen_v3.svg')
    egg = load_experiment_objects(world, CABBAGE_ONLY=True, name='eggblock',
                                  color=TAN, wb=.03, hb =.04) ##
    bottom, lid = load_pot_lid(world)

    world.remove_object(floor)
    robot = create_pr2_robot(world, base_q=(2.5, 6, PI / 2 + PI / 2))

    name_to_body = world.name_to_body
    name_to_object = world.name_to_object

    door = world.add_joints_by_keyword('fridge', 'fridge_door')[0]

    ## ---- test 1
    world.put_on_surface(egg, 'braiser_bottom')
    goals = [("On", egg, name_to_body('front_left_stove'))]  ## successful

    ## ---- test 2
    world.put_on_surface(egg, 'indigo_tmp')
    goals = [("On", egg, name_to_body('braiser_bottom'))]  ## successful

    ## ---- test 3
    world.put_on_surface(egg, 'shelf_bottom')
    # world.open_joint_by_name('fridge_door')n
    goals = [("GraspedHandle", door)]
    goals = [("Holding", "left", egg)]  ## successful
    # goals = [("On", name_to_body('braiserlid'), name_to_body('front_left_stove'))]  ## successful
    # goals = [("On", egg, name_to_body('front_left_stove'))]  ## successful
    # goals = [("On", egg, name_to_body('braiser_bottom'))]  ## successful

    ## ---- test 4 debug HPN  ## successful
    # mp = ((1.448, 4.306, 1.256), (1.415, 7.708, 1.37), 1.78)
    # mp = ((1.203, 8.093, 2.752), (0.017, 4.264, 0.653), 0)
    # world.put_on_surface(name_to_object('braiserlid'), 'front_left_stove')
    # world.open_joint_by_name('fridge_door', pstn=mp[2])
    # robot.set_base_positions(mp[0])
    # goals = [('AtBConf', Conf(robot, get_group_joints(robot, 'base'), mp[1]))]

    ## ---- test 5 more joints and surfaces
    world.remove_body_from_planning(name_to_body('hitman_countertop')) ## also added two back stoves
    # world.add_surface_by_keyword('counter', 'back_left_stove')
    # world.add_surface_by_keyword('counter', 'back_right_stove')
    # world.add_joints_by_keyword('counter', 'chewie_door_left_joint')
    # world.add_joints_by_keyword('counter', 'chewie_door_right_joint')

    set_all_static()
    state = State(world)
    exogenous = []

    pddlstream_problem = pddlstream_from_state_goal(state, goals, custom_limits=args.base_limits,
                                                    domain_pddl=args.domain_pddl, stream_pddl=args.stream_pddl)

    # set_camera_target_body(name_to_body('dishwasher'), dx=1, dy=0, dz=1)
    # door = world.add_joints_by_keyword('dishwasher', 'dishwasher_door')[0]
    # world.open_joint_by_name('dishwasher_door')
    # set_camera_target_body(name_to_body('dishwasher'), dx=1, dy=0, dz=1)
    # world.put_on_surface(name_to_object('microwave'), 'hitman_countertop')

    # set_camera_target_body(name_to_body('microwave'), dx=3, dy=2, dz=1)

    # to_lisdf(world, 'kitchen_v2.svg', pddlstream_problem.init, world_name='kitchen_basics', root_path=KITCHEN_WORLD)
    return state, exogenous, goals, pddlstream_problem

#######################################################


def test_opened_space(args):
    world = create_world(args)
    world.set_skip_joints()

    floor = load_kitchen_floor_plan(world, plan_name='kitchen_v2.svg')
    egg = load_experiment_objects(world, CABBAGE_ONLY=True, name='eggblock',
                                  color=TAN, wb=.03, hb =.04) ##
    bottom, lid = load_pot_lid(world)

    world.remove_object(floor)
    robot = create_pr2_robot(world, base_q=(2.5, 6, PI / 2 + PI / 2))

    name_to_body = world.name_to_body
    name_to_object = world.name_to_object

    door = world.add_joints_by_keyword('fridge', 'fridge_door')[0]
    fridge = name_to_body('fridge')
    world.put_on_surface(egg, 'shelf_bottom')
    world.remove_body_from_planning(name_to_body('hitman_countertop'))
    world.add_to_cat(fridge, 'space')

    ## need to uncomment `('IsJointTo', body, body[0])` in world.py get_facts()
    goals = [('OpenedJoint', door)]
    # goals = [('OpenedSpace', fridge)]  ## fail
    # goals = [('Unreachable', egg)]  ## return fail, uninformative, use Debug1
    # goals = [('Debug1', )]  ## successful with (In ?o ?r) (Unreachable ?o)
    # goals = [('Holding', 'left', egg)]  ## just return fail
    # goals = [("On", egg, name_to_body('front_left_stove'))]  ## successful
    # goals = [("On", egg, name_to_body('braiser_bottom'))]  ## successful

    set_all_static()
    state = State(world)
    exogenous = []

    pddlstream_problem = pddlstream_from_state_goal(state, goals, custom_limits=args.base_limits,
                                                    domain_pddl=args.domain_pddl, stream_pddl=args.stream_pddl)

    # set_camera_target_body(name_to_body('microwave'), dx=3, dy=2, dz=1)
    return state, exogenous, goals, pddlstream_problem

#######################################################


def test_kitchen_demo(args):
    world = create_world(args)
    world.set_skip_joints()

    name_to_body = world.name_to_body
    name_to_object = world.name_to_object

    surfaces = {
        'counter': {
            'front_left_stove': [],
            'front_right_stove': ['BraiserBody'],
            'hitman_tmp': ['Microwave'],
            'indigo_tmp': ['BraiserLid'], ## , 'MeatTurkeyLeg'
        },
        'Fridge': {
            'shelf_top': ['MilkBottle'],
            'shelf_bottom': [], ## 'VeggieCabbage'
        },
        'Basin': {
            'faucet_platform': ['Faucet']
        }
    }

    floor = load_kitchen_floor_plan(world, plan_name='kitchen_v3.svg', surfaces=surfaces)

    egg = load_experiment_objects(world, CABBAGE_ONLY=True, name='eggblock',
                                  color=TAN, wb=.03, hb =.04) ##
    world.put_on_surface(egg, 'shelf_bottom')
    world.add_to_cat(egg, 'Edible')
    # egg = name_to_body('cabbage')

    world.remove_object(floor)
    robot = create_pr2_robot(world, base_q=(2.5, 6, PI / 2 + PI / 2))

    load_kitchen_mechanism(world)
    world.remove_object(name_to_body('braiserlid'))

    world.open_joint_by_name('fridge_door', pstn=1.5)
    set_camera_pose(camera_point=[3, 4.5, 3], target_point=[0, 5, 0.5])  ## made video for clean cabbage
    set_camera_pose(camera_point=[4, 5, 4], target_point=[0, 5, 0.5])  ## made video for cook cabbage
    # world.remove_object(name_to_object('fridge'))

    world.open_joint_by_name('fridge_door', pstn=1.8)
    goals = ('test_handle_grasps', 'fridge_door')
    goals = [("On", egg, name_to_body('braiser_bottom'))]  ## successful
    # goals = [("GraspedHandle", name_to_body('joint_faucet_0'))]  ## successful
    # goals = [("On", egg, name_to_body('basin_bottom'))]  ## successful
    # goals = [("Cleaned", egg)]  ## made video
    # goals = [("Cooked", egg)]  ## successful

    set_all_static()
    state = State(world)
    exogenous = []

    pddlstream_problem = pddlstream_from_state_goal(state, goals, custom_limits=args.base_limits,
                                                    domain_pddl=args.domain_pddl, stream_pddl=args.stream_pddl)

    # set_camera_target_body(name_to_body('microwave'), dx=3, dy=2, dz=1)
    return state, exogenous, goals, pddlstream_problem

#######################################################


def test_kitchen_demo_two(args):
    world = create_world(args)
    world.set_skip_joints()

    name_to_body = world.name_to_body
    name_to_object = world.name_to_object

    surfaces = {
        'counter': {
            'front_left_stove': [],
            'front_right_stove': ['BraiserBody'],
            'hitman_tmp': ['Microwave'],
            'indigo_tmp': ['BraiserLid', 'MeatTurkeyLeg'], ##
        },
        'Fridge': {
            'shelf_top': ['MilkBottle'],
            'shelf_bottom': ['VeggieCabbage'],
        },
        'Basin': {
            'faucet_platform': ['Faucet']
        }
    }

    floor = load_kitchen_floor_plan(world, plan_name='kitchen_v3.svg', surfaces=surfaces)
    world.remove_object(floor)
    robot = create_pr2_robot(world, base_q=(2.5, 6, PI / 2 + PI / 2))
    load_kitchen_mechanism(world)

    cabbage = name_to_body('cabbage')
    chicken = name_to_body('turkey')
    for ingredient in [cabbage, chicken]:
        world.add_to_cat(ingredient, 'Edible')
    world.put_on_surface(cabbage, 'shelf_bottom')
    # world.put_on_surface(chicken, 'indigo_tmp')
    # world.add_to_cat(chicken, 'cleaned')

    set_camera_pose(camera_point=[4, 5, 4], target_point=[0, 5, 0.5])  ## made video for cook cabbage

    # goals = [("Holding", 'left', cabbage)]
    world.open_joint_by_name('fridge_door', pstn=1.57)

    world.open_joint_by_name('fridge_door', pstn=1.2)
    # goals = [("Cleaned", cabbage)]  ## successful
    # goals = [("Cooked", cabbage)]  ## successful
    # goals = [("Cooked", cabbage), ("Cooked", chicken)]  ## successful

    # # ---- for record put food together
    # set_camera_target_body(name_to_body('braiser'), dx=1, dy=0, dz=1)
    # world.put_on_surface(cabbage, 'indigo_tmp')
    goals = [("On", cabbage, name_to_body('braiser_bottom')),
             ("On", chicken, name_to_body('braiser_bottom'))]  ## successful
    goals = [("On", chicken, name_to_body('braiser_bottom'))]  ## successful
    #
    # # ---- for screenshot final state
    # set_camera_target_body(name_to_body('braiser'), dx=0, dy=0, dz=0.5)
    # world.put_on_surface(cabbage, 'braiser_bottom')
    # world.put_on_surface(chicken, 'braiser_bottom')
    # world.remove_object(name_to_object('braiserlid'))
    # goals = [("On", cabbage, name_to_body('braiser_bottom')),
    #          ("On", chicken, name_to_body('braiser_bottom'))]  ## successful

    set_all_static()
    state = State(world)
    exogenous = []

    pddlstream_problem = pddlstream_from_state_goal(state, goals, custom_limits=args.base_limits,
                                                    domain_pddl=args.domain_pddl, stream_pddl=args.stream_pddl)

    # set_camera_target_body(name_to_body('microwave'), dx=3, dy=2, dz=1)
    set_camera_target_body(name_to_body('oven'), dx=0.5, dy=0.5, dz=0.8)
    save_to_kitchen_worlds(state, pddlstream_problem, exp_name='kitchen',
                           floorplan='kitchen_v3.svg', world_name='kitchen_lunch')
    return state, exogenous, goals, pddlstream_problem


#######################################################

def test_kitchen_demo_objects(args):
    world = create_world(args)
    world.set_skip_joints()

    name_to_body = world.name_to_body
    name_to_object = world.name_to_object

    floor = load_kitchen_floor_plan(world, plan_name='kitchen_v3.svg')
    world.remove_object(floor)
    robot = create_pr2_robot(world, base_q=(2.5, 6, PI / 2 + PI / 2))
    load_kitchen_mechanism(world)

    cabbage = name_to_body('cabbage')
    chicken = name_to_body('turkey')
    for ingredient in [cabbage, chicken]:
        world.add_to_cat(ingredient, 'Edible')
    world.put_on_surface(cabbage, 'shelf_bottom')
    world.put_on_surface(chicken, 'indigo_tmp')

    # world.open_joint_by_name('fridge_door', pstn=1.2)
    set_camera_pose(camera_point=[4, 5, 4], target_point=[0, 5, 0.5])  ## made video for cook cabbage

    goals = [("Cleaned", cabbage)]  ## made video
    goals = [("Cleaned", cabbage), ("Cooked", cabbage)]  ## successful
    goals = [("Cleaned", cabbage), ("Cooked", cabbage),
             ("On", cabbage, name_to_body('braiser_bottom')), ("Cooked", chicken)]  ## successful

    ## ---- for record a crowded fridge
    world.open_joint_by_name('fridge_door', pstn=1.5)
    set_camera_target_body(name_to_body('fridge'), dx=0.3, dy=0, dz=0.3)
    name_to_object('meatchicken').set_pose(((1.428, 4.347, 0.914), (0, 0, 0, 1)))
    # world.put_on_surface(cabbage, 'indigo_tmp')
    # world.put_on_surface(lid, 'hitman_countertop')
    # goals = [("On", cabbage, name_to_body('braiser_bottom')),
    #          ("On", chicken, name_to_body('braiser_bottom'))]  ## successful

    set_all_static()
    state = State(world)
    exogenous = []

    pddlstream_problem = pddlstream_from_state_goal(state, goals, custom_limits=args.base_limits,
                                                    domain_pddl=args.domain_pddl, stream_pddl=args.stream_pddl)

    # set_camera_target_body(name_to_body('microwave'), dx=3, dy=2, dz=1)
    return state, exogenous, goals, pddlstream_problem


def test_pr2_cabinets(args):
    world = create_world(args)
    world.set_skip_joints()

    camera_pose = ((3.7, 8, 1.3), (0.5, 0.5, -0.5, -0.5))
    custom_limits = {0: (0, 4), 1: (5, 12), 2: (0, 2)}

    robot = create_pr2_robot(world, base_q=(2.5, 6, PI / 2 + PI / 2))

    pot, lid, turkey, counter, oil, vinegar = load_cabinet_test_scene(world, MORE_MOVABLE=True)
    set_camera_target_body(lid, dx=2, dy=0, dz=0.5)
    world.remove_object(world.name_to_object('braiserlid'))
    world.remove_object(world.name_to_object('turkey'))
    world.remove_object(world.name_to_object('cabbage'))

    # world.add_camera(camera_pose)
    # world.visualize_image(((3.7, 8, 1.3), (0.5, 0.5, -0.5, -0.5)))

    oven = world.name_to_body('oven')
    counter = world.name_to_body('indigo_tmp')
    left_door = world.name_to_body('chewie_door_left_joint')
    right_door = world.name_to_body('chewie_door_right_joint')
    right_cabinet = world.name_to_body('dagger')
    body = oil
    world.open_all_doors_drawers()

    ## ------- grasping and placing various objects
    goals = ("test_object_grasps", body)
    # goals = [('OpenedJoint', left_door)]
    # goals = [('OpenedJoint', left_door), ('OpenedJoint', right_door)]
    # goals = [('Holding', 'hand', body), ('OpenedJoint', left_door), ('OpenedJoint', right_door)]
    # goals = [('On', oil, counter), ('OpenedJoint', left_door), ('OpenedJoint', right_door)]
    #
    # ## ------- help task planning find joints to manipulation
    # goals = [('Toggled', left_door)]
    # goals = ('test_reachable_pose', turkey)
    # goals = ('test_sample_wconf', body)
    # goals = ('test_at_reachable_pose', body)
    # goals = [('Toggled', left_door), ('Holding', 'hand', body)] ## successful
    # goals = [('Holding', 'hand', body)]
    # goals = [("On", oil, counter)]

    ## door is partially open, enough gap to get in but not enough to get out → HPN will fail?
    # open_joint(left_door[0], left_door[1], extent=0.3)
    # open_joint(right_door[0], right_door[1], extent=0.3)
    # goals = [("On", oil, counter)]

    ## must exist after all objects and categories have been set
    set_renderer(True)
    set_all_static()
    state = State(world, grasp_types=robot.grasp_types)
    exogenous = []

    pddlstream_problem = pddlstream_from_state_goal(state, goals, custom_limits=custom_limits,
                                                    domain_pddl=args.domain_pddl, stream_pddl=args.stream_pddl)
    save_to_kitchen_worlds(state, pddlstream_problem, exp_name='test_pr2_pick_1_opened',
                           floorplan='counter.svg', world_name='test_pr2_pick_1_opened', exit=False)
    return state, exogenous, goals, pddlstream_problem


####################################################


def test_nvidia_kitchen_domain(args, world_loader_fn, initial_xy=(1.5, 6), **kwargs):
    set_camera_pose(camera_point=[3, 5, 3], target_point=[0, 6, 1])  ## target fridge
    set_camera_pose(camera_point=[3, 7, 4], target_point=[0, 7, 1])
    if 'robot_builder_args' not in kwargs:
        kwargs['robot_builder_args'] = args.robot_builder_args
    if 'custom_limits' not in kwargs['robot_builder_args']:
        kwargs['robot_builder_args']['custom_limits'] = ((1, 3, 0), (5, 10, 3))
    if 'initial_xy' not in kwargs['robot_builder_args']:
        kwargs['robot_builder_args']['initial_xy'] = initial_xy
    return problem_template(args, robot_builder_fn=build_robot_from_args,
                            world_loader_fn=world_loader_fn, **kwargs)


def test_skill_knob_faucet(args, **kwargs):
    def loader_fn(world, **world_builder_args):
        world.set_skip_joints()

        surfaces = {
            'Basin': {
                'faucet_platform': ['Faucet'],
                'basin_bottom': []
            }
        }
        floor = load_kitchen_floor_plan(world, surfaces=surfaces, plan_name='basin.svg')
        world.remove_object(floor)

        faucet, left_knob = load_basin_faucet(world)
        set_camera_target_body(faucet, dx=1.4, dy=1.5, dz=1)
        goals = ('test_handle_grasps', left_knob)
        goals = [("HandleGrasped", 'left', left_knob)]
        goals = [("GraspedHandle", left_knob)]

        return {'goals': goals}

    return test_nvidia_kitchen_domain(args, loader_fn, **kwargs)


def test_kitchen_drawers(args, **kwargs):
    def loader_fn(world, difficulty=15, **world_builder_args):
        """
        0:  open the drawer
        1:  drawer is opened, drawer is a static link
            pick up the cabbage
        1x: drawer is opened, drawer is a movable link
            11: pick up the cabbage
            12: pick up the cabbage and place it on the counter
            1y: pick up the cabbage and place it in the pot
                13: Rummy, PR2 dual-arm with large torso joint limits
                14: PR2 dual-arm with normal torso joint limits
                15: PR2 single-arm
        111: drawer is closed, drawer is a movable link
            pick up the cabbage and place it on the counter
        2:  random test cases
        """
        surfaces = {
            'counter': {
                'front_right_stove': [],
                'hitman_countertop': [],
                'indigo_tmp': ['BraiserBody', 'BraiserLid'],
            },
        }

        cabbage = load_full_kitchen(world, surfaces=surfaces)
        skeleton = []
        objects = []

        name_to_body = world.name_to_body
        drawer = 'hitman_drawer'  ## 'baker_joint'  ## 'indigo_door_left_joint'  ##
        drawer = world.add_joints_by_keyword('counter', 'hitman_drawer')[0]
        drawer = world.add_joints_by_keyword('counter', 'indigo_drawer')[0]
        world.add_to_cat(cabbage, 'graspable')

        ## ================================================
        ##  GOALS
        ## ================================================
        if difficulty == 0:
            # --------- grasp and open handle - hitman
            set_camera_pose(camera_point=[1.3, 7.5, 2], target_point=[1, 7, 0])
            goals = ('test_handle_grasps', 'hitman_drawer_top_joint')
            goals = [("HandleGrasped", 'left', name_to_body('hitman_drawer_top_joint'))]
            # world.put_in_space(cabbage, 'hitman_drawer_top')
            goals = [("AtPosition", name_to_body('hitman_drawer_top_joint'),
                      Position(name_to_body('hitman_drawer_top_joint'), 'max'))]
            goals = [("AtPosition", name_to_body('hitman_drawer_bottom_joint'),
                      Position(name_to_body('hitman_drawer_bottom_joint'), 'max'))]
            goals = [("OpenedJoint", name_to_body('hitman_drawer_top_joint'))]

        if difficulty in [1, 11, 12, 13, 14, 15, 111]:
            ## --------- use kitchen fridge and cabinet
            # world.put_on_surface(cabbage, 'indigo_tmp')
            # world.open_joint_by_name('hitman_drawer_top_joint')
            goals = [("Holding", 'left', cabbage)]
            goals = [("In", cabbage, name_to_body('hitman_drawer_top'))]

            ## --------- grasp and open handle - indigo
            set_camera_pose(camera_point=[1.3, 9, 2], target_point=[1, 9, 0])
            world.make_transparent('indigo_tmp')

            # goals = [("AtPosition", name_to_body('indigo_drawer_top_joint'),
            #           Position(name_to_body('indigo_drawer_top_joint'), 'max'))]
            # goals = [("OpenedJoint", name_to_body('indigo_drawer_top_joint'))]

            # goals = [("On", name_to_body('cabbage'), name_to_body('indigo_tmp'))]

            # goals = [("OpenedJoint", name_to_body('indigo_drawer_top_joint')),
            #          ("On", name_to_body('cabbage'), name_to_body('indigo_tmp'))]
            # goals = ("test_object_grasps", 'cabbage')
            # goals = ('test_grasp_ik', 'cabbage')

            # world.robot.add_operator_names_to_remove(['place', 'place_to_supporter'])
            world.robot.add_operator_names_to_remove(['place_to_supporter'])

            arm = 'left'
            drawer = name_to_body('indigo_drawer_top_joint')
            counter = name_to_body('indigo_tmp')
            stovetop = name_to_body('front_right_stove')

            ## drawer as static link
            if difficulty == 1:
                world.open_joint_by_name('indigo_drawer_top_joint')
                world.put_on_surface(cabbage, 'indigo_tmp')
                goals = [("Holding", "left", cabbage.pybullet_name)]  ## successful

            ## drawer as movable link, don't open drawer
            if difficulty in [11, 12, 13, 14, 15]:

                objects += [drawer]
                world.open_joint_by_name('indigo_drawer_top_joint')
                world.put_in_space(cabbage, 'indigo_drawer_top')
                cabbage = cabbage.body
                braiserlid = name_to_body('braiserlid')

                if difficulty == 11:
                    goals = [("Holding", "left", cabbage)]  ## successful
                    # skeleton += [(k, arm, cabbage) for k in pick_place_rel_actions[:1]]

                elif difficulty == 12:
                    goals = [("On", cabbage, counter)]  ## successful
                    skeleton += [(k, arm, cabbage) for k in pick_place_rel_actions[:1]]
                    skeleton += [(k, arm, cabbage) for k in pick_arrange_actions[1:]]

                elif difficulty in [13, 14, 15]:
                    """
                    13: Rummy, PR2 dual-arm with large torso joint limits
                    14: PR2 dual-arm with normal torso joint limits
                    15: PR2 single-arm
                    """
                    load_pot_lid(world)
                    braiser = name_to_body('braiser_bottom')
                    world.put_on_surface(name_to_body('braiserbody'), 'indigo_tmp').adjust_pose(y=8.922, yaw=0)
                    world.put_on_surface(name_to_body('braiserbody'), 'indigo_tmp').adjust_pose(y=8.922, yaw=0, x=0.585, z=0.923)
                    world.put_on_surface(braiserlid, 'indigo_tmp').adjust_pose(x=0.7, y=8.622)
                    # world.name_to_object('braiserlid').change_pose_interactive()
                    world.make_transparent('braiserbody')

                    goals = [("On", cabbage, braiser)]  ## successful

                    skeleton += [(k, arm, cabbage) for k in pick_place_rel_actions[:1]]

                    if difficulty == 15:
                        # world.robot.remove_arm('right')
                        ## need to remove pick's precondition (not (picked))

                        # goals = [("On", cabbage, stovetop)]  ## successful
                        # objects.append(stovetop)

                        # goals = [("On", cabbage, counter)]  ## successful
                        # goals = [("Holding", arm, cabbage)]  ## successful

                        objects += [counter]
                        # goals += [("HandleGrasped", arm, drawer)]  ## successful
                        skeleton += [(k, arm, cabbage, counter) for k in pick_arrange_actions[1:]]
                        # skeleton += [(k, arm, cabbage) for k in pick_place_actions[1:]]
                        skeleton += [(k, arm, drawer) for k in pull_with_link_actions]
                        skeleton += [(k, arm, cabbage) for k in pick_arrange_actions[:1]]
                        # skeleton += [(k, arm, cabbage, counter) for k in pick_arrange_actions[1:]]

                    if difficulty == 14:
                        skeleton += [(k, 'right', drawer) for k in pull_with_link_actions]

                    skeleton += [(k, arm, cabbage, braiser) for k in pick_arrange_actions[1:]]

            ## drawer as movable link
            if difficulty == 111:
                goals = [("On", cabbage, counter)]
                skeleton += [(k, arm, drawer) for k in pull_with_link_actions]
                skeleton += [(k, arm, cabbage) for k in pick_place_rel_actions[:1]]
                # skeleton += [(k, 'right', drawer) for k in pull_with_link_actions]
                skeleton += [(k, arm, cabbage) for k in pick_arrange_actions[1:]]

        if difficulty == 2:

            ## --------- ideally, requires closing the drawer in order to put on
            # world.open_joint_by_name('indigo_drawer_top_joint')
            # world.put_on_surface(cabbage, 'hitman_countertop')
            # goals = [("On", name_to_body('cabbage'), name_to_body('indigo_tmp'))]

            ## --------- test moving objects from drawer to counter
            # goals = [("On", cabbage, name_to_body('hitman_countertop'))] ## successful

            # world.put_in_space(cabbage, 'hitman_drawer_top')
            # goals = [("On", cabbage, name_to_body('hitman_countertop'))] ## unsuccessful

            # world.open_joint_by_name('hitman_drawer_top_joint')
            # world.put_in_space(cabbage, 'hitman_drawer_top', learned=True)
            # goals = ("test_object_grasps", 'cabbage')
            # goals = ('test_grasp_ik', 'cabbage')
            # goals = [("Holding", "left", cabbage)] ## unsuccessful
            goals = [("On", cabbage, name_to_body('hitman_countertop'))]  ## unsuccessful

            ## --------- temporary
            # goals = [("HandleGrasped", 'left', name_to_body('hitman_drawer_top_joint'))]

        world.remove_bodies_from_planning(goals, exceptions=objects)
        return {'goals': goals, 'skeleton': skeleton}

    return test_nvidia_kitchen_domain(args, loader_fn, **kwargs)


def test_kitchen_doors(args, **kwargs):
    def loader_fn(world, difficulty=2, supporter_name='sektion', arm='left'):
        """
        difficulty:
        0:  open door to the supporter
        1:  door is open, pick up something from the supporter
        2:  open the door and pick up from the supporter (with skeleton help)
        3:  open the door and pick up from the supporter (without skeleton help)
        """
        world.set_skip_joints()
        spaces = {
            'counter': {
                'sektion': ['VinegarBottle'],  ## 'OilBottle',
                # 'dagger': [],  ## 'Salter'
                # 'hitman_drawer_top': [],  ## 'Pan'
                # 'hitman_drawer_bottom': [],
                # 'indigo_drawer_top': [],  ## 'Fork', 'Knife'
                # 'indigo_drawer_bottom': ['Fork', 'Knife'],
                # 'indigo_tmp': ['PotBody']
            },
        }
        surfaces = {
            'counter': {
                # 'front_left_stove': [],  ## 'Kettle'
                'front_right_stove': ['BraiserBody'],  ## 'PotBody',
                # 'back_left_stove': [],
                # 'back_right_stove': [],
                # 'range': [], ##
                # 'hitman_countertop': [],  ##  'Microwave'
                # 'indigo_tmp': ['BraiserLid'],  ## 'MeatTurkeyLeg', 'Toaster',
            },
            'fridge': {
                # 'shelf_top': [],
                'shelf_bottom': [],
            }
        }
        load_full_kitchen(world, surfaces=surfaces, spaces=spaces, load_cabbage=False)
        bottle = world.name_to_object('vinegarbottle')
        movable = bottle.body

        """ add doors """
        supporter_to_doors = load_nvidia_kitchen_joints(world)
        supporters = [supporter_name]  ## 'sektion', 'dagger', 'indigo_tmp', 'shelf_bottom'
        selected_id = 0  ## random.choice(range(len(counter_doors)))
        selected_door = None
        for supporter_id, supporter_name in enumerate(supporters):
            for door, pstn in supporter_to_doors[supporter_name]:
                world.add_joint_object(door[0], door[1], 'door')
                world.make_transparent((door[0], door[1]))
                if difficulty == 1:
                    world.open_joint(door, extent=pstn)

            if supporter_id == selected_id:
                doors = [t[0] for t in supporter_to_doors[supporter_name]]
                place_in_nvidia_kitchen_space(bottle, supporter_name, interactive=False)
                handles = {d: (d[0], world.BODY_TO_OBJECT[d].handle_link) for d in doors}
                selected_door = select_door_closer_to_body(handles, movable)
                if supporter_name == 'shelf_bottom':
                    set_camera_pose(camera_point=[2.717, 4.514, 2.521], target_point=[1, 5, 1])

        """ goals """
        door = selected_door
        skeleton = []
        if difficulty == 0:
            goals = ("test_object_grasps", movable)
            goals = ('test_handle_grasps', door)
            # goals = [("HandleGrasped", 'left', door)]
            # goals = [("AtPosition", door, Position(door, 'min'))]
            goals = [("OpenedJoint", door)]
            # goals = [("GraspedHandle", door)]

        elif difficulty in [1, 2, 3]:
            goals = [("Holding", arm, movable)]

            if difficulty == 1:
                skeleton += [(k, arm, bottle) for k in pick_place_actions[:1]]

            if difficulty == 2:
                goals = [("OpenedJoint", door), ("Holding", arm, movable)]
                skeleton += [(k, arm, door) for k in pull_actions]
                skeleton += [(k, arm, bottle) for k in pick_place_actions[:1]]

        else:
            return
            ## --- for recording door open demo
            # world.open_joint_by_name('chewie_door_left_joint')

        world.remove_bodies_from_planning(goals=goals, skeleton=skeleton)
        return {'goals': goals, 'skeleton': skeleton}

    return test_nvidia_kitchen_domain(args, loader_fn, initial_xy=(2, 5), **kwargs)


##########################################################################################


def test_kitchen_braiser(args, **kwargs):
    def loader_fn(world, **world_builder_args):
        spaces = {
            'counter': {
                'sektion': [],
            },
        }
        surfaces = {
            'counter': {
                'front_left_stove': [],
                'front_right_stove': ['BraiserBody'],
                'indigo_tmp': ['BraiserLid'],
            },
        }
        custom_supports = {
            'fork': 'indigo_tmp'
        }

        load_full_kitchen(world, surfaces=surfaces, spaces=spaces, load_cabbage=False)
        movables, movable_to_doors = load_nvidia_kitchen_movables(world, custom_supports=custom_supports)
        load_cooking_mechanism(world)

        movable = world.name_to_body('fork')
        obstacle = world.name_to_body('braiserlid')
        counter = world.name_to_body('indigo_tmp')
        side_surface = world.name_to_body('front_left_stove')
        target_surface = world.name_to_body('braiser_bottom')
        counter_surface = world.name_to_body('indigo_tmp')
        objects = [obstacle, side_surface]
        skeleton = []
        subgoals = []

        #########################################################################

        """ goals """
        arm = 'left'
        # goals = [("Holding", arm, obstacle)]
        goals = [("Picked", obstacle)]
        # goals = [("On", obstacle, side_surface)]
        # goals = [("On", obstacle, counter_surface)]

        # goals = [("Holding", arm, movable)]
        # goals = [("On", movable, target_surface)]

        #########################################################################

        if args.use_skeleton_constraints:
            skeleton += [(k, arm, obstacle) for k in pick_place_actions]
            skeleton += [(k, arm, movable) for k in pick_place_actions]

        if args.use_subgoal_constraints:
            subgoals = [('on', obstacle, counter)] + goals

        #########################################################################

        ## --- for recording door open demo
        # world.open_joint_by_name('chewie_door_left_joint')

        world.remove_bodies_from_planning(goals, exceptions=objects)

        return {'goals': goals, 'skeleton': skeleton, 'subgoals': subgoals}

    return test_nvidia_kitchen_domain(args, loader_fn, initial_xy=(2, 5), **kwargs)


def test_skill_knob_stove(args, **kwargs):
    """
    difficulty ==
    0:  turn the knob to the right front stove
    1:  move the pot to the counter before turning on the stove (then move the pot to the stove)
    12: move the lid to the counter before turning on the stove (then move the pot to the stove)
    2:  turn on the knob of the other stove, (then move the pot there)
    """
    def loader_fn(world, difficulty=12, **world_builder_args):
        surfaces = {
            'counter': {
                'front_left_stove': [],
                'front_right_stove': [],
                'indigo_tmp': [],
            }
        }
        if difficulty in [0]:
            surfaces['counter']['indigo_tmp'] = ['BraiserBody']
        if difficulty in [1, 12]:
            surfaces['counter']['front_right_stove'] = ['BraiserBody']
            if difficulty in [12]:
                surfaces['counter']['indigo_tmp'] = ['BraiserLid']

        floor = load_floor_plan(world, plan_name='counter.svg', surfaces=surfaces)
        world.remove_object(floor)

        name_to_body = world.name_to_body

        oven = name_to_body('oven')
        set_camera_target_body(oven, dx=0.5, dy=-0.3, dz=1.1)

        knob_names = ['knob_joint_2', 'knob_joint_3']
        load_stove_knobs(world, knob_names)

        right_knob = name_to_body(knob_names[0])
        left_knob = name_to_body(knob_names[1])
        arm = world.robot.arms[0]
        surface = name_to_body('indigo_tmp')

        objects = []
        skeleton = []
        if difficulty == 0:
            knob = right_knob
            goals = ('test_handle_grasps', knob)  ## for choosing grasps
            goals = [("HandleGrasped", 'left', knob)]
            goals = [("GraspedHandle", knob)]
            goals = [("PulledOneAction", knob)]

        elif difficulty in [1, 12]:

            pot = name_to_body('braiserbody')
            world.add_to_cat('braiserbody', 'movable')
            fix_braiser_orientation(world)

            if difficulty in [1]:
                goals = [("GraspedHandle", left_knob)]
                goals = ('test_object_grasps', pot)
                goals = [("Holding", arm, pot)]
                goals = [("On", pot, surface)]

            elif difficulty in [12]:
                lid = world.name_to_body('braiserlid')
                world.put_on_surface(lid, 'braiserbody')
                world.add_to_cat('braiserlid', 'movable')
                goals = [("On", lid, surface)]
                goals = [("On", pot, surface)]
                # goals = [("GraspedHandle", right_knob)]; objects += [pot, lid, surface]

                skeleton += [(k, arm, pot) for k in pick_place_actions]
                skeleton += [(k, arm, right_knob) for k in pull_actions]

            goals = [("GraspedHandle", right_knob)]; objects += [pot, surface]

        else:
            print('Invalid difficulty level:', difficulty)
            return {}

        world.remove_bodies_from_planning(goals, exceptions=objects)

        return {'goals': goals, 'skeleton': skeleton}

    return test_nvidia_kitchen_domain(args, loader_fn, initial_xy=(2, 5), **kwargs)


def test_kitchen_chicken_soup(args, **kwargs):
    """
    Note: The grasp poses of the fork need to be hand-specified
    """
    def loader_fn(world, **world_builder_args):
        goal_object = ['chicken-leg', 'cabbage', 'fork', 'salt-shaker'][0]
        open_doors_for = []  ## goal_object
        objects, movables = load_open_problem_kitchen(world, open_doors_for=open_doors_for)

        """ goals """
        arm = random.choice(world.robot.arms)
        objects = []
        fridge_door = world.name_to_body('fridge_door')
        counter = world.name_to_body('indigo_tmp')
        drawer_joint = world.name_to_body('indigo_drawer_top_joint')
        drawer_link = world.name_to_body('indigo_drawer_top')
        cabinet_doors = [world.name_to_body(name) for name in ['chewie_door_left_joint', 'chewie_door_right_joint']]
        dishwasher_space = world.name_to_body('upper_shelf')
        dishwasher_joint = world.name_to_body('dishwasher_door')
        cabinet_space = world.name_to_body('sektion')
        braiser_body = world.name_to_body('braiserbody')

        joint = fridge_door  ## fridge_door | cabinet_doors[0] | cabinet_doors[1]
        # goals = ('test_joint_open', joint)
        # goals = ('test_joint_closed', joint)
        goals = ('test_handle_grasps', joint)
        goals = [("OpenedJoint", joint)]
        # goals = [("ClosedJoint", joint)]
        world.open_joint(joint, extent=1)

        joint = drawer_joint
        movable = movables['fork']
        # goals = ('test_handle_grasps', joint)  ## fail for some reason
        goals = [("OpenedJoint", joint)]
        # goals = ("test_object_grasps", movable);
        # goals = [("Holding", arm, movable)]; world.open_joint(joint, extent=1)
        goals = [("OpenedJoint", joint), ("Holding", arm, movable)]

        # movable = movables[goal_object]
        # # goals = ("test_object_grasps", movable)
        # goals = [("Holding", arm, movable)]
        # goals = [("On", movable, counter)]
        # goals = [("In", movable, cabinet_space)]
        #
        # # goals = ('test_relpose_inside_gen', (movable, drawer_link))
        # # goals = [("In", movable, drawer_link)]
        #
        # # lid = world.name_to_body('braiserlid')
        # # braiser =  world.name_to_body('braiserbody')
        # # world.BODY_TO_OBJECT[counter].place_obj(world.BODY_TO_OBJECT[lid])
        # # world.add_to_cat(lid, 'movable')
        # # world.add_to_cat(braiser, 'surface')
        # # goals = [("On", lid, braiser)]
        #
        # obj = world.name_to_object('chicken-leg')
        # world.name_to_object('indigo_tmp').place_obj(world.name_to_object('braiserlid'))
        # goals = [("In", movable, braiser_body)]
        #
        # movable = world.name_to_body('chicken-leg')
        # goals = ("test_object_grasps", movable)

        # sample_obj_in_body_link_space(obj, braiser_body, link=None, PLACEMENT_ONLY=False,
        #                               draw=False, verbose=True, visualize=True, max_trial=3)

        #########################################################################

        # objects += [movable]  ## Holding
        # objects += [drawer_joint]  ## OpenedJoint
        # objects += [drawer_link]  ## In (place)
        # objects += [drawer_joint, drawer_link]  ## In (place_rel)
        # objects += [dishwasher_space]  ## dishwasher_joint,
        # objects += [dishwasher_joint, dishwasher_space]
        # objects += cabinet_doors

        subgoals = None
        skeleton = []

        #########################################################################

        if goals[0][0] == "ClosedJoint":
            joint = goals[0][1]
            world.open_joint(joint, extent=0.5)

        if goals == [("On", movable, counter)]:
            skeleton += [('pick', arm, movable), ('arrange', arm, movable, counter)]
            goals += [("Arranged", movable)]

        if goals == [("In", movable, cabinet_space)]:
            skeleton += [('pick', arm, movable), ('arrange', arm, movable, cabinet_space)]
            goals += [("Arranged", movable)]
            [world.open_joint(cc, extent=1) for cc in cabinet_doors]

        #########################################################################

        # skeleton += [(k, arm, goal_object) for k in pick_place_actions[:1]]
        # skeleton += [(k, arm, joint) for k in pull_actions]
        # skeleton += [(k, arm, drawer_joint) for k in pull_with_link_actions]
        # skeleton += [(k, arm, movable) for k in pick_place_rel_actions[:1]]
        # skeleton += [(k, arm, movable) for k in ['pick', 'place_to_supporter']]
        # skeleton += [(k, arm, movable) for k in ['pick_from_supporter', 'place']]
        # skeleton += [(k, arm, movable) for k in pick_place_rel_actions]

        #########################################################################

        ## --- for recording door open demo
        # world.open_joint_by_name('chewie_door_left_joint')

        world.remove_bodies_from_planning(goals, exceptions=objects)

        return {'goals': goals, 'skeleton': skeleton, 'subgoals': subgoals}

    return test_nvidia_kitchen_domain(args, loader_fn, initial_xy=(2, 5), **kwargs)


def test_kitchen_plan_constraints(args, **kwargs):
    """
    Note: The grasp poses of the fork need to be hand-specified
    """
    def loader_fn(world, **world_builder_args):
        goal_object = ['chicken-leg', 'cabbage', 'fork', 'salt-shaker'][2]
        open_doors_for = []  ## goal_object
        objects, movables = load_open_problem_kitchen(world, open_doors_for=open_doors_for)
        plate = load_plate_on_counter(world, counter_name='indigo_tmp')

        """ goals """
        robot = world.robot
        arm = 'left'
        objects = []
        fridge_door = world.name_to_body('fridge_door')
        braiser_bottom = world.name_to_body('braiser_bottom')
        braiser_lid = world.name_to_body('braiserlid')
        counter = world.name_to_body('indigo_tmp')
        drawer_joint = world.name_to_body('indigo_drawer_top_joint')
        drawer_link = world.name_to_body('indigo_drawer_top')
        cabinet_doors = [world.name_to_body(name) for name in ['chewie_door_left_joint', 'chewie_door_right_joint']]

        joint = drawer_joint
        goals = [("OpenedJoint", joint)]

        movable = movables[goal_object]
        # goals = ("test_object_grasps", movable)
        # goals = [("Holding", arm, movable)]
        # goals = [("On", movable, counter)]; world.open_joint(joint, extent=0.5)
        # goals = [("In", movable, drawer_link)]

        # goals = [("On", movable, braiser_bottom)]; world.open_joint(joint, extent=0.5)
        # goals = [("On", braiser_lid, counter), ("Holding", arm, movable)]
        # goals = [("On", braiser_lid, counter)]
        goals = [("On", movable, plate)]  ## ; world.open_joint(joint, extent=0.5)

        #########################################################################

        # objects += [movable]  ## Holding
        # objects += [drawer_joint]  ## OpenedJoint
        # objects += [drawer_link]  ## In (place)
        objects += [drawer_joint, drawer_link]  ## On (place_rel)
        objects += [braiser_lid, counter]  ## On (braiser_bottom)
        # objects += [dishwasher_space]  ## dishwasher_joint,
        # objects += [dishwasher_joint, dishwasher_space]
        # objects += cabinet_doors

        #########################################################################

        if goals[0][0] == "ClosedJoint":
            joint = goals[0][1]
            world.open_joint(joint, extent=0.5)

        #########################################################################

        subgoals = None
        if args.use_subgoal_constraints:
            if goals == [("On", movables['fork'], counter)]:
                subgoals = [("OpenedJoint", drawer_joint), ("On", movable, counter)]

        #########################################################################

        skeleton = []
        if args.use_skeleton_constraints:

            if goals == [("OpenedJoint", drawer_joint)]:
                skeleton += [(k, arm, drawer_joint) for k in pull_with_link_actions]

            if goals == [("On", braiser_lid, counter), ("Holding", arm, movable)]:
                skeleton += [(k, arm, braiser_lid) for k in pick_place_actions]
                skeleton += [(k, arm, drawer_joint) for k in pull_with_link_actions]
                skeleton += [('pick_from_supporter', arm, movable)]

            if goals == [("On", movables['fork'], braiser_bottom)]:
                skeleton += [(k, arm, braiser_lid) for k in pick_place_actions]
                skeleton += [(k, arm, drawer_joint) for k in pull_with_link_actions]
                skeleton += [('pick_from_supporter', arm, movable)]
                skeleton += [('place', arm, movable)]

            if goals in [
                [("On", movables['fork'], counter)],
                [("On", movables['fork'], plate)]
            ]:
                if robot.dual_arm:
                    skeleton += [(k, 'right', drawer_joint) for k in pull_with_link_actions]
                    skeleton += [('pick_from_supporter', 'right', movable)]
                    skeleton += [(k, 'left', drawer_joint) for k in pull_with_link_actions]
                    skeleton += [('place', 'right', movable)]
                else:
                    skeleton += [(k, arm, drawer_joint) for k in pull_with_link_actions]
                    skeleton += [(k, arm, movable) for k in ['pick_from_supporter', 'place']]
                set_camera_target_body(drawer_link, dx=1, dy=0.5, dz=2)

            if goals in [("test_object_grasps", movables['salt-shaker']), ("test_object_grasps", movables['pepper-shaker']),
                         [('Holding', arm, movables['salt-shaker'])], [('Holding', arm, movables['pepper-shaker'])]]:

                if goals in [("test_object_grasps", movables['salt-shaker']), [('Holding', arm, movables['salt-shaker'])]]:
                    skeleton += [(k, arm, cabinet_doors[0]) for k in pull_actions]

                if goals in [("test_object_grasps", movables['pepper-shaker']), [('Holding', arm, movables['pepper-shaker'])]]:
                    skeleton += [(k, arm, cabinet_doors[1]) for k in pull_actions]

                body = goals[-1] if isinstance(goals, tuple) else goals[0][-1]
                skeleton += [(pick_place_actions[0], arm, body)]
                set_camera_target_body(body, dx=1, dy=0, dz=1.75)

        #########################################################################

        ## --- for recording door open demo
        # world.open_joint_by_name('chewie_door_left_joint')

        world.remove_bodies_from_planning(goals, exceptions=objects, skeleton=skeleton, subgoals=subgoals)

        return {'goals': goals, 'skeleton': skeleton, 'subgoals': subgoals}

    return test_nvidia_kitchen_domain(args, loader_fn, initial_xy=(2, 8), **kwargs)


def test_kitchen_sprinkle(args, **kwargs):
    """
    Note: The grasp poses of the fork need to be hand-specified
    """
    def loader_fn(world, **world_builder_args):
        robot = world.robot

        open_doors_for = []  ## goal_object
        objects, movables = load_open_problem_kitchen(world, open_doors_for=open_doors_for)
        plate = load_plate_on_counter(world, counter_name='indigo_tmp')

        salt_shaker = world.name_to_body('salt-shaker')
        salt_shaker = world.name_to_body('pepper-shaker')
        plate = world.name_to_body('plate')
        braiser = world.name_to_object('braiserbody')
        lid = world.name_to_object('braiserlid')
        braiser_bottom = world.name_to_object('braiser_bottom')
        counter = world.name_to_body('indigo_tmp')
        side_surface = world.name_to_object('front_left_stove')
        left_door = world.name_to_body('chewie_door_left_joint')
        right_door = world.name_to_body('chewie_door_right_joint')
        arm = robot.arms[0]

        objects = []
        skeleton = []
        subgoals = []

        world.add_to_cat(salt_shaker, 'sprinkler')
        world.add_to_cat(braiser_bottom, 'region')
        world.add_to_cat(braiser, 'region')

        ## changes to env for different goals
        side_surface.place_obj(lid)
        # world.open_joint(left_door)
        # world.open_joint(right_door)
        objects += [left_door, right_door]

        """ test 1: sprinkle into """
        # goals = [['OpenedJoint', right_door]]
        goals = [['Holding', arm, salt_shaker]]
        goals = ('test_pose_above_gen', (salt_shaker, plate))
        # goals = [['SprinkledTo', salt_shaker, plate]]
        # goals = [['SprinkledTo', salt_shaker, braiser_bottom.pybullet_name]]
        # goals = [['SprinkledTo', salt_shaker, braiser.pybullet_name]]

        """ test 2: remove movable obstacles """
        # put_lid_on_braiser(world, lid, braiser)
        # goals = [['On', lid, plate]]
        # goals = [['SprinkledTo', salt_shaker, braiser_bottom.pybullet_name]]
        # objects += [lid] + [plate] ## + [side_surface.pybullet_name]
        # skeleton += [(k, arm, lid) for k in pick_place_actions]
        # skeleton += [(k, arm, salt_shaker) for k in pick_sprinkle_actions]

        """ test 3: remove articulated obstacles """
        # world.close_joint(left_door)
        # world.close_joint(right_door)
        # goals = ('test_joint_open', right_door)

        ## need push rim action to open the joint
        if 'mobile_v3' in args.domain_pddl:
            goals = [['ClosedJoint', right_door]]
            objects += [right_door]

        world.remove_bodies_from_planning(goals, exceptions=objects, skeleton=skeleton, subgoals=subgoals)

        return {'goals': goals, 'skeleton': skeleton, 'subgoals': subgoals}

    return test_nvidia_kitchen_domain(args, loader_fn, initial_xy=(2, 8), **kwargs)


def test_kitchen_faucet_braiser_and_stove(args, **kwargs):
    def loader_fn(world, case=4, **world_builder_args):
        robot = world.robot
        objects, movables = load_open_problem_kitchen(world)
        faucet, left_knob = load_basin_faucet(world)
        set_camera_target_body(faucet, dx=1.4, dy=1.5, dz=1)

        if case == 0:
            goals = ('test_handle_grasps', left_knob)
            goals = [("HandleGrasped", 'left', left_knob)]
            goals = [("GraspedHandle", left_knob)]

            world.name_to_object('hitman_countertop').place_obj(world.name_to_object('braiserlid'))
            braiser = world.name_to_body('braiserbody')
            basin = world.name_to_body('basin_bottom')
            goals = [("On", braiser, basin)]
            goals = [("On", braiser, basin), ("GraspedHandle", left_knob)]

        ## debug the aabb of hitman_tmp
        if case == 1:
            left_counter = world.name_to_body('hitman_countertop')
            set_camera_target_body(left_counter, dx=1.4, dy=-1.5, dz=1)
            goals = ('test_pose_gen', (braiser, left_counter))
            goals = [("On", braiser, left_counter)]

        ## debug picking from fridge
        elif case == 2:
            fridge_door = world.name_to_body('fridge_door')
            chicken = world.name_to_body('chicken-leg')
            world.open_joint(fridge_door)
            goals = ("test_object_grasps", chicken)
            goals = [("picked", chicken)]

        ## debug picking lid
        elif case == 3:
            world.name_to_object('front_left_stove').place_obj(world.name_to_object('braiserlid'))
            lid = movable = world.name_to_body('braiserlid')
            body = movable = world.name_to_body('braiserbody')
            set_camera_target_body(movable, dx=1.4, dy=1.5, dz=1)
            goals = ("test_object_grasps", movable)
            goals = [("picked", lid)]
            # goals = ('test_pose_gen', (lid, body))
            goals = [("on", lid, body)]

        ## debug picking lid
        elif case == 4:
            ## compute some poses
            braiser = world.name_to_object('braiserbody')
            _, quat = braiser.get_pose()
            world.name_to_object('front_left_stove').place_obj(braiser)
            body_pose = braiser.get_pose()
            body_pose = (body_pose[0], quat)
            surface_pose = world.name_to_object('front_left_stove').get_pose()
            rel_pose = multiply(invert(surface_pose), body_pose)

            braiser.set_pose(multiply(surface_pose, rel_pose))
            braiser.set_pose(multiply(world.name_to_object('front_right_stove').get_pose(), rel_pose))

            stove = world.name_to_body('front_left_stove')
            body = world.name_to_body('braiserbody')
            goals = ('test_pose_gen', (body, stove))
            # goals = [("on", body, stove)]

        else:
            goals = []

        world.remove_bodies_from_planning(goals)
        return {'goals': goals}

    return test_nvidia_kitchen_domain(args, loader_fn, initial_xy=(2, 8), **kwargs)


def test_kitchen_nudge_door(args, **kwargs):
    """
    Note: The grasp poses of the fork need to be hand-specified
    """
    def loader_fn(world, **world_builder_args):
        robot = world.robot

        open_doors_for = []  ## goal_object
        objects, movables = load_open_problem_kitchen(world, open_doors_for=open_doors_for)

        salt_shaker = world.name_to_body('salt-shaker')
        pepper_shaker = world.name_to_body('pepper-shaker')
        counter = world.name_to_body('indigo_tmp')
        left_door = world.name_to_body('chewie_door_left_joint')
        right_door = world.name_to_body('chewie_door_right_joint')
        arm = robot.arms[0]

        door = right_door
        # world.open_joint(door, extent=0.7)
        world.add_to_cat(salt_shaker, 'graspable')
        world.add_to_cat(pepper_shaker, 'graspable')

        objects = [door]
        skeleton = []
        subgoals = []

        goals = ('test_pull_nudge_joint_positions', None)
        goals = ('test_nudge_grasps', door)
        goals = [("NudgedDoor", door)]
        goals = [("NudgedDoor", right_door), ("Holding", arm, pepper_shaker)]
        # goals = [("NudgedDoor", left_door), ("Holding", arm, salt_shaker)]
        goals = [("Holding", arm, pepper_shaker)]

        # if ("Holding", arm, pepper_shaker) in goals:
        #     world.open_joint(left_door, extent=0.8)
        #     world.open_joint(right_door, extent=0.8)

        # goals = ('test_nudge_back_grasps', door)  ## not working yet
        # goals = [("Closed", door)]
        # goals = [("Closed", door)]

        world.remove_bodies_from_planning(goals, exceptions=objects, skeleton=skeleton, subgoals=subgoals)

        return {'goals': goals, 'skeleton': skeleton, 'subgoals': subgoals}

    return test_nvidia_kitchen_domain(args, loader_fn, initial_xy=(2, 8), **kwargs)
