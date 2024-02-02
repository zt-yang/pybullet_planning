from world_builder.loaders_nvidia_kitchen import *
from world_builder.robot_builders import build_table_domain_robot, build_robot_from_args

from problem_sets.problem_utils import create_world, pddlstream_from_state_goal, save_to_kitchen_worlds, \
    test_template, pull_actions, pick_place_actions, pick_place_rel_actions, pull_with_link_actions


#######################################################

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

    goals = ('test_grasps', cabbage)
    goals = [("Holding", "left", cabbage)]
    goals = [("Holding", "left", lid)]
    # goals = [("On", cabbage, bottom)]
    # goals = [("On", cabbage, world.name_to_body('hitman_tmp'))]

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
    goals = ('test_grasps', lid)
    goals = ('test_pose_gen', (lid, surface))  ## success
    goals = [('AtBConf', Conf(robot, get_group_joints(robot, 'base'), (1.239, 8.136, 3.113)))]
    goals = [("On", lid, surface)]
    # world.remove_object(name_to_object('braiserlid'))
    world.add_to_cat(lid, 'moveable')

    ## ------ remove lid in order to put in egg
    goals = ('test_pose_gen', (egg, bottom))  ## success
    goals = [("On", lid, surface)]  ## success
    # goals = [("On", egg, bottom)]  ## fail

    ## ======== debug option 1 =========
    # uncomment test_grasp_ik(state, state.get_facts()+preimage, name='eggblock') in pr2_agent.py
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

    world.remove_body_from_planning(name_to_body('hitman_tmp'))
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

    floor = load_floor_plan(world, plan_name='kitchen_v3.svg')
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
    world.remove_body_from_planning(name_to_body('hitman_tmp')) ## also added two back stoves
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
    # world.put_on_surface(name_to_object('microwave'), 'hitman_tmp')

    # set_camera_target_body(name_to_body('microwave'), dx=3, dy=2, dz=1)

    # to_lisdf(world, 'kitchen_v2.svg', pddlstream_problem.init, world_name='kitchen_basics', root_path=KITCHEN_WORLD)
    return state, exogenous, goals, pddlstream_problem

#######################################################


def test_opened_space(args):
    world = create_world(args)
    world.set_skip_joints()

    floor = load_floor_plan(world, plan_name='kitchen_v2.svg')
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
    world.remove_body_from_planning(name_to_body('hitman_tmp'))
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


# def test_skill_knob_faucet(args):
#     world = create_world(args)
#     world.set_skip_joints()
#
#     floor = load_floor_plan(world, plan_name='basin.svg')
#     world.remove_object(floor)
#     robot = create_pr2_robot(world, base_q=(2.5, 6, PI / 2 + PI / 2))
#
#     faucet, left_knob = load_basin_faucet(world)
#     set_camera_target_body(faucet, dx=1.4, dy=1.5, dz=1)
#     goals = ('test_handle_grasps', left_knob)
#     goals = [("HandleGrasped", 'left', left_knob)]
#     goals = [("KnobTurned", 'left', left_knob)]
#     goals = [("GraspedHandle", left_knob)]
#
#     set_all_static()
#     state = State(world)
#     exogenous = []
#
#     pddlstream_problem = pddlstream_from_state_goal(state, goals, custom_limits=args.base_limits,
#                                                     domain_pddl=args.domain_pddl, stream_pddl=args.stream_pddl)
#
#     return state, exogenous, goals, pddlstream_problem

#######################################################


def test_skill_knob_stove(args):
    world = create_world(args)
    world.set_skip_joints()

    floor = load_floor_plan(world, plan_name='counter.svg')
    egg = load_experiment_objects(world, CABBAGE_ONLY=True, name='eggblock',
                                  color=TAN, wb=.03, hb=.04)  ##
    bottom, lid = load_pot_lid(world)
    world.remove_object(floor)
    robot = create_pr2_robot(world, base_q=(2.5, 6, PI / 2 + PI / 2))

    name_to_body = world.name_to_body
    name_to_object = world.name_to_object

    oven = name_to_body('oven')
    handles = world.add_joints_by_keyword('oven', 'knob_joint_2', 'knob')
    set_camera_target_body(oven, dx=0.5, dy=-0.3, dz=1.1)

    counter = name_to_body('counter')
    for surface in ['front_left_stove', 'front_right_stove', 'back_left_stove', 'back_right_stove']:
        set_color(counter, GREY, link_from_name(counter, surface))
    world.put_on_surface(egg, 'braiser_bottom')

    left_knob = name_to_object('knob_joint_2')
    set_color(left_knob.body, GREY, left_knob.handle_link)

    left_knob = name_to_body('knob_joint_2')
    goals = ('test_handle_grasps', left_knob) ## for choosing grasps
    goals = [("HandleGrasped", 'left', left_knob)]
    goals = [("KnobTurned", 'left', left_knob)]
    goals = [("GraspedHandle", left_knob)]

    set_all_static()
    state = State(world)
    exogenous = []

    pddlstream_problem = pddlstream_from_state_goal(state, goals, custom_limits=args.base_limits,
                                                    domain_pddl=args.domain_pddl, stream_pddl=args.stream_pddl)
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

    floor = load_floor_plan(world, plan_name='kitchen_v3.svg', surfaces=surfaces)

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

    floor = load_floor_plan(world, plan_name='kitchen_v3.svg', surfaces=surfaces)
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

    floor = load_floor_plan(world, plan_name='kitchen_v3.svg')
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
    # world.put_on_surface(lid, 'hitman_tmp')
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
    goals = ('test_grasps', body)
    # goals = [('OpenedJoint', left_door)]
    # goals = [('OpenedJoint', left_door), ('OpenedJoint', right_door)]
    # goals = [('Holding', 'hand', body), ('OpenedJoint', left_door), ('OpenedJoint', right_door)]
    # goals = [('On', oil, counter), ('OpenedJoint', left_door), ('OpenedJoint', right_door)]
    #
    # ## ------- help task planing find joints to manipulation
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
                           floorplan='counter.svg', world_name='test_pr2_pick_1_opened', EXIT=False)
    return state, exogenous, goals, pddlstream_problem


####################################################


def test_nvidia_kitchen_domain(args, world_loader_fn, initial_xy=(1.5, 6), **kwargs):
    set_camera_pose(camera_point=[3, 5, 3], target_point=[0, 6, 1])
    kwargs['robot_builder_args'].update({
        'custom_limits': ((1, 3, 0), (5, 10, 3)),
        'initial_xy': initial_xy,
        'draw_base_limits': True
    })
    return test_template(args, robot_builder_fn=build_robot_from_args, world_loader_fn=world_loader_fn, **kwargs)


def test_skill_knob_faucet(args, **kwargs):
    def loader_fn(world, **world_builder_args):
        world.set_skip_joints()

        floor = load_floor_plan(world, plan_name='basin.svg')
        world.remove_object(floor)

        faucet, left_knob = load_basin_faucet(world)
        set_camera_target_body(faucet, dx=1.4, dy=1.5, dz=1)
        goals = ('test_handle_grasps', left_knob)
        goals = [("HandleGrasped", 'left', left_knob)]
        goals = [("KnobTurned", 'left', left_knob)]
        goals = [("GraspedHandle", left_knob)]

        return goals, []

    return test_nvidia_kitchen_domain(args, loader_fn, **kwargs)


def test_kitchen_drawers(args, **kwargs):
    def loader_fn(world, **world_builder_args):
        cabbage = load_full_kitchen(world)
        skeleton = []

        name_to_body = world.name_to_body
        drawer = 'hitman_drawer'  ## 'baker_joint' ##'indigo_door_left_joint' ##
        drawer = world.add_joints_by_keyword('counter', 'hitman_drawer')[0]
        drawer = world.add_joints_by_keyword('counter', 'indigo_drawer')[0]

        ## ================================================
        ##  GOALS
        ## ================================================

        ## --------- use kitchen fridge and cabinet
        # world.put_on_surface(cabbage, 'indigo_tmp')
        # world.open_joint_by_name('hitman_drawer_top_joint')
        goals = [("Holding", 'left', cabbage)]
        goals = [("In", cabbage, name_to_body('hitman_drawer_top'))]

        ## --------- grasp and open handle - hitmam
        # set_camera_pose(camera_point=[1.3, 7.5, 2], target_point=[1, 7, 0])
        # goals = ('test_handle_grasps', 'hitman_drawer_top_joint')
        # goals = [("HandleGrasped", 'left', name_to_body('hitman_drawer_top_joint'))]
        # # world.put_in_space(cabbage, 'hitman_drawer_top')
        # goals = [("AtPosition", name_to_body('hitman_drawer_top_joint'),
        #           Position(name_to_body('hitman_drawer_top_joint'), 'max'))]
        # goals = [("AtPosition", name_to_body('hitman_drawer_bottom_joint'),
        #           Position(name_to_body('hitman_drawer_bottom_joint'), 'max'))]
        # goals = [("OpenedJoint", name_to_body('hitman_drawer_top_joint'))]

        ## --------- grasp and open handle - indigo
        set_camera_pose(camera_point=[1.3, 9, 2], target_point=[1, 9, 0])
        world.make_transparent('indigo_tmp')
        # world.open_joint_by_name('indigo_drawer_top_joint')
        world.put_in_space(cabbage, 'indigo_drawer_top')
        # goals = [("AtPosition", name_to_body('indigo_drawer_top_joint'),
        #           Position(name_to_body('indigo_drawer_top_joint'), 'max'))]
        # goals = [("OpenedJoint", name_to_body('indigo_drawer_top_joint'))]
        goals = [("On", name_to_body('cabbage'), name_to_body('indigo_tmp'))]
        # goals = [("OpenedJoint", name_to_body('indigo_drawer_top_joint')),
        #          ("On", name_to_body('cabbage'), name_to_body('indigo_tmp'))]
        # goals = ('test_grasps', 'cabbage')
        # goals = ('test_grasp_ik', 'cabbage')

        arm = 'left'
        drawer = name_to_body('indigo_drawer_top_joint')
        skeleton += [(k, arm, drawer) for k in pull_actions]
        skeleton += [(k, arm, cabbage) for k in pick_place_actions]

        ## --------- ideally, requires closing the drawer in order to put on
        # world.open_joint_by_name('indigo_drawer_top_joint')
        # world.put_on_surface(cabbage, 'hitman_tmp')
        # goals = [("On", name_to_body('cabbage'), name_to_body('indigo_tmp'))]

        ## --------- test moving objects from drawer to counter
        # goals = [("On", cabbage, name_to_body('hitman_tmp'))] ## successful

        # world.put_in_space(cabbage, 'hitman_drawer_top')
        # goals = [("On", cabbage, name_to_body('hitman_tmp'))] ## unsuccessful

        # world.open_joint_by_name('hitman_drawer_top_joint')
        # world.put_in_space(cabbage, 'hitman_drawer_top', learned=True)
        # goals = ('test_grasps', 'cabbage')
        # goals = ('test_grasp_ik', 'cabbage')
        # goals = [("Holding", "left", cabbage)] ## unsuccessful
        # goals = [("On", cabbage, name_to_body('hitman_tmp'))] ## unsuccessful

        ## --------- temporary
        # goals = [("HandleGrasped", 'left', name_to_body('hitman_drawer_top_joint'))]

        return goals, skeleton

    return test_nvidia_kitchen_domain(args, loader_fn, **kwargs)


def test_kitchen_doors(args, **kwargs):
    def loader_fn(world, **world_builder_args):
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
                'hitman_tmp': [],  ##  'Microwave'
                'indigo_tmp': ['BraiserLid'],  ## 'MeatTurkeyLeg', 'Toaster',
            },
        }
        load_full_kitchen(world, surfaces=surfaces, spaces=spaces, load_cabbage=False)

        """ add doors """
        supporter_to_doors = load_nvidia_kitchen_joints(world)
        supporters = ['sektion', 'shelf_bottom']  ## 'dagger', 'indigo_tmp'
        bottle = world.name_to_object('vinegarbottle')
        selected_id = 1  ## random.choice(range(len(counter_doors)))
        selected_door = None
        for supporter_id, supporter_name in enumerate(supporters):
            for door, pstn in supporter_to_doors[supporter_name]:
                world.open_joint(door, extent=pstn)
            if supporter_id == selected_id:
                place_in_nvidia_kitchen_space(bottle, supporter_name, interactive=False)
                selected_door = door
                if supporter_name == 'shelf_bottom':
                    set_camera_pose(camera_point=[2.717, 4.514, 2.521], target_point=[1, 5, 1])

        """ goals """
        arm = 'left'

        door = selected_door
        goals = ('test_handle_grasps', door)
        # goals = [("HandleGrasped", 'left', door)]
        # goals = [("AtPosition", door, Position(door, 'min'))]
        goals = [("OpenedJoint", door)]
        # goals = [("GraspedHandle", door)]

        movable = bottle.body
        goals = ('test_grasps', movable)
        goals = [("Holding", arm, movable)]
        # goals = [("OpenedJoint", door), ("Holding", arm, movable)]
        # goals = [("OpenedJoint", door)]

        skeleton = []
        # skeleton += [(k, arm, door) for k in pull_actions]
        # skeleton += [(k, arm, bottle) for k in pick_place_actions[:1]]

        ## --- for recording door open demo
        # world.open_joint_by_name('chewie_door_left_joint')

        return goals, skeleton

    return test_nvidia_kitchen_domain(args, loader_fn, initial_xy=(2, 5), **kwargs)


##########################################################################################


def test_kitchen_chicken_soup(args, **kwargs):
    def loader_fn(world, **world_builder_args):
        spaces = {
            'counter': {
                'sektion': [],  ## 'OilBottle',
                # 'dagger': [],  ## 'Salter'
                # 'hitman_drawer_top': [],  ## 'Pan'
                # 'hitman_drawer_bottom': [],
                'indigo_drawer_top': [],
                # 'indigo_drawer_bottom': [],
                # 'indigo_tmp': ['PotBody']
            },
            'dishwasher': {
                'upper_shelf': []
            }
        }
        surfaces = {
            'counter': {
                # 'front_left_stove': [],  ## 'Kettle'
                'front_right_stove': ['BraiserBody'],  ## 'PotBody',
                # 'back_left_stove': [],
                # 'back_right_stove': [],
                # 'range': [], ##
                'hitman_tmp': [],  ##  'Microwave'
                'indigo_tmp': ['BraiserLid'],  ## 'MeatTurkeyLeg', 'Toaster',
            },
        }
        custom_supports = {
            'cabbage': 'shelf_bottom',
            'fork': 'indigo_drawer_top'
        }
        goal_object = ['chicken-leg', 'cabbage', 'fork'][0]
        open_doors_for = [goal_object]

        load_full_kitchen(world, surfaces=surfaces, spaces=spaces, load_cabbage=False)
        movables, movable_to_doors = load_nvidia_kitchen_movables(world, open_doors_for=open_doors_for,
                                                                  custom_supports=custom_supports)
        counter = world.name_to_body('indigo_tmp')

        #########################################################################

        """ goals """
        arm = 'left'
        movable = movables[goal_object]
        objects = []
        door = world.name_to_body('fridge_door')
        drawer_joint = world.name_to_body('indigo_drawer_top_joint')
        drawer_link = world.name_to_body('indigo_drawer_top')
        dishwasher_space = world.name_to_body('upper_shelf')
        dishwasher_joint = world.name_to_body('dishwasher_door')

        # world.open_joint(drawer_joint)
        world.open_joint(dishwasher_joint)

        goals = ('test_grasps', movable)
        goals = [("Holding", arm, movable)]
        goals = [("On", movable, counter)]
        # goals = ('test_relpose_inside_gen', (movable, drawer_link))
        # goals = [("In", movable, drawer_link)]

        # goals = ('test_joint_closed', door)
        # goals = ('test_handle_grasps', door)
        # goals = [("OpenedJoint", drawer_joint)]
        # goals = [("ClosedJoint", door)]

        # goals = ('test_handle_grasps', drawer)
        # goals = [("OpenedJoint", drawer)]
        # goals = [("OpenedJoint", drawer), ("Holding", arm, movable)]

        #########################################################################

        objects += [movable]  ## Holding
        # objects += [drawer_joint]  ## OpenedJoint
        # objects += [drawer_link]  ## In (place)
        # objects += [drawer_joint, drawer_link]  ## In (place_rel)
        # objects += [dishwasher_space]  ## dishwasher_joint,
        # objects += [dishwasher_joint, dishwasher_space]

        #########################################################################

        skeleton = []
        # skeleton += [(k, arm, drawer) for k in pull_actions]
        # skeleton += [(k, arm, goal_object) for k in pick_place_actions[:1]]

        # skeleton += [(k, arm, drawer_joint) for k in pull_with_link_actions]
        # skeleton += [(k, arm, movable) for k in pick_place_rel_actions[:1]]
        # skeleton += [(k, arm, movable) for k in ['pick', 'place_to_supporter']]
        skeleton += [(k, arm, movable) for k in ['pick_from_supporter', 'place']]

        # skeleton += [(k, arm, movable) for k in pick_place_rel_actions]

        #########################################################################

        ## --- for recording door open demo
        # world.open_joint_by_name('chewie_door_left_joint')

        world.remove_bodies_from_planning(goals, exceptions=objects)

        return goals, skeleton

    return test_nvidia_kitchen_domain(args, loader_fn, initial_xy=(2, 5), **kwargs)

