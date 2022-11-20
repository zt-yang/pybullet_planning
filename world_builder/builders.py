import math
import random
import json
from os.path import join, abspath, basename
import sys

import pybullet as p
from .world import World, State
from .entities import Object, Region, Environment, Robot, Camera, Floor, Stove,\
    Surface, Moveable, Supporter, Steerable, Door
from .loaders import create_pr2_robot, load_rooms, load_cart, load_cart_regions, load_blocked_kitchen, \
    load_blocked_sink, load_blocked_stove, load_floor_plan, load_experiment_objects, load_pot_lid, load_basin_faucet, \
    load_kitchen_mechanism, create_gripper_robot, load_cabinet_test_scene, load_random_mini_kitchen_counter, \
    load_another_table, load_another_fridge_food, random_set_doors, ensure_robot_cfree, load_kitchen_mini_scene
from pybullet_tools.utils import Pose, Euler, PI, create_box, TAN, Point, set_camera_pose, link_from_name, \
    connect, enable_preview, draw_pose, unit_pose, set_all_static, wait_if_gui, reset_simulation, get_aabb

from pybullet_tools.bullet_utils import set_camera_target_body, set_camera_target_robot, draw_collision_shapes, \
    open_joint
from world_builder.world_generator import EXP_PATH


def set_time_seed():
    import numpy as np
    import time
    seed = int(time.time())
    np.random.seed(seed)
    random.seed(seed)
    return seed


def get_robot_builder(builder_name):
    if builder_name == 'build_fridge_domain_robot':
        return build_fridge_domain_robot
    elif builder_name == 'build_table_domain_robot':
        return build_table_domain_robot
    return None

############################################


def maybe_add_robot(world, template_dir=None):
    config_file = join(template_dir, 'planning_config.json')
    planning_config = json.load(open(config_file, 'r'))
    if 'robot_builder' not in planning_config:
        return
    custom_limits = planning_config['base_limits']
    robot_name = planning_config['robot_name']
    robot_builder = get_robot_builder(planning_config['robot_builder'])
    robot_builder(world, robot_name=robot_name, custom_limits=custom_limits)


def create_pybullet_world(args, builder,
                          verbose=False,
                          SAMPLING=False,
                          SAVE_LISDF=False,
                          DEPTH_IMAGES=False,
                          RESET=False,
                          SAVE_TESTCASE=False,
                          SAVE_RGB=False,
                          template_dir=None,
                          out_dir=None,
                          root_dir='..'):
    """ build a pybullet world with lisdf & pddl files into test_cases folder,
        given a text_case folder to copy the domain, stream, and config from """

    if template_dir is None:
        template_name = builder.__name__
    else:
        template_name = basename(template_dir)
    template_dir = abspath(join(root_dir, EXP_PATH, template_name))

    """ ============== initiate simulator ==================== """
    ## for viewing, not the size of depth image
    connect(use_gui=args.viewer, shadows=False, width=1980, height=1238)

    # set_camera_pose(camera_point=[2.5, 0., 3.5], target_point=[1., 0, 1.])
    if args.camera:
        enable_preview()
        p.configureDebugVisualizer(p.COV_ENABLE_SEGMENTATION_MARK_PREVIEW, False)
    draw_pose(unit_pose(), length=1.)

    """ ============== sample world configuration ==================== """

    world = World(time_step=args.time_step, camera=args.camera, segment=args.segment)
    maybe_add_robot(world, **args.config.robot)
    goal = builder(world, verbose=verbose, **args.config.world_builder)

    ## no gravity once simulation starts
    set_all_static()
    if verbose: world.summarize_all_objects()

    """ ============== save world configuration ==================== """
    file = None
    if SAVE_LISDF:   ## only lisdf files
        file = world.save_lisdf(verbose=verbose)

    if SAVE_TESTCASE:
        world.save_test_case(goal, out_dir, template_dir=template_dir,
                             save_rgb=SAVE_RGB, save_depth=DEPTH_IMAGES)

    if RESET:
        reset_simulation()
        return file
    return world, goal


def test_pick(world, w=.5, h=.9, mass=1):

    table = world.add_box(
        Object(create_box(w, w, h, color=(.75, .75, .75, 1)), category='supporter', name='table'),
        Pose(point=Point(x=2, y=0, z=h / 2)))

    cabbage = world.add_box(
        Moveable(create_box(.07, .07, .1, mass=mass, color=(0, 1, 0, 1)), name='cabbage'),
        Pose(point=Point(x=2, y=0, z=h + .1 / 2)))

    robot = create_pr2_robot(world, base_q=(0, 2, -PI / 2))

    return []


def test_exist_omelette(world, w=.5, h=.9, mass=1):

    fridge = world.add_box(
        Supporter(create_box(w, w, h, color=(.75, .75, .75, 1)), name='fridge'),
        Pose(point=Point(2, 0, h / 2)))

    egg = world.add_box(
        Moveable(create_box(.07, .07, .1, mass=mass, color=(1, 1, 0, 1)), category='egg', name='egg'),
        Pose(point=Point(2, -0.18, h + .1 / 2)))

    cabbage = world.add_box(
        Moveable(create_box(.07, .07, .1, mass=mass, color=(0, 1, 0, 1)), category='veggie', name='cabbage'),
        Pose(point=Point(2, 0, h + .1 / 2)))

    salter = world.add_box(
        Moveable(create_box(.07, .07, .1, mass=mass, color=(0, 0, 0, 1)), category='salter', name='salter'),
        Pose(point=Point(2, 0.18, h + .1 / 2)))

    plate = world.add_box(
        Moveable(create_box(.07, .07, .1, mass=mass, color=(0.4, 0.4, 0.4, 1)), category='plate', name='plate'),
        Pose(point=Point(2 + 0.18, 0, h + .1 / 2)))

    sink = world.add_box(
        Supporter(create_box(w, w, h, color=(.25, .25, .75, 1)), category='sink', name='sink'),
        Pose(point=Point(0, 2, h / 2)))

    stove = world.add_box(
        Supporter(create_box(w, w, h, color=(.75, .25, .25, 1)), category='stove', name='stove'),
        Pose(point=Point(0, -2, h / 2)))

    counter = world.add_box(
        Supporter(create_box(w, w, h, color=(.25, .75, .25, 1)), category='counter', name='counter'),
        Pose(point=Point(-2, 2, h / 2)))

    table = world.add_box(
        Supporter(create_box(w, w, h, color=(.75, .75, .25, 1)), category='table', name='table'),
        Pose(point=Point(-2, -2, h / 2)))

    robot = create_pr2_robot(world, base_q=(0, 0, 0))

    return []


def test_kitchen_oven(world, floorplan='counter.svg', verbose=False):

    set_camera_pose(camera_point=[3, 5, 3], target_point=[0, 6, 1])
    floor = load_floor_plan(world, plan_name=floorplan)
    # cabbage = load_experiment_objects(world)
    # floor = load_floor_plan(world, plan_name='fridge_v2.svg')
    egg = load_experiment_objects(world, CABBAGE_ONLY=True, name='eggblock', color=TAN)
    world.remove_object(floor)
    robot = create_pr2_robot(world, base_q=(1.79, 6, PI / 2 + PI / 2))

    name_to_body = world.name_to_body

    ## -- test the position of stove
    # world.put_on_surface(egg, 'front_right_stove')

    ## -- prepare the pot
    oven = world.name_to_object('oven')
    pot = world.name_to_object('braiserbody')
    world.put_on_surface(pot, 'front_right_stove')
    set_camera_target_body(oven, dx=1, dy=0, dz=1)
    bottom = world.add_object(Surface(pot.body, link_from_name(pot, 'braiser_bottom')))
    world.put_on_surface(egg, 'braiser_bottom')
    # world.remove_object(oven)
    # world.remove_object(pot)

    ## -- test draw body collision shapes
    # draw_collision_shapes(world.name_to_body('braiserlid'))
    # draw_collision_shapes(world.name_to_body('oven'))

    return []


def test_feg_pick(world, floorplan='counter.svg', verbose=True):

    """ ============== [State Space] Add world objects (Don't change) ================ """

    ## so that when loading a floor plan, no cabinet doors or drawers
    ## will be automatically added to planning objects. We'll add manually later
    world.set_skip_joints()

    ## add all objects, with dynamic object instances randomly drawn from assets/{category}/
    ## and collision free poses randomly drawn for objects. all joints are set to closed state
    pot, lid, turkey, counter, oil, vinegar = load_cabinet_test_scene(world, RANDOM_INSTANCE=True, verbose=verbose)

    """ ============== [Init] Add robot ==================== """

    ## you may change robot initial state
    custom_limits = {0: (0, 4), 1: (5, 12), 2: (0, 2)}  ## = {x: (x_min, x_max), y: ...}
    initial_q = [0.9, 8, 0.7, 0, -math.pi / 2, 0]  ## = {x, y, z, roll, pitch, yaw}
    robot = create_gripper_robot(world, custom_limits, initial_q=initial_q)

    """ ============== [Init] Modify initial object states ==================== """

    oven = world.name_to_body('oven')
    counter = world.name_to_body('indigo_tmp')
    left_door = world.name_to_body('chewie_door_left_joint')
    right_door = world.name_to_body('chewie_door_right_joint')
    right_cabinet = world.name_to_body('dagger')

    ## --- Randomization Strategy 1:
    ## open a particular door with an epsilon greedy strategy
    epsilon = 0.3
    for door in [left_door, right_door]:
        if random.random() < epsilon:
            open_joint(door[0], door[1], extent=random.random())
    # open_joint(left_door[0], left_door[1])
    # open_joint(right_door[0], right_door[1])

    ## --- Randomization Strategy 2:
    ## place the pot on one of the burners on the stove
    # set_pose(turkey, pose)
    # world.put_on_surface(lid, counter)
    # world.put_on_surface(pot, world.name_to_body('front_right'))

    ## --- Just Checking:
    ## this is not the depth camera, which is small, 256 by 256 pixels in size
    ## this is the camera for viewing on your screen, defined in relation to a body, or robot
    set_camera_target_body(lid, dx=2, dy=0, dz=0.5)
    # set_camera_target_body(right_door[0], link=right_door[1], dx=2, dy=0, dz=0.5)
    # wait_if_gui('proceed?')

    ## see object poses and joint positions that's occluded by closed joints
    # world.open_all_doors_drawers()

    """ ============== [Goal] Sample goals ==================== """

    ## --- Randomization Strategy 3:
    ## sample a movable and a surface
    body = [oil, vinegar, turkey][random.randint(0, 2)]
    surface = world.name_to_body('hitman_tmp')

    goal_template = [
        [('Holding', body)],
        [('On', body, surface)],
        [('On', body, surface), ('On', body, surface)]
    ]
    goal = random.choice(goal_template)

    return goal

############################################


def build_table_domain_robot(world, robot_name, custom_limits=None, initial_q=None):
    """ simplified cooking domain """
    if robot_name == 'feg':
        if custom_limits is None:
            custom_limits = {0: (0, 4), 1: (3, 12), 2: (0, 2)}
        if initial_q is None:
            initial_q = [0.9, 8, 0.7, 0, -math.pi / 2, 0]
        robot = create_gripper_robot(world, custom_limits, initial_q=initial_q)
    else:
        if custom_limits is None:
            custom_limits = ((0, 0), (8, 8))
        if initial_q is None:
            initial_q = (1.79, 6, PI / 2 + PI / 2)
        robot = create_pr2_robot(world, base_q=initial_q, custom_limits=custom_limits,
                                 USE_TORSO=False, DRAW_BASE_LIMITS=True)
    return robot


def build_fridge_domain_robot(world, robot_name, custom_limits=None):
    """ counter and fridge in the (6, 6) range """
    x, y = (5, 3)
    if robot_name == 'feg':
        if custom_limits is None:
            custom_limits = {0: (0, 6), 1: (0, 6), 2: (0, 2)}
        robot = create_gripper_robot(world, custom_limits, initial_q=[x, y, 0.7, 0, -math.pi / 2, 0])
        robot.set_spawn_range(((2.5, 2, 0.5), (3.8, 3.5, 1.9)))
    else:
        if custom_limits is None:
            custom_limits = ((0, 0, 0), (6, 6, 1.5))
        robot = create_pr2_robot(world, custom_limits=custom_limits, base_q=(x, y, PI / 2 + PI / 2))
        robot.set_spawn_range(((4.2, 2, 0.5), (5, 3.5, 1.9)))
    return robot


############################################


def test_one_fridge(world, movable_category='food', verbose=True, **kwargs):
    #set_time_seed()
    sample_one_fridge_scene(world, movable_category, verbose=verbose, **kwargs)
    goal = sample_one_fridge_goal(world)
    return goal


def sample_one_fridge_scene(world, movable_category='food', verbose=True, open_doors=True, **kwargs):

    ## later we may want to automatically add irrelevant objects and joints
    world.set_skip_joints()

    """ ============== Sample an initial conf for robot ==================== """
    world.robot.randomly_spawn()

    """ ============== Add world objects ================ """
    minifridge_doors = load_random_mini_kitchen_counter(world, movable_category, **kwargs)

    """ ============== Change joint positions ================ """
    ## only after all objects have been placed inside
    if open_doors:
        random_set_doors(minifridge_doors, epsilon=0.5)

    """ ============== Check collisions ================ """
    ensure_robot_cfree(world, verbose=verbose)

    return minifridge_doors


def sample_one_fridge_goal(world, movable_category='food'):
    cabbage = world.cat_to_bodies(movable_category)[0]  ## world.name_to_body('cabbage')
    fridge = world.name_to_body('fridgestorage')
    counter = world.name_to_body('counter')

    arm = world.robot.arms[0]
    goal_candidates = [
        [('Holding', arm, cabbage)],
        # [('On', cabbage, counter)],
        # [('In', cabbage, fridge)],
    ]
    return random.choice(goal_candidates)


############################################


def test_fridge_table(world, movable_category='food', verbose=True, **kwargs):
    #set_time_seed()
    sample_fridge_table_scene(world, movable_category, verbose, **kwargs)
    goal = sample_fridge_table_goal(world, movable_category)
    return goal


def sample_fridge_table_scene(world, movable_category='food', verbose=True, **kwargs):
    sample_one_fridge_scene(world, movable_category, verbose=verbose, **kwargs)
    load_another_table(world)


def sample_fridge_table_goal(world, movable_category='food'):
    cabbage = world.cat_to_objects(movable_category)[0]  ## world.name_to_body('cabbage')
    fridge = world.name_to_body('fridgestorage')
    counter = world.name_to_body('counter')
    table = world.name_to_object('table')

    arm = world.robot.arms[0]

    if random.random() < 0.5: ## 0.5
        goal_candidates = [
            [('Holding', arm, cabbage.body)],
            # [('On', cabbage, table.body)],
        ]
    else:
        table.place_obj(cabbage)
        table.attach_obj(cabbage)
        goal_candidates = [
            # [('Holding', arm, cabbage.body)],
            [('In', cabbage.body, fridge)],
        ]

    return random.choice(goal_candidates)


############################################


def test_fridges_tables(world, movable_category='food', verbose=True, **kwargs):
    sample_fridges_tables_scene(world, movable_category, verbose=verbose, **kwargs)
    goal = sample_fridges_tables_goal(world, movable_category)
    return goal


def sample_fridges_tables_scene(world, movable_category='food', verbose=True, **kwargs):
    epsilon = 0.45
    minifridge_doors = sample_one_fridge_scene(world, movable_category, open_doors=False, **kwargs)
    load_another_table(world, four_ways=False)
    load_another_fridge_food(world, movable_category, epsilon=epsilon, **kwargs)
    random_set_doors(minifridge_doors, epsilon=epsilon, extent_max=0.5)
    ensure_robot_cfree(world, verbose=verbose)


def sample_fridges_tables_goal(world, movable_category='food'):
    food = random.choice(world.cat_to_objects(movable_category))
    spaces = world.cat_to_objects('space')
    other = [s for s in spaces if s != food.suppoting_surface][0]

    arm = world.robot.arms[0]

    ## the goal will be to pick one object and put in the other fridge
    goal_candidates = [
        # [('Holding', arm, food)],
        [('In', food.pybullet_name, other.pybullet_name)],
    ]

    return random.choice(goal_candidates)


############################################


def test_fridges_tables_conjunctive(world, movable_category='food', verbose=True, **kwargs):
    sample_fridges_tables_scene(world, movable_category, verbose=verbose, **kwargs)
    goal = sample_conjunctive_fridges_tables_goal(world, movable_category)
    return goal


def sample_conjunctive_fridges_tables_goal(world, movable_category='food'):
    foods = world.cat_to_objects(movable_category)
    random.shuffle(foods)
    arm = world.robot.arms[0]
    spaces = world.cat_to_objects('space')

    cases = ['in2', 'in1hold1']
    open_surfaces = get_open_surfaces(world)
    if len(open_surfaces) > 0:
        cases.extend(['in1on1'])

    ## because of data imbalance
    case = random.choice(cases)
    # case = 'in2'
    # if len(open_surfaces) > 0:
    #     case = 'in1on1'

    goals = []
    if case == 'in2':
        goals.append(get_goal_in(foods[0], spaces=spaces))
        goals.append(get_goal_in(foods[1], spaces=spaces))
    elif case == 'in1hold1':
        goals.append(get_goal_in(foods[0], spaces=spaces))
        goals.append(('Holding', arm, foods[1]))
    elif case == 'in1on1':
        goals.append(get_goal_in(foods[0], spaces=spaces))
        goals.append(get_goal_on(foods[1], open_surfaces))

    return goals


############################################


def test_three_fridges_tables(world, movable_category='food', **kwargs):
    sample_three_fridges_tables_scene(world, movable_category, **kwargs)
    goal = sample_three_fridges_tables_goal(world, movable_category)
    return goal


def sample_three_fridges_tables_scene(world, movable_category='food', verbose=True, **kwargs):
    epsilon = 0.45
    sample_one_fridge_scene(world, verbose=verbose, open_doors=False, **kwargs)

    load_another_table(world, four_ways=False, table_name='table')
    load_another_table(world, four_ways=False, table_name='station')

    load_another_fridge_food(world, verbose=verbose, table_name='table',
                             fridge_name='cabinet', epsilon=epsilon, **kwargs)
    load_another_fridge_food(world, verbose=verbose, table_name='station',
                             fridge_name='sterilizer', epsilon=epsilon, **kwargs)

    random_set_doors(world.cat_to_bodies('door'), extent_max=0.5, epsilon=epsilon)
    ensure_robot_cfree(world, verbose=verbose)


def get_goal_in(food, spaces=None, world=None):
    """ random sample another fridge as destination """
    if spaces is None:
        spaces = world.cat_to_bodies('space')
    other = random.choice([s for s in spaces if s != food.supporting_surface])
    return ('In', food.pybullet_name, other.pybullet_name)


def get_open_surfaces(world):
    return [ind.pybullet_name for ind in world.cat_to_objects('supporter') if has_open_surface(ind)]


def has_open_surface(obj):
    aabb_fridge = get_aabb(obj.supported_objects[0].body)
    aabb_table = get_aabb(obj.body)
    return abs(aabb_fridge.lower[1] - aabb_table.lower[1]) > 0.2 or \
           abs(aabb_fridge.upper[1] - aabb_table.upper[1]) > 0.2


def get_goal_on(food, open_surfaces=None, world=None):
    """ random sample an open table as destination """
    if open_surfaces is None:
        open_surfaces = get_open_surfaces(world)
    table = random.choice(open_surfaces)
    return ('On', food, table)


def sample_three_fridges_tables_goal(world, placement, movable_category='food'):
    arm = world.robot.arms[0]
    foods = world.cat_to_objects(movable_category)
    spaces = world.cat_to_objects('space')
    random.shuffle(foods)
    cases = ['in1', 'hold1', 'in2', 'in2hold1', 'in3']

    open_surfaces = get_open_surfaces(world)
    if len(open_surfaces) > 0:
        cases.extend(['on1', 'in2on1'])

    goals = []
    case = random.choice(cases)
    if case == 'in1':
        goals.append(get_goal_in(foods[0], spaces=spaces))
    elif case == 'hold1':
        goals.append(('Holding', arm, foods[0]))
    elif case == 'on1':
        goals.append(get_goal_on(foods[0], open_surfaces))
    elif case == 'in2':
        goals.append(get_goal_in(foods[0], spaces=spaces))
        goals.append(get_goal_in(foods[1], spaces=spaces))
    elif case == 'in2hold1':
        goals.append(get_goal_in(foods[0], spaces=spaces))
        goals.append(get_goal_in(foods[1], spaces=spaces))
        goals.append(('Holding', arm, foods[2]))
    elif case == 'in2on1':
        goals.append(get_goal_in(foods[0], spaces=spaces))
        goals.append(get_goal_in(foods[1], spaces=spaces))
        goals.append(get_goal_on(foods[2], open_surfaces))
    elif case == 'in3':
        goals.append(get_goal_in(foods[0], spaces=spaces))
        goals.append(get_goal_in(foods[1], spaces=spaces))
        goals.append(get_goal_in(foods[2], spaces=spaces))

    return goals


##########################################################################################


def test_feg_kitchen_mini(world, **kwargs):
    sample_kitchen_mini_scene(world, **kwargs)
    goal = sample_kitchen_mini_goal(world)
    return goal


def sample_kitchen_mini_scene(world, **kwargs):
    """ implemented by Felix for the FEG gripper """
    load_kitchen_mini_scene(world)


def sample_kitchen_mini_goal(world):
    bottle = random.choice(world.cat_to_bodies('bottle'))

    hand = world.robot.arms[0]
    goal_candidates = [
        [('Holding', hand, bottle)],
        # [('On', cabbage, counter)],
        # [('In', cabbage, fridge)],
    ]
    return random.choice(goal_candidates)


##########################################################################################


def test_kitchen_clean(world, **kwargs):
    sample_kitchen_scene(world, **kwargs)
    goal = sample_kitchen_mini_goal(world)
    return goal
