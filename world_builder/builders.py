import math
import random
from os.path import join, abspath, basename
import os

import pybullet as p
from world_builder.world import World
from world_builder.entities import Object, Surface, Movable, Supporter
from world_builder.loaders import load_floor_plan, load_experiment_objects
from world_builder.loaders_partnet_kitchen import load_random_mini_kitchen_counter, \
    load_another_table, load_another_fridge_food, random_set_doors, ensure_robot_cfree, load_kitchen_mini_scene, \
    sample_full_kitchen
from world_builder.loaders_nvidia_kitchen import load_cabinet_test_scene
from world_builder.world_utils import get_domain_constants

from robot_builder.robot_builders import get_robot_builder, create_gripper_robot, create_pr2_robot

from pybullet_tools.utils import Pose, PI, create_box, TAN, Point, set_camera_pose, link_from_name, \
    connect, enable_preview, draw_pose, unit_pose, set_all_static, get_aabb
from pybullet_tools.bullet_utils import open_joint, get_datetime
from pybullet_tools.camera_utils import set_camera_target_body


def set_time_seed():
    import numpy as np
    import time
    seed = int(time.time())
    np.random.seed(seed)
    random.seed(seed)
    return seed


def initialize_pybullet(config):
    ## for viewing, not the size of depth image
    connect(use_gui=config.sim.viewer, shadows=False, width=1980, height=1238)

    # set_camera_pose(camera_point=[2.5, 0., 3.5], target_point=[1., 0, 1.])
    if config.sim.camera:
        enable_preview()
        p.configureDebugVisualizer(p.COV_ENABLE_SEGMENTATION_MARK_PREVIEW, False)
    draw_pose(unit_pose(), length=1.)


def get_world_builder(builder_name):
    from inspect import getmembers, isfunction
    import sys
    import nsplan_tools.nsplan_builders as nsplan_builders
    current_module = sys.modules[__name__]
    result = [a[1] for a in getmembers(current_module) if isfunction(a[1]) and a[0] == builder_name]
    result += [a[1] for a in getmembers(nsplan_builders) if isfunction(a[1]) and a[0] == builder_name]
    if len(result) == 0:
        raise ValueError('Builder {} not found'.format(builder_name))
    return result[0]


def sample_world_and_goal(config, builder=None):
    """ build a pybullet world with lisdf & pddl files into test_cases folder,
        given a text_case folder to copy the domain, stream, and config from """

    """ ============== initiate simulator ==================== """
    initialize_pybullet(config)
    constants = get_domain_constants(config.planner.domain_pddl)
    world = World(time_step=config.sim.time_step, segment=config.sim.segment, constants=constants)

    """ ============== load robot ==================== """
    robot_builder = get_robot_builder(config.robot.builder_name)
    robot = robot_builder(world, config.robot.robot_name, **config.robot.builder_kwargs)

    """ ============== sample world and goal ==================== """
    if builder is None:
        builder = get_world_builder(config.world.builder_name)

    verbose = config.verbose
    if 'verbose' in config.world.builder_kwargs:
        verbose = verbose and config.world.builder_kwargs['verbose']
        config.world.builder_kwargs.pop('verbose')
    goal = builder(world, verbose=verbose, **config.world.builder_kwargs)

    ## no gravity once simulation starts
    set_all_static()
    if config.verbose: world.summarize_all_objects()

    return world, goal


def save_world_problem(world, goal, config, save_lisdf=True, save_problem=True):

    ## if basename is already configured to be an index
    out_dir = abspath(config.data.out_dir)
    if not basename(config.data.out_dir).isdigit():
        out_dir = join(out_dir, get_datetime(seconds=True))
        config.data.out_dir = out_dir
    os.makedirs(out_dir, exist_ok=True)

    if save_lisdf:   ## only lisdf files
        world.save_lisdf(out_dir, verbose=config.verbose)

    if save_problem:   ## save generated world conf, sampled problem and planning config
        world.planning_config['camera_poses'] = config.data.cameras
        world.save_test_case(out_dir, goal=goal, **config.data.images)

    return out_dir


def test_exist_omelette(world, w=.5, h=.9, mass=1):

    fridge = world.add_box(
        Supporter(create_box(w, w, h, color=(.75, .75, .75, 1)), name='fridge'),
        Pose(point=Point(2, 0, h / 2)))

    egg = world.add_box(
        Movable(create_box(.07, .07, .1, mass=mass, color=(1, 1, 0, 1)), category='egg', name='egg'),
        Pose(point=Point(2, -0.18, h + .1 / 2)))

    cabbage = world.add_box(
        Movable(create_box(.07, .07, .1, mass=mass, color=(0, 1, 0, 1)), category='veggie', name='cabbage'),
        Pose(point=Point(2, 0, h + .1 / 2)))

    salter = world.add_box(
        Movable(create_box(.07, .07, .1, mass=mass, color=(0, 0, 0, 1)), category='salter', name='salter'),
        Pose(point=Point(2, 0.18, h + .1 / 2)))

    plate = world.add_box(
        Movable(create_box(.07, .07, .1, mass=mass, color=(0.4, 0.4, 0.4, 1)), category='plate', name='plate'),
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
    pot, lid, turkey, counter, oil, vinegar = load_cabinet_test_scene(world, random_instance=True, verbose=verbose)

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
    surface = world.name_to_body('hitman_countertop')

    goal_template = [
        [('Holding', body)],
        [('On', body, surface)],
        [('On', body, surface), ('On', body, surface)]
    ]
    goal = random.choice(goal_template)

    return goal


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
    other = [s for s in spaces if s != food.supporting_surface][0]

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
    goals = random.choice(goal_candidates)

    world.add_to_cat(bottle, 'movable')
    world.remove_bodies_from_planning(goals=goals)

    return goals


##########################################################################################


def test_kitchen_full(world, verbose=True, **kwargs):
    world.set_skip_joints()
    world.note = 1

    sample_full_kitchen(world, verbose=verbose, pause=False)
    goal = sample_kitchen_full_goal(world, **kwargs)
    return goal


def sample_kitchen_full_goal(world, movable_categories=['edible', 'bottle'],
                             goal_predicates=None):
    objects = []

    movable_candidates = []
    for category in movable_categories:
        movable_candidates += world.cat_to_bodies(category)
    movable = random.choice(movable_candidates)

    counter = world.name_to_body('counter#1')
    fridge = world.name_to_object('minifridge')
    fridge_space = world.name_to_body(f'minifridge::storage')
    fridge_door = fridge.doors[0]
    # objects += [fridge_door]

    hand = world.robot.arms[0]
    goals_candidates = [
        [('Holding', hand, movable)],
        # [('On', movable, counter)],
        [('In', movable, fridge_space)],
        [('OpenedJoint', fridge_door)],
    ]
    goals = []
    while len(goals) == 0:
        goals = random.choice(goals_candidates)
        if goal_predicates is not None:
            goals = [g for g in goals if g[0].lower() in goal_predicates]

    """ optional: modify the world to work with the goal """
    for goal in goals:
        if goal[0] == 'In' and goal[-1] == fridge_space:
            for door in fridge.doors:
                world.open_joint(door[0], door[1], hide_door=True)

        if goal[0] == 'OpenedJoint':
            door = goal[-1]
            world.close_joint(door[0], door[1])

        if goal[0] in ['Holding']:
            world.add_to_cat(goal[-1], 'movable')

        if goal[0] in ['In', 'On']:
            world.add_to_cat(goal[1], 'movable')

    world.remove_bodies_from_planning(goals=goals, exceptions=objects)

    return goals

