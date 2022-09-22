import math
import random
from os.path import join

from .world import World, State
from .entities import Object, Region, Environment, Robot, Camera, Floor, Stove,\
    Surface, Moveable, Supporter, Steerable, Door
from .loaders import create_pr2_robot, load_rooms, load_cart, load_cart_regions, load_blocked_kitchen, \
    load_blocked_sink, load_blocked_stove, load_floor_plan, load_experiment_objects, load_pot_lid, load_basin_faucet, \
    load_kitchen_mechanism, create_gripper_robot, load_cabinet_test_scene, load_random_mini_kitchen_counter, \
    load_another_table, load_another_fridge_food, random_set_doors, ensure_robot_cfree
from pybullet_tools.utils import Pose, Euler, PI, create_box, TAN, Point, set_camera_pose, link_from_name
from pybullet_tools.bullet_utils import set_camera_target_body, set_camera_target_robot, draw_collision_shapes, \
    open_joint, collided


def test_pick(world, w=.5, h=.9, mass=1):

    table = world.add_box(
        Object(create_box(w, w, h, color=(.75, .75, .75, 1)), category='supporter', name='table'),
        Pose(point=Point(x=2, y=0, z=h / 2)))

    cabbage = world.add_box(
        Moveable(create_box(.07, .07, .1, mass=mass, color=(0, 1, 0, 1)), name='cabbage'),
        Pose(point=Point(x=2, y=0, z=h + .1 / 2)))

    robot = create_pr2_robot(world, base_q=(0, 2, -PI / 2))

    return None, []


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

    return None, []

def test_kitchen_oven(world, floorplan='counter.svg'):

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

    return floorplan, []

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

    """ ============== [Output] Save depth image ==================== """
    ## you may purturb the camera pose ((point), (quaternian))
    camera_pose = ((3.7, 8, 1.3), (0.5, 0.5, -0.5, -0.5))
    world.add_camera(camera_pose)

    return floorplan, goal

############################################

def set_time_seed():
    import numpy as np
    import time
    seed = int(time.time())
    np.random.seed(seed)
    random.seed(seed)
    return seed


def test_one_fridge(world, verbose=True):
    #set_time_seed()
    sample_one_fridge_scene(world, verbose)
    goal = sample_one_fridge_goal(world)
    return None, goal


def sample_one_fridge_scene(world, verbose=True, open_doors=True):

    ## later we may want to automatically add irrelevant objects and joints
    world.set_skip_joints()

    """ ============== Sample an initial conf for robot ==================== """
    world.robot.randomly_spawn()

    """ ============== Add world objects ================ """
    minifridge_doors = load_random_mini_kitchen_counter(world)

    """ ============== Change joint positions ================ """
    ## only after all objects have been placed inside
    if open_doors:
        random_set_doors(minifridge_doors, epsilon=0.2)

    """ ============== Check collisions ================ """
    ensure_robot_cfree(world, verbose=verbose)

    return minifridge_doors


def sample_one_fridge_goal(world):
    cabbage = world.cat_to_bodies('food')[0]  ## world.name_to_body('cabbage')
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


def test_fridge_table(world, verbose=True):
    #set_time_seed()
    sample_fridge_table_scene(world, verbose)
    goal = sample_fridge_table_goal(world)
    return None, goal


def sample_fridge_table_scene(world, verbose=True):
    sample_one_fridge_scene(world, verbose=verbose)
    load_another_table(world)


def sample_fridge_table_goal(world):
    cabbage = world.cat_to_objects('food')[0]  ## world.name_to_body('cabbage')
    fridge = world.name_to_body('fridgestorage')
    counter = world.name_to_body('counter')
    table = world.name_to_object('table')

    arm = world.robot.arms[0]

    if random.random() < 0.5 and False: ## 0.5
        goal_candidates = [
            [('Holding', arm, cabbage.body)],
            [('On', cabbage, table.body)],
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


def test_fridges_tables(world, verbose=True):
    placement = sample_fridges_tables_scene(world)
    goal = sample_fridges_tables_goal(world, placement)
    return None, goal


def sample_fridges_tables_scene(world, verbose=True):
    minifridge_doors = sample_one_fridge_scene(world, open_doors=False)
    load_another_table(world, four_ways=False)
    placement = load_another_fridge_food(world)
    random_set_doors(minifridge_doors, epsilon=0.1, extent_max=0.5)
    ensure_robot_cfree(world, verbose=verbose)
    return placement


def sample_fridges_tables_goal(world, placement):
    foods = world.cat_to_bodies('food')
    if random.random() < 0.5:
        food = foods[0]
    else:
        food = foods[1]
    # food = random.choice(world.cat_to_bodies('food'))
    spaces = world.cat_to_bodies('space')
    other = [s for s in spaces if s != placement[food]][0]

    arm = world.robot.arms[0]

    ## the goal will be to pick one object and put in the other fridge
    goal_candidates = [
        [('Holding', arm, food)],
        [('In', food, other)],
    ]

    return random.choice(goal_candidates)
