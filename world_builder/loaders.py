import random

import numpy as np
import math
import os
import string

import pybullet as p
from .utils import LIGHT_GREY, read_xml, load_asset, FLOOR_HEIGHT, WALL_HEIGHT, \
    find_point_for_single_push, ASSET_PATH

from .world import World, State
from .entities import Object, Region, Environment, Robot, Camera, Floor, Stove, Supporter,\
    Surface, Moveable, Space, Steerable
from .robots import PR2Robot, FEGripper

import sys
from pybullet_tools.pr2_utils import draw_viewcone, get_viewcone, get_group_conf, set_group_conf, get_other_arm, \
    get_carry_conf, set_arm_conf, open_arm, close_arm, arm_conf, REST_LEFT_ARM
from pybullet_tools.pr2_problems import create_pr2, create_table, create_floor
from pybullet_tools.pr2_primitives import get_base_custom_limits
from pybullet_tools.utils import apply_alpha, get_camera_matrix, LockRenderer, HideOutput, load_model, TURTLEBOT_URDF, \
    set_all_color, dump_body, draw_base_limits, multiply, Pose, Euler, PI, draw_pose, unit_pose, create_box, TAN, Point, \
    GREEN, create_cylinder, INF, BLACK, WHITE, RGBA, GREY, YELLOW, BLUE, BROWN, stable_z, set_point, set_camera_pose, \
    set_all_static, get_model_info, load_pybullet, remove_body, get_aabb, set_pose, wait_if_gui, get_joint_names, \
    get_min_limit, get_max_limit, set_joint_position, set_joint_position, get_joints, get_joint_info, get_moving_links, \
    get_pose, get_joint_position, enable_gravity, enable_real_time, get_links, set_color, dump_link, draw_link_name, \
    get_link_pose, get_aabb, get_link_name, sample_aabb, aabb_contains_aabb, aabb2d_from_aabb, sample_placement, \
    aabb_overlap, get_links, get_collision_data, get_visual_data, link_from_name, body_collision, get_closest_points, \
    load_pybullet, FLOOR_URDF, get_aabb_center, AABB, INF, clip, aabb_union, get_aabb_center, Pose, Euler, get_box_geometry, \
    get_aabb_extent, multiply, GREY, create_shape_array, create_body, STATIC_MASS, set_renderer, quat_from_euler, \
    get_joint_name, wait_for_user, draw_aabb, get_bodies
from pybullet_tools.bullet_utils import place_body, add_body, Pose2d, nice, OBJ_YAWS, \
    sample_obj_on_body_link_surface, sample_obj_in_body_link_space, set_camera_target_body, \
    open_joint, close_joint, set_camera_target_robot, summarize_joints, get_partnet_doors, get_partnet_spaces, \
    set_pr2_ready, BASE_LINK, BASE_RESOLUTIONS, BASE_VELOCITIES, BASE_JOINTS, draw_base_limits, \
    BASE_LIMITS, CAMERA_FRAME, CAMERA_MATRIX, EYE_FRAME, collided

OBJ = '?obj'

BASE_VELOCITIES = np.array([1., 1., math.radians(180)]) / 1. # per second
BASE_RESOLUTIONS = np.array([0.05, 0.05, math.radians(10)]) # from examples.pybullet.namo.stream import BASE_RESOLUTIONS
zero_limits = 0 * np.ones(2)
half_limits = 12 * np.ones(2)
BASE_LIMITS = (-half_limits, +half_limits) ## (zero_limits, +half_limits) ##
BASE_LIMITS = ((-1, 3), (6, 13))

CAMERA_FRAME = 'high_def_optical_frame'
EYE_FRAME = 'wide_stereo_gazebo_r_stereo_camera_frame'
CAMERA_MATRIX = get_camera_matrix(width=640, height=480, fx=525., fy=525.) # 319.5, 239.5 | 772.55, 772.5

LOAD_MOVEABLES = True
GRASPABLES = ['BraiserLid', 'Egg', 'VeggieCabbage', 'MeatTurkeyLeg', 'VeggieGreenPepper', 'VeggieArtichoke',
                        'VeggieTomato',
                        'VeggieZucchini', 'VeggiePotato', 'VeggieCauliflower',
                        'MeatChicken']
GRASPABLES = [o.lower() for o in GRASPABLES]

#######################################################


def set_pr2_ready(pr2, arm='left', grasp_type='top', DUAL_ARM=False):
    other_arm = get_other_arm(arm)
    if not DUAL_ARM:
        initial_conf = get_carry_conf(arm, grasp_type)
        set_arm_conf(pr2, arm, initial_conf)
        open_arm(pr2, arm)
        set_arm_conf(pr2, other_arm, arm_conf(other_arm, REST_LEFT_ARM))
        close_arm(pr2, other_arm)
    else:
        for a in [arm, other_arm]:
            initial_conf = get_carry_conf(a, grasp_type)
            set_arm_conf(pr2, a, initial_conf)
            open_arm(pr2, a)


def create_pr2_robot(world, base_q=(0,0,0),
                     DUAL_ARM=False, USE_TORSO=True,
                     custom_limits=BASE_LIMITS,
                     resolutions=BASE_RESOLUTIONS,
                     max_velocities=BASE_VELOCITIES, robot=None):

    if robot is None:
        with LockRenderer(lock=True):
            with HideOutput(enable=True):
                robot = create_pr2()
                set_pr2_ready(robot, DUAL_ARM=DUAL_ARM)
        if len(base_q) == 3:
            set_group_conf(robot, 'base', base_q)
        elif len(base_q) == 4:
            set_group_conf(robot, 'base-torso', base_q)

    with np.errstate(divide='ignore'):
        weights = np.reciprocal(resolutions)

    if isinstance(custom_limits, dict):
        custom_limits = np.asarray(list(custom_limits.values())).T.tolist()

    draw_base_limits(custom_limits)
    robot = PR2Robot(robot, base_link=BASE_LINK, joints=BASE_JOINTS,
                     DUAL_ARM=DUAL_ARM, USE_TORSO=USE_TORSO,
                     custom_limits=get_base_custom_limits(robot, custom_limits),
                     resolutions=resolutions, weights=weights)
    world.add_robot(robot, max_velocities=max_velocities)

    # print('initial base conf', get_group_conf(robot, 'base'))
    # set_camera_target_robot(robot, FRONT=True)

    camera = Camera(robot, camera_frame=CAMERA_FRAME, camera_matrix=CAMERA_MATRIX, max_depth=2.5, draw_frame=EYE_FRAME)
    robot.cameras.append(camera)

    ## don't show depth and segmentation data yet
    # if args.camera: robot.cameras[-1].get_image(segment=args.segment)

    return robot

from pybullet_tools.flying_gripper_utils import create_fe_gripper, plan_se3_motion, Problem, \
    get_free_motion_gen, set_gripper_positions, get_se3_joints, set_gripper_positions, \
    set_se3_conf ## se3_from_pose,


def create_gripper_robot(world, custom_limits, initial_q=(0, 0, 0, 0, 0, 0), robot=None):
    from pybullet_tools.flying_gripper_utils import BASE_RESOLUTIONS, BASE_VELOCITIES, BASE_LINK

    if robot is None:
        with LockRenderer(lock=True):
            with HideOutput(enable=True):
                robot = create_fe_gripper()
        set_se3_conf(robot, initial_q)

    with np.errstate(divide='ignore'):
        weights = np.reciprocal(BASE_RESOLUTIONS)
    robot = FEGripper(robot, base_link=BASE_LINK, joints=get_se3_joints(robot),
                  custom_limits=custom_limits, resolutions=BASE_RESOLUTIONS, weights=weights)
    world.add_robot(robot, max_velocities=BASE_VELOCITIES)

    return robot


######################################################

def Box(x1=0., x2=0.,
        y1=0., y2=0.,
        z1=0., z2=0.):
    return AABB(lower=[x1, y1, z1],
                upper=[x2, y2, z2])

def get_room_boxes(width=1., length=None, height=None, thickness=0.01, gap=INF, yaw=0.):
    # TODO: interval of beams
    if length is None:
        length = width
    if height is None:
        height = width
    gap = clip(gap, min_value=0., max_value=length)
    y_nongap = (length - gap) / 2.
    assert (width >= thickness) and (length >= thickness)
    walls = [
        Box(x1=0, x2=width,
            y1=0, y2=0 + thickness,
            z1=0, z2=height),
        Box(x1=0, x2=width,
            y1=length - thickness, y2=length,
            z1=0, z2=height),
        Box(x1=0, x2=0 + thickness,
            y1=0, y2=length,
            z1=0, z2=height),
        # Box(x1=width - thickness, x2=width,
        #     y1=0, y2=length,
        #     z1=0, z2=height),

        Box(x1=width - thickness, x2=width,
            y1=0, y2=0 + y_nongap,
            z1=0, z2=height),
        Box(x1=width - thickness, x2=width,
            y1=length - y_nongap, y2=length,
            z1=0, z2=height),
    ]
    aabb = aabb_union(walls)
    lower, _ = aabb
    center = get_aabb_center(aabb)
    base_center = np.append(center[:2], lower[2:])
    origin = Pose(euler=Euler(yaw=yaw)) # -base_center,
    # TODO: transform OOBB into a centered form
    return [(get_box_geometry(*get_aabb_extent(box)),
             multiply(origin, Pose(get_aabb_center(box)-base_center))) for box in walls]


def create_room(color=GREY, *args, **kwargs):
    shapes = get_room_boxes(*args, **kwargs)
    geoms, poses = zip(*shapes)
    colors = len(shapes)*[color]
    collision_id, visual_id = create_shape_array(geoms, poses, colors)
    return create_body(collision_id, visual_id, mass=STATIC_MASS)

#######################################################

SHAPE_INDICES = {
    # TODO: add doorways
    # TODO: include the floor
    'fence': [[-1, +1], [-1, +1], []],
    '+x_walls': [[-1], [-1, +1], []],
    '-x_walls': [[+1], [-1, +1], []],

    'x_walls': [[-1, +1], [], []],
    'y_walls': [[], [-1, +1], []],
}


def create_hollow_shapes(indices, width=1., length=1., height=1., thickness=0.01):
    assert len(indices) == 3
    dims = [width, length, height]
    center = [0., 0., height/2.]
    coordinates = string.ascii_lowercase[-len(dims):]

    # TODO: no way to programmatically set the name of the geoms or links
    # TODO: rigid links version of this
    shapes = []
    for index, signs in enumerate(indices):
        link_dims = np.array(dims)
        link_dims[index] = thickness
        for sign in sorted(signs):
            #name = '{:+d}'.format(sign)
            name = '{}{}'.format('-' if sign < 0 else '+', coordinates[index])
            geom = get_box_geometry(*link_dims)
            link_center = np.array(center)
            link_center[index] += sign*(dims[index] - thickness)/2.
            pose = Pose(point=link_center) # TODO: can always rotate
            shapes.append((name, geom, pose))
    return shapes


def create_hollow(category, color=GREY, *args, **kwargs):
    indices = SHAPE_INDICES[category]
    shapes = create_hollow_shapes(indices, *args, **kwargs)
    _, geoms, poses = zip(*shapes)
    colors = len(shapes)*[color]
    collision_id, visual_id = create_shape_array(geoms, poses, colors)
    return create_body(collision_id, visual_id, mass=STATIC_MASS)

#######################################################

def create_short_table(height=0.1, **kwargs):
    # create_floor
    return create_table(height=height, **kwargs)

def create_rotating_door(**kwargs):
    raise NotImplementedError()

def create_sliding_door(**kwargs):
    raise NotImplementedError()

def create_drawer(**kwargs):
    raise NotImplementedError()


#######################################################

def load_experiment_objects(world, w=.5, h=.9, wb=.07, hb=.1, mass=1, EXIST_PLATE=True,
                            CABBAGE_ONLY=True, name='cabbage', color=(0, 1, 0, 1)):

    if not CABBAGE_ONLY:
        table = world.add_object(
            Object(create_box(w, w, h, color=(.75, .75, .75, 1)), category='supporter', name='table'),
            Pose(point=Point(x=4, y=6, z=h / 2)))

    cabbage = world.add_object(
        Moveable(create_box(wb, wb, hb, mass=mass, color=color), name=name),
        Pose(point=Point(x=4, y=6, z=h + .1/2)))

    if not CABBAGE_ONLY:
        sink = world.add_object(
            Object(create_box(w/4, w/4, h, color=(.25, .25, .75, 1)), category='supporter', name='sink'),
            Pose(point=Point(x=2, y=5, z=h / 2)))
        if EXIST_PLATE:
            plate = world.add_object(
                Moveable(create_box(.07, .07, .1, mass=mass, color=(1, 1, 1, 1)), name='plate'),
                Pose(point=Point(x=2, y=5, z=h + .1 / 2)))

    return cabbage

def studio(args):
    """
    for testing fridge: plan_name = 'fridge.svg', robot pose: x=1.79, y=4.5
    for testing fridge: plan_name = 'kitchen.svg', robot pose: x=1.79, y=8 | 5.5
    for testing planning: plan_name = 'studio0.svg', robot pose: x=1.79, y=8
    """
    world = World(args, time_step=args.time_step)

    floor = load_floor_plan(world, plan_name='kitchen.svg') ## studio0, studio1
    # load_experiment_objects(world, CABBAGE_ONLY=False)
    world.remove_object(floor)  ## remove the floor for support

    ## base_q=(0, 0, 0))  ## 4.309, 5.163, 0.82))  ##
    robot = create_pr2_robot(world, base_q=(1.79, 6, PI/2+PI/2))
    # set_camera_target_robot(robot, FRONT=False)
    if args.camera: robot.cameras[-1].get_image(segment=args.segment)
    # remove_object(floor) ## remove the floor for support

    # floor = load_pybullet(FLOOR_URDF)

    # open_doors_drawers(5)  ## open fridge
    # open_joint(2, 'indigo_drawer_top_joint')
    # open_joint(2, 56)  ## open fridge
    # open_joint(2, 58)  ## open fridge
    # open_all_doors_drawers()  ## for debugging
    set_all_static()
    # enable_gravity()
    # wait_if_gui('Begin?')

    exogenous = []
    state = State(world)

    return robot, state, exogenous

    # enable_gravity()
    # enable_real_time()
    # run_thread(robot)


def load_floor_plan(world, plan_name='studio1.svg', DEBUG=False, spaces=None, surfaces=None,
                    asset_path=ASSET_PATH, RANDOM_INSTANCE=False, verbose=True):
    world.floorplan = plan_name

    if spaces == None:
        spaces = {
            'counter': {
                # 'sektion': [], ## 'OilBottle', 'VinegarBottle'
                # 'dagger': ['Salter'],
                # 'hitman_drawer_top': [],  ## 'Pan'
                # 'hitman_drawer_bottom': ['Pan'],
                # 'indigo_drawer_top': ['Fork'],  ## 'Fork', 'Knife'
                # 'indigo_drawer_bottom': ['Fork', 'Knife'],
                # 'indigo_tmp': ['Pot']
            },
        }
    if surfaces == None:
        surfaces = {
            'counter': {
                'front_left_stove': [],  ## 'Kettle'
                'front_right_stove': ['BraiserBody'],  ## 'PotBody',
                # 'back_left_stove': [],
                # 'back_right_stove': [],
                # 'range': [], ##
                'hitman_tmp': ['Microwave'],  ##
                'indigo_tmp': ['BraiserLid'],  ## 'MeatTurkeyLeg', 'Toaster'
            },
            'Fridge': {
                'shelf_top': ['MilkBottle'],  ## 'Egg', 'Egg',
                # 'shelf_bottom': [  ## for recording many objects
                #     'VeggieCabbage', ## 'MeatTurkeyLeg',
                #     'VeggieArtichoke',
                #     'VeggieTomato',
                #     'VeggieZucchini', 'VeggiePotato', 'VeggieCauliflower',
                #     'MeatChicken',
                #     'VeggieGreenPepper',
                # ]
                # 'shelf_bottom': ['VeggieCabbage']  ## for kitchen demo
                'shelf_bottom': []  ## 'VeggieCabbage' ## for HPN testing
            },
            'Basin': {
                'faucet_platform': ['Faucet']
            }
        }
    spaces = {k.lower(): v for k, v in spaces.items()}
    surfaces = {k.lower(): v for k, v in surfaces.items()}
    regions = list(surfaces.keys()) + list(spaces.keys())

    ## read xml file
    objects, X_OFFSET, Y_OFFSET, SCALING, FLOOR_X_MIN, FLOOR_X_MAX, FLOOR_Y_MIN, FLOOR_Y_MAX = read_xml(plan_name, asset_path=asset_path)

    #######################################################
    ## add reference floor
    w = (FLOOR_X_MAX - FLOOR_X_MIN) / SCALING
    l = (FLOOR_Y_MAX - FLOOR_Y_MIN) / SCALING
    x = ((FLOOR_X_MIN + FLOOR_X_MAX) / 2 - X_OFFSET) / SCALING
    y = ((FLOOR_Y_MIN + FLOOR_Y_MAX) / 2 - Y_OFFSET) / SCALING
    floor = world.add_object(
        Floor(create_box(w=round(w, 1), l=round(l, 1), h=FLOOR_HEIGHT, color=TAN, collision=True)),
        Pose(point=Point(x=round(x, 1), y=round(y, 1), z=-2 * FLOOR_HEIGHT)))

    #######################################################
    ## add each static object
    for name, o in objects.items():
        cat = o['category'].lower()
        x = (o['x'] - X_OFFSET) / SCALING
        y = (o['y'] - Y_OFFSET) / SCALING
        w = o['w'] / SCALING
        l = o['l'] / SCALING

        if cat == 'floor':
            world.add_box(
                Floor(create_box(w=round(w, 1), l=round(l, 1), h=FLOOR_HEIGHT, color=TAN, collision=True), name=name),
                Pose(point=Point(x=round(x, 1), y=round(y, 1), z=-2 * FLOOR_HEIGHT)))
            continue

        ## add the object itself
        yaw = {0: 0, 90: PI / 2, 180: PI, 270: -PI / 2}[o['yaw']]
        obj = world.add_object(Object(
            load_asset(cat, x=round(x, 1), y=round(y, 1), yaw=yaw, floor=floor,
                       w=round(w, 1), l=round(l, 1), RANDOM_INSTANCE=RANDOM_INSTANCE),
            category=cat))
        body = obj.body
        if 'door' in cat.lower():
            world.add_box(
                Floor(create_box(w=round(w, 1), l=round(l, 1), h=FLOOR_HEIGHT, color=TAN, collision=True), name=f'doorway_{name}'),
                Pose(point=Point(x=round(x, 1), y=round(y, 1), z=-2 * FLOOR_HEIGHT)))

        #######################################################
        ## add moveable objects on designated places
        if not LOAD_MOVEABLES: continue

        ## PLACE UTENCILS & INGREDIENTS
        # if cat.lower() == 'dishwasher':
        #     print("cat.lower() == 'dishwasher'")
        #     DEBUG = True
        # else:
        #     DEBUG = False

        if cat in regions:

            if DEBUG:
                world.open_doors_drawers(body, ADD_JOINT=False)
                set_camera_target_body(body, dx=0.05, dy=0.05, dz=0.5)

            for link in get_links(body):
                # dump_link(body, link)
                # set_color(body, YELLOW, link)
                # draw_link_name(body, link)

                link_name = get_link_name(body, link)

                # if link_name == 'front_right_stove':
                #     print('ss front_right_stove')
                if cat in surfaces and link_name in surfaces[cat]:
                    surface = Surface(body, link=link)
                    world.add_object(surface)
                    for o in surfaces[cat][link_name]:
                        surface.place_new_obj(o)

                if cat in spaces and link_name in spaces[cat]:
                    space = Space(body, link=link)
                    world.add_object(space)
                    for o in spaces[cat][link_name]:
                        space.place_new_obj(o) ##, verbose=cat.lower() == 'dishwasher'
            if DEBUG:
                world.close_doors_drawers(body)

    world.close_all_doors_drawers()
    for surface in ['faucet_platform', 'shelf_top']:
        world.remove_body_from_planning(world.name_to_body(surface))
    set_renderer(True)
    return floor

def load_rooms(world, DOOR_GAP = 1.9):

    kitchen = world.add_object(
        Environment(create_room(width=3, length=3, height=WALL_HEIGHT,
                                thickness=0.05, gap=DOOR_GAP, yaw=0.), name='kitchen'),
        Pose(point=Point(x=-1, y=0, z=0)))

    storage = world.add_object(
        Environment(create_room(width=3, length=2, height=WALL_HEIGHT,
                                thickness=0.05, gap=DOOR_GAP, yaw=0.), name='storage'),
        Pose(point=Point(x=-1, y=-2.5, z=0)))

    laundry_room = world.add_object(
        Environment(create_room(width=3, length=3, height=WALL_HEIGHT,
                                thickness=0.05, gap=DOOR_GAP, yaw=PI), name='laundry_room'),
        Pose(point=Point(x=4, y=0, z=0)))

    hallway = world.add_object(
        Environment(create_box(w=2, l=5, h=FLOOR_HEIGHT, color=YELLOW, collision=True),
                    name='hallway'),
        Pose(point=Point(x=1.5, y=-1, z=-2 * FLOOR_HEIGHT)))

    living_room = world.add_object(
        Environment(create_room(width=3, length=8, height=WALL_HEIGHT,
                                thickness=0.05, gap=2, yaw=-PI / 2), name='living_room'),
        Pose(point=Point(x=1.5, y=3, z=0)))

    return kitchen


def load_cart(world, cart_x=-0.25, cart_y=0, marker_name='marker'):
    floor = world.add_object(
        Floor(create_box(w=10, l=10, h=FLOOR_HEIGHT, color=BLACK, collision=True)),
        Pose(point=Point(x=0, y=0, z=-2 * FLOOR_HEIGHT)))

    cart_visual = world.add_object(Steerable(
        load_asset('CartVisual', x=cart_x, y=cart_y, yaw=PI, floor=floor), category='cart'))
    x, y, z = find_point_for_single_push(cart_visual)
    world.remove_object(cart_visual)

    cart = world.add_object(Steerable(
        load_asset('Cart', x=cart_x, y=cart_y, yaw=PI, floor=floor), category='cart'))
    world.remove_object(floor)

    ## --------- mark the cart handle to grasp --------
    marker = world.add_object(
        Moveable(create_box(.05, .05, .05, mass=1, color=LIGHT_GREY), ## (0, 1, 0, 1)
                 category='marker', name=marker_name),
        Pose(point=Point(x, y, z))) ## +0.05
    cart.add_grasp_marker(marker)
    # sample_points_along_line(cart, marker)
    ## -------------------------------------------------

    return cart, marker

def load_cart_regions(world, w=.5, h=.9, mass=1):
    """ used in problem=test_cart_obstacle and test_cart_obstacle_wconf """
    table = world.add_object(
        Object(create_box(w, w, h, color=(.75, .75, .75, 1)), category='supporter', name='table'),
        Pose(point=Point(x=-1.3, y=0, z=h / 2)))

    cabbage = world.add_object(
        Moveable(create_box(.07, .07, .1, mass=mass, color=(0, 1, 0, 1)), name='cabbage'),
        Pose(point=Point(x=-1.3, y=0, z=h + .1 / 2)))

    kitchen = world.add_object(
        Environment(create_room(width=3, length=3, height=WALL_HEIGHT,
                                thickness=0.05, gap=1.9, yaw=0.), name='kitchen'),
        Pose(point=Point(x=-1, y=0, z=0)))

    laundry_room = world.add_object(
        Environment(create_box(w=2, l=5, h=FLOOR_HEIGHT, color=YELLOW, collision=True),
                    name='laundry_room'),
        Pose(point=Point(x=5, y=0, z=-2 * FLOOR_HEIGHT)))

    # doorway = world.add_object(
    #     Environment(create_box(w=2, l=5, h=FLOOR_HEIGHT, color=YELLOW, collision=True),
    #                 name='doorway'),
    #     Pose(point=Point(x=8, y=0, z=-2 * FLOOR_HEIGHT)))

    storage = world.add_object(
        Environment(create_box(w=3, l=2, h=FLOOR_HEIGHT, color=YELLOW, collision=True),
                    name='storage'),
        Pose(point=Point(x=-1, y=-2.5, z=-2 * FLOOR_HEIGHT)))

    cart, marker = load_cart(world)

    return cabbage, kitchen, laundry_room, storage, cart, marker

def load_blocked_kitchen(world, w=.5, h=.9, mass=1, DOOR_GAP=1.9):
    table = world.add_object(
        Object(create_box(w, w, h, color=(.75, .75, .75, 1)), category='supporter', name='table'),
        Pose(point=Point(x=-1.3, y=0, z=h / 2)))

    cabbage = world.add_object(
        Moveable(create_box(.07, .07, .1, mass=mass, color=(0, 1, 0, 1)), name='cabbage'),
        Pose(point=Point(x=-1.3, y=0, z=h + .1 / 2)))

    kitchen = world.add_object(
        Environment(create_room(width=3, length=3, height=WALL_HEIGHT,
                                thickness=0.05, gap=DOOR_GAP, yaw=0.), name='kitchen'),
        Pose(point=Point(x=-1, y=0, z=0)))

    cart, marker = load_cart(world)

    return cabbage, cart, marker

def load_blocked_sink(world, w=.5, h=.9, mass=1, DOOR_GAP=1.9):
    sink = world.add_object(
        Supporter(create_box(w, w, h, color=(.25, .25, .75, 1)), category='sink'),
        Pose(point=Point(x=-1.3, y=-3, z=h / 2)))

    storage = world.add_object(
        Environment(create_room(width=3, length=3, height=WALL_HEIGHT,
                                thickness=0.05, gap=DOOR_GAP, yaw=0.), name='storage'),
        Pose(point=Point(x=-1, y=-3, z=0)))

    cart2, marker2 = load_cart(world, cart_x=-0.25, cart_y=-3, marker_name='marker2')

    return sink, cart2, marker2

def load_blocked_stove(world, w=.5, h=.9, mass=1, DOOR_GAP=1.9):
    stove = world.add_object(
        Supporter(create_box(w, w, h, color=(.75, .25, .25, 1)), category='stove'),
        Pose(point=Point(x=-1.3, y=3, z=h / 2)))

    storage2 = world.add_object(
        Environment(create_room(width=3, length=3, height=WALL_HEIGHT,
                                thickness=0.05, gap=DOOR_GAP, yaw=0.), name='storage2'),
        Pose(point=Point(x=-1, y=3, z=0)))

    cart3, marker3 = load_cart(world, cart_x=-0.25, cart_y=3, marker_name='marker3')

    return stove, cart3, marker3

#######################################################


def load_pot_lid(world):
    name_to_body = world.name_to_body
    name_to_object = world.name_to_object

    ## -- add pot
    pot = name_to_body('braiserbody')
    world.put_on_surface(pot, 'front_right_stove', OAO=True)

    ## ------ put in egg without lid
    world.add_object(Surface(pot, link_from_name(pot, 'braiser_bottom')))
    bottom = name_to_body('braiser_bottom')

    lid = name_to_body('braiserlid')
    world.put_on_surface(lid, 'braiserbody')
    world.add_to_cat(lid, 'moveable')

    return bottom, lid


def load_basin_faucet(world):
    from .actions import ChangeLinkColorEvent, CreateCylinderEvent, RemoveBodyEvent
    cold_blue = RGBA(0.537254902, 0.811764706, 0.941176471, 1.)

    name_to_body = world.name_to_body
    name_to_object = world.name_to_object

    faucet = name_to_body('faucet')
    basin = world.add_surface_by_keyword('basin', 'basin_bottom')
    set_color(basin.body, GREY, basin.link)

    handles = world.add_joints_by_keyword('faucet', 'joint_faucet', 'knob')
    # set_color(faucet, RED, name_to_object('joint_faucet_0').handle_link)

    ## left knob gives cold water
    left_knob = name_to_object('joint_faucet_0')
    events = []
    h_min = 0.05
    h_max = 0.4
    num_steps = 5
    x, y, z = get_aabb_center(get_aabb(faucet, link=link_from_name(faucet, 'tube_head')))
    for step in range(num_steps):
        h = h_min + step / num_steps * (h_max - h_min)
        event = CreateCylinderEvent(0.005, h, cold_blue, ((x, y, z - h / 2), (0, 0, 0, 1)))
        # events.extend([event, RemoveBodyEvent(event=event)])
        events.append(event)
        # water = create_cylinder(radius=0.005, height=h, color=cold_blue)
        # set_pose(water, ((x, y, z-h/2), (0, 0, 0, 1)))
        # remove_body(water)
    events.append(ChangeLinkColorEvent(basin.body, cold_blue, basin.link))
    left_knob.add_events(events)

    # ## right knob gives warm water
    # right_knob = name_to_body('joint_faucet_1')

    left_knob = name_to_body('joint_faucet_0')
    return faucet, left_knob

def load_kitchen_mechanism(world):
    name_to_body = world.name_to_body
    name_to_object = world.name_to_object

    bottom, lid = load_pot_lid(world)
    faucet, left_knob = load_basin_faucet(world)

    world.add_joints_by_keyword('fridge', 'fridge_door')
    world.add_joints_by_keyword('oven', 'knob_joint_2', 'knob')
    world.remove_body_from_planning(name_to_body('hitman_tmp'))

    world.add_to_cat(name_to_body('basin_bottom'), 'CleaningSurface')
    world.add_to_cat(name_to_body('braiser_bottom'), 'HeatingSurface')
    name_to_object('joint_faucet_0').add_controlled(name_to_body('basin_bottom'))
    name_to_object('knob_joint_2').add_controlled(name_to_body('braiser_bottom'))

def load_kitchen_mechanism_stove(world):
    name_to_body = world.name_to_body
    name_to_object = world.name_to_object

    controllers = {
        'back_right_stove': 'knob_joint_1',
        'back_left_stove': 'knob_joint_3',
        'front_left_stove': 'knob_joint_4',
    }
    for k, v in controllers.items():
        world.add_joints_by_keyword('oven', v, 'knob')
        world.add_to_cat(name_to_body(k), 'HeatingSurface')
        name_to_object(v).add_controlled(name_to_body(k))

def load_gripper_test_scene(world):
    surfaces = {
        'counter': {
            'front_left_stove': [],
            'front_right_stove': ['BraiserBody'],
            'hitman_tmp': [],
            'indigo_tmp': ['BraiserLid', 'MeatTurkeyLeg'],  ## , 'VeggieCabbage'
        }
    }

    floor = load_floor_plan(world, plan_name='counter.svg', surfaces=surfaces)
    world.remove_object(floor)
    pot, lid = load_pot_lid(world)
    set_camera_target_body(lid, dx=1.5, dy=0, dz=0.7)

    turkey = world.name_to_body('turkey')
    world.add_to_cat(turkey, 'moveable')
    world.add_to_cat(lid, 'moveable')
    return pot, lid, turkey

def load_cabinet_test_scene(world, RANDOM_INSTANCE=False, MORE_MOVABLE=False, verbose=True):
    surfaces = {
        'counter': {
            'front_left_stove': [],
            'front_right_stove': ['BraiserBody'],
            'hitman_tmp': [],
            'indigo_tmp': ['BraiserLid', 'MeatTurkeyLeg'],  ## , 'VeggieCabbage'
        }
    }
    spaces = {
        'counter': {
            'sektion': ['Bottle'], ##
            'dagger': [], ## 'Salter', 'VinegarBottle'
            'hitman_drawer_top': [],  ## 'Pan'
            # 'hitman_drawer_bottom': ['Pan'],
            # 'indigo_drawer_top': ['Fork'],  ## 'Fork', 'Knife'
            # 'indigo_drawer_bottom': ['Fork', 'Knife'],
            # 'indigo_tmp': ['Pot']
        },
    }
    if MORE_MOVABLE:
        surfaces['counter']['hitman_tmp'].append('VeggieCabbage')

    floor = load_floor_plan(world, plan_name='counter.svg', DEBUG=True, verbose=verbose,
                            surfaces=surfaces, spaces=spaces, RANDOM_INSTANCE=RANDOM_INSTANCE)
    world.remove_object(floor)
    pot, lid = load_pot_lid(world)

    lid = world.name_to_body('lid')
    pot = world.name_to_body('braiser_bottom')
    turkey = world.name_to_body('turkey')
    counter = world.name_to_body('indigo_tmp')
    oil = world.name_to_body('bottle')
    vinegar = world.name_to_body('vinegarbottle')

    world.add_to_cat(oil, 'moveable')
    world.add_to_cat(lid, 'moveable')
    world.add_joints_by_keyword('counter', 'chewie_door')
    world.add_joints_by_keyword('counter', 'dagger_door')

    ### ------- more objects
    if MORE_MOVABLE:
        world.add_to_cat(turkey, 'moveable')

        veggie = world.name_to_body('veggiecabbage')
        world.add_to_cat(veggie, 'moveable')
        world.put_on_surface(veggie, pot)

    return pot, lid, turkey, counter, oil, vinegar


def load_cabinet_rearrange_scene(world):
    surfaces = {
        'counter': {
            'front_left_stove': [],
            'front_right_stove': ['BraiserBody'],
            'hitman_tmp': [],
            'indigo_tmp': ['BraiserLid', 'MeatTurkeyLeg', 'VeggieCabbage'],  ##
        }
    }
    spaces = {
        'counter': {
            'sektion': [],  ##
            'dagger': ['VinegarBottle', 'OilBottle'],  ## 'Salter',
            'hitman_drawer_top': [],  ## 'Pan'
            # 'hitman_drawer_bottom': ['Pan'],
            # 'indigo_drawer_top': ['Fork'],  ## 'Fork', 'Knife'
            # 'indigo_drawer_bottom': ['Fork', 'Knife'],
            # 'indigo_tmp': ['Pot']
        },
    }

    floor = load_floor_plan(world, plan_name='counter.svg', surfaces=surfaces, spaces=spaces)
    world.remove_object(floor)
    pot, lid = load_pot_lid(world)

    turkey = world.name_to_body('turkey')
    counter = world.name_to_body('indigo_tmp')
    oil = world.name_to_body('bottle')
    vinegar = world.name_to_body('vinegarbottle')
    veggie = world.name_to_body('veggie')

    world.add_to_cat(oil, 'bottle')
    world.add_to_cat(vinegar, 'bottle')
    world.add_to_cat(vinegar, 'moveable')
    world.add_to_cat(oil, 'moveable')
    world.add_to_cat(lid, 'moveable')
    world.add_to_cat(turkey, 'moveable')
    world.add_to_cat(veggie, 'moveable')
    world.add_to_cat(turkey, 'edible')
    world.add_to_cat(veggie, 'edible')

    world.add_joints_by_keyword('counter', 'chewie_door')
    world.add_joints_by_keyword('counter', 'dagger_door')

    return pot, lid, turkey, veggie, counter, oil, vinegar


def load_feg_kitchen_dishwasher(world):
    surfaces = {
        'counter': {
            'front_left_stove': [],
            'front_right_stove': ['BraiserBody'],
            # 'back_left_stove': [],
            # 'back_right_stove': [],
            'hitman_tmp': [],
            'indigo_tmp': ['BraiserLid', 'MeatTurkeyLeg', 'VeggieCabbage'],  ##
        },
        'Fridge': {
            'shelf_top': [],  ## 'Egg', 'Egg', 'MilkBottle'
            # 'shelf_bottom': [  ## for recording many objects
            #     'VeggieCabbage', ## 'MeatTurkeyLeg',
            #     'VeggieArtichoke',
            #     'VeggieTomato',
            #     'VeggieZucchini', 'VeggiePotato', 'VeggieCauliflower',
            #     'MeatChicken',
            #     'VeggieGreenPepper',
            # ]
            # 'shelf_bottom': ['VeggieCabbage']  ## for kitchen demo
            'shelf_bottom': []  ## 'VeggieCabbage' ## for HPN testing
        },
        'Basin': {
            'faucet_platform': ['Faucet']
        },
        'dishwasher': {
            "surface_plate_left": ['PlateFat'],  ## 'VeggieTomato'
            # "surface_plate_right": ['PlateFlat']  ## two object attached to one joint is too much
        }
    }
    spaces = {
        # 'counter': {
        #     # 'sektion': [],  ##
        #     # 'dagger': ['VinegarBottle', 'OilBottle'],  ## 'Salter',
        #     # 'hitman_drawer_top': [],  ## 'Pan'
        #     # 'hitman_drawer_bottom': ['Pan'],
        #     # 'indigo_drawer_top': ['Fork'],  ## 'Fork', 'Knife'
        #     # 'indigo_drawer_bottom': ['Fork', 'Knife'],
        #     # 'indigo_tmp': ['Pot']
        # }
    }
    floor = load_floor_plan(world, plan_name='kitchen_v3.svg', surfaces=surfaces, spaces=spaces)
    world.remove_object(floor)
    load_kitchen_mechanism(world)
    # load_kitchen_mechanism_stove(world)
    dishwasher_door = world.add_joints_by_keyword('dishwasher', 'dishwasher_door')[0]

    cabbage = world.name_to_body('cabbage')
    chicken = world.name_to_body('turkey')
    for ingredient in [cabbage, chicken]:
        world.add_to_cat(ingredient, 'edible')
        world.add_to_cat(ingredient, 'moveable')
    world.put_on_surface(cabbage, 'shelf_bottom')
    world.put_on_surface(chicken, 'indigo_tmp')

    lid = world.name_to_body('lid')
    world.open_joint_by_name('fridge_door', pstn=1.5)
    # world.put_on_surface(lid, 'indigo_tmp')

    # world.add_to_cat(chicken, 'cleaned')

    ## ------- test placement with tomato
    # obj = world.name_to_object('tomato')
    # world.name_to_object('surface_plate_left').attach_obj(obj)
    # world.add_to_init(['ContainObj', obj.body])
    # world.add_to_init(['AtAttachment', obj.body, dishwasher_door])

    world.open_joint_by_name('dishwasher_door')
    obj = world.name_to_object('PlateFat')
    obj.set_pose(((0.97, 6.23, 0.512), quat_from_euler((0, 0, math.pi))))
    world.name_to_object('surface_plate_left').attach_obj(obj)
    world.add_to_cat(obj.body, 'movable')
    world.add_to_cat(obj.body, 'surface')
    world.add_to_init(['ContainObj', obj.body])
    world.add_to_init(['AtAttachment', obj.body, dishwasher_door])
    world.close_joint_by_name('dishwasher_door')

    ## ------- two object attached to one joint is too much
    # obj = world.name_to_object('PlateFlat')
    # obj.set_pose(((0.97, 6.23, 0.495), quat_from_euler((0, 0, math.pi))))
    # world.name_to_object('surface_plate_right').attach_obj(obj)
    # world.add_to_init(['ContainObj', obj.body])
    # world.add_to_init(['AtAttachment', obj.body, dishwasher_door])

    return dishwasher_door

def load_feg_kitchen(world):
    surfaces = {
        'counter': {
            'front_left_stove': [],
            'front_right_stove': ['BraiserBody'],
            # 'back_left_stove': [],
            # 'back_right_stove': [],
            'hitman_tmp': [],
            'indigo_tmp': ['BraiserLid', 'MeatTurkeyLeg', 'VeggieCabbage'],  ##
        },
        'Fridge': {
            'shelf_top': [],  ## 'Egg', 'Egg', 'MilkBottle'
            # 'shelf_bottom': [  ## for recording many objects
            #     'VeggieCabbage', ## 'MeatTurkeyLeg',
            #     'VeggieArtichoke',
            #     'VeggieTomato',
            #     'VeggieZucchini', 'VeggiePotato', 'VeggieCauliflower',
            #     'MeatChicken',
            #     'VeggieGreenPepper',
            # ]
            # 'shelf_bottom': ['VeggieCabbage']  ## for kitchen demo
            'shelf_bottom': []  ## 'VeggieCabbage' ## for HPN testing
        },
        'Basin': {
            'faucet_platform': ['Faucet']
        },
    }
    floor = load_floor_plan(world, plan_name='kitchen_v3.svg', surfaces=surfaces)
    world.remove_object(floor)
    load_kitchen_mechanism(world)
    # load_kitchen_mechanism_stove(world)

    cabbage = world.name_to_body('cabbage')
    chicken = world.name_to_body('turkey')
    for ingredient in [cabbage, chicken]:
        world.add_to_cat(ingredient, 'edible')
        world.add_to_cat(ingredient, 'moveable')
    world.put_on_surface(cabbage, 'shelf_bottom')
    world.put_on_surface(chicken, 'indigo_tmp')

    lid = world.name_to_body('lid')
    world.open_joint_by_name('fridge_door', pstn=1.5)
    # world.put_on_surface(lid, 'indigo_tmp')

    world.add_to_cat(chicken, 'cleaned')


def load_random_mini_kitchen_counter(world, w=6, l=6, h=0.9, wb=.07, hb=.1, table_only=False):
    """ each kitchen counter has one minifridge and one microwave
    """
    floor = world.add_object(
        Floor(create_box(w=w, l=l, h=FLOOR_HEIGHT, color=TAN, collision=True)),
        Pose(point=Point(x=w/2, y=l/2, z=-2 * FLOOR_HEIGHT)))

    h = random.uniform(0.3, 0.9)
    # h = random.uniform(0.4, 1.1)
    counter = world.add_object(Object(
        load_asset('KitchenCounter', x=w/2, y=l/2, yaw=math.pi, floor=floor, h=h,
                   RANDOM_INSTANCE=True, verbose=False), category='supporter', name='counter'))

    ## --- add cabage on an external table
    x, y = 1, 3
    # table = world.add_object(
    #     Object(create_box(0.5, 0.5, h, color=(.75, .75, .75, 1)), category='supporter', name='table'),
    #     Pose(point=Point(x=x, y=y, z=h / 2)))
    cat = 'Food' ## 'VeggieCabbage'
    yaw = random.uniform(-math.pi, math.pi)
    cabbage = world.add_object(Moveable(
        load_asset(cat, x=x, y=y, yaw=yaw, floor=floor, RANDOM_INSTANCE=True), ## , SAMPLING=True
        category=cat
    ))

    if table_only:
        return None

    ## --- ADD A FRIDGE TO BE PUT INTO OR ONTO, ALIGN TO ONE SIDE
    minifridge = world.add_object(Object(
        load_asset('MiniFridge', x=w/2, y=l/2, yaw=math.pi, floor=counter, ## SAMPLING=cabbage,
                   RANDOM_INSTANCE=True), name='minifridge'))
    x = get_aabb(counter).upper[0] - get_aabb_extent(get_aabb(minifridge))[0]/2 + 0.2
    y_min = get_aabb(counter).lower[1] + get_aabb_extent(get_aabb(minifridge))[1]/2
    y_max = get_aabb_center(get_aabb(counter))[1]
    if y_min > y_max:
        y = y_max
    else:
        y = random.uniform(y_min, y_max)
    (_, _, z), quat = get_pose(minifridge)
    z += 0.05
    set_pose(minifridge, ((x, y, z), quat))
    set_camera_target_body(minifridge, dx=2, dy=0, dz=2)

    ## --- ADD EACH DOOR JOINT
    minifridge_doors = get_partnet_doors(minifridge.path, minifridge.body)
    for door_joint in minifridge_doors:
        world.add_joint_object(door_joint[0], door_joint[1], 'door')
    # world.add_joints_by_keyword('minifridge', category='door')

    ## --- ADD ONE SPACE TO BE PUT INTO
    minifridge_spaces = get_partnet_spaces(minifridge.path, minifridge.body)
    for b, _, l in minifridge_spaces:
        fridgestorage = world.add_object(Space(b, l, name='fridgestorage'))

        ## --- PICK FROM THE STORAGE
        fridgestorage.place_obj(cabbage)
        (x0, y0, z0), quat0 = get_pose(cabbage)
        y0 = max(y0, get_aabb(b, link=l).lower[0] + 0.5)
        y0 = min(y0, get_aabb(b, link=l).upper[0] - 0.5)
        offset = get_aabb_extent(get_aabb(cabbage))[0] / 2 + random.uniform(0.05, 0.15) ## 0.05
        x0 = get_aabb(b, link=l).upper[0] - offset
        set_pose(cabbage, ((x0, y0, z0), quat0))
        fridgestorage.include_and_attach(cabbage)
        break

    # ## --- PICK FROM THE TABLE
    # counter.place_obj(cabbage)
    # obstacles = [minifridge.body]
    # trials = 10
    # while collided(cabbage.body, obstacles):
    #     counter.place_obj(cabbage)
    #     trials -= 1
    #     if trials <= 0:
    #         sys.exit()

    # world.open_all_doors_drawers()
    # set_camera_target_body(cabbage, dx=0.4, dy=0, dz=0)

    set_renderer(True)
    set_renderer(False)

    # wait_for_user()

    x += np.random.normal(4, 0.2)
    y += np.random.normal(0, 0.2)
    camera_pose = ((x, y, 1.3), (0.5, 0.5, -0.5, -0.5))
    world.add_camera(camera_pose)
    world.visualize_image()

    # body_joint = random.choice(list(minifridge_doors.keys()))
    # return body_joint
    return list(minifridge_doors.keys())


def load_another_mini_kitchen_counter(world, w=6, l=6):
    counter = world.name_to_body('counter')
    floor = world.name_to_body('floor')
    cabbage = world.cat_to_objects('food')[0]

    h = random.uniform(0.3, 0.9)
    table = world.add_object(Object(
        load_asset('KitchenCounter', x=w/2, y=0, yaw=math.pi, floor=floor, h=h,
                   RANDOM_INSTANCE=True, verbose=False), category='supporter', name='table'))

    (x, y, z), quat = get_pose(table)
    ## half of time on the left, half on the right
    if random.random() < 0.5:
        y = get_aabb(counter).lower[1] - get_aabb_extent(get_aabb(table))[1]/2 - 0.05
    else:
        y = get_aabb(counter).upper[1] + get_aabb_extent(get_aabb(table))[1] / 2 + 0.05
    x = get_pose(counter)[0][0]
    set_pose(table, ((x, y, z), quat))

    set_camera_target_body(table, dx=1.6, dy=1.2, dz=1.6)
    set_renderer(True)

    table.place_obj(cabbage)

    # set_renderer(False)

