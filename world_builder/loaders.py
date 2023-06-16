import random

import copy
import numpy as np
import math
import time
from pprint import pprint
import os
import string
from collections import defaultdict

from world_builder.utils import LIGHT_GREY, read_xml, load_asset, FLOOR_HEIGHT, WALL_HEIGHT, \
    find_point_for_single_push, ASSET_PATH, FURNITURE_WHITE, FURNITURE_GREY, FURNITURE_YELLOW, HEX_to_RGB, \
    get_instances, adjust_for_reachability

from world_builder.world import World, State
from world_builder.entities import Object, Region, Environment, Robot, Camera, Floor, Stove, Supporter,\
    Surface, Moveable, Space, Steerable
from world_builder.robots import PR2Robot, FEGripper
from world_builder.robot_builders import create_pr2_robot
from world_builder.utils import get_partnet_doors, get_partnet_spaces, get_partnet_links_by_type

import sys
from pybullet_tools.pr2_primitives import get_base_custom_limits
from pybullet_tools.utils import apply_alpha, get_camera_matrix, LockRenderer, HideOutput, load_model, TURTLEBOT_URDF, \
    set_all_color, dump_body, draw_base_limits, multiply, Pose, Euler, PI, draw_pose, unit_pose, create_box, TAN, Point, \
    GREEN, create_cylinder, INF, BLACK, WHITE, RGBA, GREY, YELLOW, BLUE, BROWN, stable_z, set_point, set_camera_pose, \
    set_all_static, get_model_info, load_pybullet, remove_body, get_aabb, set_pose, wait_if_gui, get_joint_names, \
    get_min_limit, get_max_limit, set_joint_position, set_joint_position, get_joints, get_joint_info, get_moving_links, \
    get_pose, get_joint_position, enable_gravity, enable_real_time, get_links, set_color, dump_link, draw_link_name, \
    get_link_pose, get_aabb, get_link_name, sample_aabb, aabb_contains_aabb, aabb2d_from_aabb, sample_placement, \
    aabb_overlap, get_links, get_collision_data, get_visual_data, link_from_name, body_collision, get_closest_points, \
    load_pybullet, FLOOR_URDF, get_aabb_center, AABB, INF, clip, aabb_union, get_aabb_center, Pose, Euler, \
    get_box_geometry, wait_unlocked, euler_from_quat, RED, \
    get_aabb_extent, multiply, GREY, create_shape_array, create_body, STATIC_MASS, set_renderer, quat_from_euler, \
    get_joint_name, wait_for_user, draw_aabb, get_bodies, euler_from_quat
from pybullet_tools.bullet_utils import place_body, add_body, Pose2d, nice, OBJ_YAWS, \
    sample_obj_on_body_link_surface, sample_obj_in_body_link_space, set_camera_target_body, \
    open_joint, close_joint, set_camera_target_robot, summarize_joints, \
    set_pr2_ready, BASE_LINK, BASE_RESOLUTIONS, BASE_VELOCITIES, BASE_JOINTS, draw_base_limits, \
    collided_around, collided, aabb_larger, equal, in_list

OBJ = '?obj'

BASE_VELOCITIES = np.array([1., 1., math.radians(180)]) / 1.  # per second
BASE_RESOLUTIONS = np.array([0.05, 0.05, math.radians(10)])  # from examples.pybullet.namo.stream import BASE_RESOLUTIONS
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


def create_table(world, w=0.5, h=0.9, xy=(0, 0), color=(.75, .75, .75, 1),
                 category='supporter', name='table'):
    return world.add_box(
        Supporter(create_box(w, w, h, color=color), category=category, name=name),
        Pose(point=Point(x=xy[0], y=xy[1], z=h / 2)))


def create_movable(world, supporter=None, xy=(0, 0), movable_category='VeggieCabbage',
                   category=None, name='cabbage'):
    # cabbage = world.add_box(
    #     Moveable(create_box(.07, .07, .1, mass=mass, color=(0, 1, 0, 1)), name='cabbage'),
    #     Pose(point=Point(x=4, y=6, z=h + .1 / 2)))
    return world.add_object(
        Moveable(load_asset(movable_category, x=xy[0], y=xy[1], yaw=0, floor=supporter),
                 category=category, name=name))


def create_house_floor(world, w=6, l=6, x=0.0, y=0.0):
    return world.add_object(
        Floor(create_box(w=w, l=l, h=FLOOR_HEIGHT, color=TAN, collision=True)),
        Pose(point=Point(x=x, y=y, z=-2 * FLOOR_HEIGHT)))


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


def load_experiment_objects(world, w=.5, h=.7, wb=.07, hb=.1, mass=1, EXIST_PLATE=True,
                            CABBAGE_ONLY=True, name='cabbage', color=(0, 1, 0, 1)):

    if not CABBAGE_ONLY:
        table = world.add_object(
            Object(create_box(w, w, h, color=(.75, .75, .75, 1)), category='supporter', name='table'),
            Pose(point=Point(x=4, y=6, z=h / 2)))

    # cabbage = world.add_object(
    #     Moveable(create_box(wb, wb, hb, mass=mass, color=color), name=name),
    #     Pose(point=Point(x=4, y=6, z=h + .1/2)))
    cabbage = world.add_object(Moveable(load_asset('VeggieCabbage', x=2, y=0, yaw=0), name=name))
    cabbage.set_pose(Pose(point=Point(x=3, y=6, z=0.9 + .1/2)))

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
    world = World(time_step=args.time_step, camera=args.camera, segment=args.segment)

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

    if spaces is None:
        spaces = {
            'counter': {
                # 'sektion': [], ## 'OilBottle', 'VinegarBottle'
                # 'dagger': ['Salter'],
                'hitman_drawer_top': [],  ## 'Pan'
                # 'hitman_drawer_bottom': [],
                'indigo_drawer_top': [],  ## 'Fork', 'Knife'
                # 'indigo_drawer_bottom': ['Fork', 'Knife'],
                # 'indigo_tmp': ['Pot']
            },
        }
    if surfaces is None:
        surfaces = {
            'counter': {
                # 'front_left_stove': [],  ## 'Kettle'
                'front_right_stove': ['BraiserBody'],  ## 'PotBody',
                # 'back_left_stove': [],
                # 'back_right_stove': [],
                # 'range': [], ##
                # 'hitman_tmp': ['Microwave'],  ##
                'indigo_tmp': ['BraiserLid'],  ## 'MeatTurkeyLeg', 'Toaster',
            },
            'Fridge': {
                # 'shelf_top': ['MilkBottle'],  ## 'Egg', 'Egg',
                # 'shelf_bottom': [  ## for recording many objects
                #     'VeggieCabbage', ## 'MeatTurkeyLeg',
                #     'VeggieArtichoke',
                #     'VeggieTomato',
                #     'VeggieZucchini', 'VeggiePotato', 'VeggieCauliflower',
                #     'MeatChicken',
                #     'VeggieGreenPepper',
                # ]
                # 'shelf_bottom': ['VeggieCabbage']  ## for kitchen demo
                # 'shelf_bottom': []  ## 'VeggieCabbage' ## for HPN testing
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
                        obj = surface.place_new_obj(o)

                        if verbose:
                            print(f'adding object {obj.name} to surface {surface.lisdf_name}')

                if cat in spaces and link_name in spaces[cat]:
                    space = Space(body, link=link)
                    world.add_object(space)
                    for o in spaces[cat][link_name]:
                        obj = space.place_new_obj(o) ##, verbose=cat.lower() == 'dishwasher'

                        if verbose:
                            print(f'adding object {obj.name} to space {space.lisdf_name}')
            if DEBUG:
                world.close_doors_drawers(body)

    world.close_all_doors_drawers()
    for surface in ['faucet_platform', 'shelf_top']:
        world.remove_body_from_planning(world.name_to_body(surface))
    set_renderer(True)
    return floor

####################################################


def load_five_table_scene(world):
    world.set_skip_joints()
    fridge = create_table(world, xy=(2, 0))
    cabbage = create_movable(world, supporter=fridge, xy=(2, 0), category='veggie')
    egg = create_movable(world, supporter=fridge, xy=(2, -0.18),
                         movable_category='Egg', category='egg', name='egg')
    salter = create_movable(world, supporter=fridge, xy=(2, 0.18),
                            movable_category='Salter', category='salter', name='salter')

    sink = create_table(world, xy=(0, 2), color=(.25, .25, .75, 1), category='sink', name='sink', w=0.1)
    plate = create_movable(world, supporter=sink, xy=(0, 2),
                           movable_category='Plate', category='plate', name='plate')

    stove = create_table(world, xy=(0, -2), color=(.75, .25, .25, 1), category='stove', name='stove')
    counter = create_table(world, xy=(-2, 2), color=(.25, .75, .25, 1), category='counter', name='counter')
    table = create_table(world, xy=(-2, -2), color=(.75, .75, .25, 1), category='table', name='table')
    return cabbage, egg, plate, salter, sink, stove, counter, table


def load_full_kitchen(world, **kwargs):
    world.set_skip_joints()

    custom_limits = ((0, 4), (4, 13))
    robot = create_pr2_robot(world, base_q=(1.79, 6, PI / 2 + PI / 2),
                             custom_limits=custom_limits, USE_TORSO=True)

    floor = load_floor_plan(world, plan_name='kitchen_v2.svg', **kwargs)  ## studio0, studio0
    cabbage = load_experiment_objects(world, CABBAGE_ONLY=True)
    counter = world.name_to_object('indigo_tmp')
    counter.place_obj(cabbage)
    (_, y, z), _ = cabbage.get_pose()
    cabbage.set_pose(Pose(point=Point(x=0.85, y=y, z=z)))
    world.remove_object(floor)

    lid = world.name_to_body('braiserlid')
    world.put_on_surface(lid, 'braiserbody')
    return cabbage


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
    world.add_not_stackable(lid, bottom)
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


def load_kitchen_mechanism(world, sink_name='sink'):
    name_to_body = world.name_to_body
    name_to_object = world.name_to_object

    bottom, lid = load_pot_lid(world)
    faucet, left_knob = load_basin_faucet(world)

    world.add_joints_by_keyword('fridge', 'fridge_door')
    world.add_joints_by_keyword('oven', 'knob_joint_2', 'knob')
    world.remove_body_from_planning(name_to_body('hitman_tmp'))

    world.add_to_cat(name_to_body(f'{sink_name}_bottom'), 'CleaningSurface')
    world.add_to_cat(name_to_body('braiser_bottom'), 'HeatingSurface')
    name_to_object('joint_faucet_0').add_controlled(name_to_body(f'{sink_name}_bottom'))
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
    counter = world.name_to_body('indigo_tmp')

    world.add_to_cat(turkey, 'moveable')
    world.add_to_cat(lid, 'moveable')

    camera_pose = ((1.7, 6.1, 1.5), (0.5, 0.5, -0.5, -0.5))
    world.add_camera(camera_pose)

    return pot, lid, turkey, counter


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

    camera_pose = ((3.7, 8, 1.3), (0.5, 0.5, -0.5, -0.5))
    world.add_camera(camera_pose)
    world.visualize_image(((3.7, 8, 1.3), (0.5, 0.5, -0.5, -0.5)))

    return pot, lid, turkey, counter, oil, vinegar


def load_cabinet_rearrange_scene(world):
    surfaces = {
        'counter': {
            # 'front_left_stove': [],
            'front_right_stove': ['BraiserBody'],
            # 'hitman_tmp': [],
            'indigo_tmp': ['BraiserLid', 'MeatTurkeyLeg', 'VeggieCabbage'],  ##
        }
    }
    spaces = {
        'counter': {
            'sektion': [],  ##
            'dagger': ['VinegarBottle', 'OilBottle'],  ## 'Salter',
            # 'hitman_drawer_top': [],  ## 'Pan'
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
            "surface_plate_left": ['Plate'],  ## 'VeggieTomato', 'PlateFat'
            # "surface_plate_right": ['Plate']  ## two object attached to one joint is too much
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
    obj = world.name_to_object('Plate')  ## 'PlateFat'
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
    turkey = world.name_to_body('turkey')
    for ingredient in [cabbage, turkey]:
        world.add_to_cat(ingredient, 'edible')
        world.add_to_cat(ingredient, 'moveable')
    world.put_on_surface(cabbage, 'shelf_bottom')
    world.put_on_surface(turkey, 'indigo_tmp')

    lid = world.name_to_body('lid')
    world.open_joint_by_name('fridge_door', pstn=1.5)
    # world.put_on_surface(lid, 'indigo_tmp')

    world.add_to_cat(turkey, 'cleaned')

    return cabbage, turkey, lid


def place_in_cabinet(fridgestorage, cabbage, place=True, world=None, learned=True):
    if not isinstance(fridgestorage, tuple):
        b = fridgestorage.body
        l = fridgestorage.link
        fridgestorage.place_obj(cabbage)
    else:
        b, _, l = fridgestorage

    (x0, y0, z0), quat0 = get_pose(cabbage)
    old_pose = (x0, y0, z0), quat0
    x_offset = random.uniform(0.05, 0.1)
    y_offset = 0.2
    y0 = max(y0, get_aabb(b, link=l).lower[1] + y_offset)
    y0 = min(y0, get_aabb(b, link=l).upper[1] - y_offset)
    x0 = get_aabb(b, link=l).upper[0] - get_aabb_extent(get_aabb(cabbage))[0] / 2 - x_offset
    pose = ((x0, y0, z0), quat0)
    # print(f'loaders.place_in_cabinet from {nice(old_pose)} to {nice(pose)}')

    # ## debug: draw the pose sampling boundary
    # x_min = get_aabb(b, link=l).upper[0] - get_aabb_extent(get_aabb(cabbage))[0] / 2 - 0.1
    # x_max = get_aabb(b, link=l).upper[0] - get_aabb_extent(get_aabb(cabbage))[0] / 2 - 0.05
    # y_min = get_aabb(b, link=l).lower[1] + y_offset
    # y_max = get_aabb(b, link=l).upper[1] - y_offset
    # z_min = z0 - get_aabb_extent(get_aabb(cabbage))[2] / 2
    # z_max = z0 + get_aabb_extent(get_aabb(cabbage))[2] / 2
    # boundary = AABB(lower=(x_min, y_min, z_min), upper=(x_max, y_max, z_max))
    # draw_aabb(boundary, color=(1, 0, 0, 1), parent=cabbage)
    # fridgestorage.world.open_all_doors_drawers(extent=0.5)

    if place:
        world = fridgestorage.world if world is None else world
        if hasattr(world, 'BODY_TO_OBJECT'):
            world.remove_body_attachment(cabbage)
        set_pose(cabbage, pose)
        fridgestorage.include_and_attach(cabbage)
    else:
        return pose


def load_random_mini_kitchen_counter(world, movable_category='food', w=6, l=6, h=0.9, wb=.07, hb=.1, table_only=False, SAMPLING=False):
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
    cat = movable_category.capitalize()
    cabbage = world.add_object(Moveable(
        load_asset(cat, x=x, y=y, yaw=random.uniform(-math.pi, math.pi),
                   floor=floor, RANDOM_INSTANCE=True, SAMPLING=SAMPLING),
        category=cat
    ))

    if table_only:
        return None

    ## --- ADD A FRIDGE TO BE PUT INTO OR ONTO, ALIGN TO ONE SIDE
    minifridge_doors = load_fridge_with_food_on_surface(world, counter, cabbage=cabbage, SAMPLING=SAMPLING)

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

    # set_renderer(True)
    # set_renderer(False)

    # wait_for_user()

    x, y = 3, 3
    x += np.random.normal(4, 0.2)
    y += np.random.normal(0, 0.2)
    camera_pose = ((x, y, 1.3), (0.5, 0.5, -0.5, -0.5))
    world.add_camera(camera_pose)
    world.visualize_image()

    # body_joint = random.choice(list(minifridge_doors.keys()))
    # return body_joint
    return minifridge_doors


def load_storage_mechanism(world, obj, epsilon=0.3):
    space = None
    ## --- ADD EACH DOOR JOINT
    doors = get_partnet_doors(obj.path, obj.body)
    for b, j in doors:
        world.add_joint_object(b, j, 'door')
        obj.doors.append((b, j))
        if random.random() < epsilon:
            world.open_joint(b, j, extent=0.9*random.random())

    ## --- ADD ONE SPACE TO BE PUT INTO
    spaces = get_partnet_spaces(obj.path, obj.body)
    for b, _, l in spaces:
        space = world.add_object(Space(b, l, name=f'{obj.category}_storage'))
        break
    return doors, space


def load_fridge_with_food_on_surface(world, counter, name='minifridge',
                                     cabbage=None, SAMPLING=False):
    (x, y, _), _ = get_pose(counter)
    SAMPLING = cabbage if SAMPLING else False
    minifridge = world.add_object(Object(
        load_asset('MiniFridge', x=x, y=y, yaw=math.pi, floor=counter, SAMPLING=SAMPLING,
                   RANDOM_INSTANCE=True), name=name))

    x = get_aabb(counter).upper[0] - get_aabb_extent(get_aabb(minifridge))[0] / 2 + 0.2
    y_min = get_aabb(counter).lower[1] + get_aabb_extent(get_aabb(minifridge))[1] / 2
    y_max = get_aabb_center(get_aabb(counter))[1]
    if y_min > y_max:
        y = y_max
    else:
        y = random.uniform(y_min, y_max)
    (_, _, z), quat = get_pose(minifridge)
    z += 0.02
    set_pose(minifridge, ((x, y, z), quat))
    world.BODY_TO_OBJECT[counter].support_obj(minifridge)

    set_camera_target_body(minifridge, dx=2, dy=0, dz=2)

    minifridge_doors, fridgestorage = load_storage_mechanism(world, minifridge)
    if cabbage is not None:
        place_in_cabinet(fridgestorage, cabbage)

    return list(minifridge_doors.keys())


def ensure_doors_cfree(doors, verbose=True, **kwargs):
    containers = [d[0] for d in doors]
    containers = list(set(containers))
    for a in containers:
        obstacles = [o for o in containers if o != a]
        if collided(a, obstacles, verbose=verbose, tag='ensure doors cfree'):
            random_set_doors(doors, **kwargs)


def ensure_robot_cfree(world, verbose=True):
    obstacles = [o for o in get_bodies() if o != world.robot]
    while collided(world.robot, obstacles, verbose=verbose, tag='ensure robot cfree'):
        world.robot.randomly_spawn()


def random_set_doors(doors, extent_max=1.0, epsilon=0.7):
    for door in doors:
        if random.random() < epsilon:
            extent = random.random()
            open_joint(door[0], door[1], extent=min(extent, extent_max))
    ensure_doors_cfree(doors, epsilon=epsilon, extent_max=extent_max)


def random_set_table_by_counter(table, counter, four_ways=True):
    (x, y, z), quat = get_pose(table)
    offset = random.uniform(0.1, 0.35)
    if four_ways:
        case = random.choice(range(4))
        # case = 3  ## debug base collision
    else:
        case = random.choice(range(2))

    if case == 0:  ## on the left of the counter
        y = get_aabb(counter).lower[1] - get_aabb_extent(get_aabb(table))[1] / 2 - offset
        x = get_pose(counter)[0][0]

    elif case == 1:  ## on the right of the counter
        y = get_aabb(counter).upper[1] + get_aabb_extent(get_aabb(table))[1] / 2 + offset
        x = get_pose(counter)[0][0]

    elif case == 2:  ## on the left of the counter, rotated 90 degrees
        y = get_aabb(counter).lower[1] - get_aabb_extent(get_aabb(table))[0] / 2 - offset
        x = get_aabb(counter).upper[0] + get_aabb_extent(get_aabb(table))[1] / 2 + offset
        r, p, yaw = euler_from_quat(quat)
        quat = quat_from_euler((r, p, yaw + math.pi / 2))

    elif case == 3:  ## on the right of the counter, rotated 90 degrees
        y = get_aabb(counter).upper[1] + get_aabb_extent(get_aabb(table))[0] / 2 + offset
        x = get_aabb(counter).upper[0] + get_aabb_extent(get_aabb(table))[1] / 2 + offset
        r, p, yaw = euler_from_quat(quat)
        quat = quat_from_euler((r, p, yaw + math.pi / 2))

    else:
        print('\n\nloaders.py whats this placement')
        sys.exit()

    set_pose(table, ((x, y, z), quat))
    return table


def load_another_table(world, w=6, l=6, table_name='table', four_ways=True):
    counter = world.name_to_body('counter')
    floor = world.name_to_body('floor')

    h = random.uniform(0.3, 0.9)
    table = world.add_object(Object(
        load_asset('KitchenCounter', x=w/2, y=0, yaw=math.pi, floor=floor, h=h,
                   RANDOM_INSTANCE=True, verbose=False),
        category='supporter', name=table_name))
    random_set_table_by_counter(table, counter, four_ways=four_ways)
    obstacles = [o for o in get_bodies() if o != table]
    while collided(table, obstacles, verbose=True, tag='load_another_table'):
        random_set_table_by_counter(table, counter, four_ways=four_ways)


def load_another_fridge(world, verbose=True, SAMPLING=False,
                        table_name='table', fridge_name='cabinet'):
    from pybullet_tools.bullet_utils import nice as r

    space = world.cat_to_bodies('space')[0]
    table = world.name_to_object(table_name)
    title = f'load_another_fridge | '

    def place_by_space(cabinet, space):
        width = get_aabb_extent(get_aabb(cabinet))[1] / 2
        y_max = get_aabb(space[0], link=space[1]).upper[1]
        y_min = get_aabb(space[0], link=space[1]).lower[1]
        y0 = get_link_pose(space[0], space[-1])[0][1]
        (x, y, z), quat = get_pose(cabinet)
        offset = random.uniform(0.4, 0.8)
        if y > y0:
            y = y_max + offset + width
        elif y < y0:
            y = y_min - offset - width
        set_pose(cabinet, ((x, y, z), quat))

    def outside_limit(cabinet):
        aabb = get_aabb(cabinet)
        limit = world.robot.custom_limits[1]
        return aabb.upper[1] > limit[1] or aabb.lower[1] < limit[0]

    ## place another fridge on the table
    doors = load_fridge_with_food_on_surface(world, table.body, fridge_name, SAMPLING=SAMPLING)
    cabinet = world.name_to_body(fridge_name)
    (x, y, z), quat = get_pose(cabinet)
    y_ori = y
    y0 = get_link_pose(space[0], space[-1])[0][1]
    place_by_space(cabinet, space)
    obstacles = [world.name_to_body('counter')]
    count = 20
    tag = f'load {fridge_name} by minifridge'
    while collided(cabinet, obstacles, verbose=True, tag=tag) or outside_limit(cabinet):
        place_by_space(cabinet, space)
        count -= 1
        if count == 0:
            print(title, f'cant {tag} after 20 trials')
            return None
    if verbose:
        (x, y, z), quat = get_pose(cabinet)
        print(f'{title} !!! moved {fridge_name} from {r(y_ori)} to {r(y)} (y0 = {r(y0)})')
    return doors


def place_another_food(world, movable_category='food', SAMPLING=False, verbose=True):
    """ place the food in one of the fridges """
    floor = world.name_to_body('floor')
    food = world.cat_to_bodies(movable_category)[0]
    space = world.cat_to_bodies('space')[0]
    placement = {}
    title = f'place_another_food ({movable_category}) |'

    def random_space():
        spaces = world.cat_to_objects('space')
        return random.choice(spaces)

    cat = movable_category.capitalize()
    new_food = world.add_object(Moveable(
        load_asset(cat, x=0, y=0, yaw=random.uniform(-math.pi, math.pi),
                   floor=floor, RANDOM_INSTANCE=True, SAMPLING=SAMPLING),
        category=cat
    ))

    s = random_space()
    place_in_cabinet(s, new_food)
    max_trial = 20
    # print(f'\nfood ({max_trial})\t', new_food.name, nice(get_pose(new_food.body)))
    # print(f'first food\t', world.body_to_name(food), nice(get_pose(food)))
    while collided(new_food, [food], verbose=verbose, tag='load food'):
        s = random_space()
        max_trial -= 1
        place_in_cabinet(s, new_food)
        # print(f'\nfood ({max_trial})\t', new_food.name, nice(get_pose(new_food.body)))
        # print(f'first food\t', world.body_to_name(food), nice(get_pose(food)))
        if max_trial == 0:
            food = world.BODY_TO_OBJECT[food].name
            if verbose:
                print(f'{title} ... unable to put {new_food} along with {food}')
            return None

    placement[new_food.body] = s.pybullet_name
    return placement


def load_another_fridge_food(world, movable_category='food', table_name='table',
                             fridge_name='cabinet', trial=0, epsilon=0.5, **kwargs):
    existing_bodies = get_bodies()

    def reset_world(world):
        for body in get_bodies():
            if body not in existing_bodies:
                obj = world.BODY_TO_OBJECT[body]
                world.remove_object(obj)
        print(f'load_another_fridge_food (trial {trial+1})')
        return load_another_fridge_food(world, movable_category, table_name=table_name,
                                        trial=trial+1, **kwargs)

    doors = load_another_fridge(world, table_name=table_name, fridge_name=fridge_name, **kwargs)
    if doors is None:
        return reset_world(world)
    placement = place_another_food(world, movable_category, **kwargs)
    if placement is None:
        return reset_world(world)
    random_set_doors(doors, epsilon=epsilon)

    return placement


##########################################################################################


def load_kitchen_mini_scene(world, **kwargs):
    """ inspired by Felix's environment for the FEG gripper """
    set_camera_pose(camera_point=[3, 0, 3], target_point=[0, 0, 1])
    draw_pose(unit_pose())

    ## load floor
    FLOOR_HEIGHT = 1e-3
    FLOOR_WIDTH = 4
    FLOOR_LENGTH = 8
    floor = world.add_object(
        Floor(create_box(w=round(FLOOR_WIDTH, 1), l=round(FLOOR_LENGTH, 1), h=FLOOR_HEIGHT, color=TAN)),
        Pose(point=Point(x=round(0.5 * FLOOR_WIDTH, 1), y=round(0, 1), z=-0.5 * FLOOR_HEIGHT)))

    ## load wall
    WALL_HEIGHT = 5
    WALL_WIDTH = FLOOR_HEIGHT
    WALL_LENGTH = FLOOR_LENGTH
    wall = world.add_object(
        Supporter(create_box(w=round(WALL_WIDTH, 1), l=round(WALL_LENGTH, 1), h=WALL_HEIGHT, color=WHITE), name='wall'),
        Pose(point=Point(x=round(-0.5 * WALL_WIDTH, 1), y=round(0, 1), z=0.5 * WALL_HEIGHT)))

    # ## sample ordering of minifridge, oven, dishwasher
    # if_fridge = random.randint(0, 4)  # 0: no fridge, odd: fridge, even: minifridge
    # if_minifridge = random.randint(0, 4)
    # if_oven = random.randint(0, 4)  # 80% chance of getting oven
    # if_dishwasher = random.randint(0, 4)
    #
    # fixtures = [None, None]
    #
    # if if_dishwasher:
    #     fixtures[0] = 'dishwasher'
    # if if_minifridge:
    #     fixtures[1] = 'minifridge'
    # random.shuffle(fixtures)
    #
    # if random.randint(0, 1):  # oven left
    #     if if_oven:
    #         fixtures.insert(0, 'oven')
    #     else:
    #         fixtures.insert(0, None)
    #     if if_fridge:
    #         fixtures.append('minifridge')
    #     else:
    #         fixtures.append(None)
    # else:
    #     if if_fridge:
    #         fixtures.insert(0, 'minifridge')
    #     else:
    #         fixtures.insert(0, None)
    #     if if_oven:
    #         fixtures.append('oven')
    #     else:
    #         fixtures.append(None)
    #
    # if if_fridge == 0:
    #     fixtures.append(None)
    #     random.shuffle(fixtures)
    # elif (if_fridge%2) == 1:
    #     if random.randint(0, 1):
    #         fixtures.insert(0, 'fridge')
    #     else:
    #         fixtures.append('fridge')
    # elif (if_fridge%2) == 0:
    #     if random.randint(0, 1):
    #         fixtures.insert(0, 'minifridge')
    #     else:
    #         fixtures.append('minifridge')

    fixtures = ['minifridge', 'oven', 'dishwasher']
    random.shuffle(fixtures)

    # sample placements of fixtures
    yaw = {0: 0, 90: PI / 2, 180: PI, 270: -PI / 2}[180]
    MIN_COUNTER_Z = 1.2
    fixtures_cfg = {}

    for idx, cat in enumerate(fixtures):
        if cat:
            center_x = 0.6
            center_y = -1.5 + idx * 1
            center_x += random.random() * 0.1 - 0.1
            center_y += random.random() * 0.1 - 0.1
            center_z = MIN_COUNTER_Z - random.random() * 0.1 - 0.05
            w = 2 * min(abs(center_x - 0), abs(1 - center_x))
            l = 2 * min(abs(center_y - 2 + idx * 1), abs(-1 + idx * 1 - center_y))
            fixture = {}
            if idx in [1, 2]:  # control height for center furniture
                obj = Object(load_asset(cat, x=center_x, y=center_y, yaw=yaw, floor=floor,
                                        w=w, l=l, h=center_z, RANDOM_INSTANCE=True), name=cat)
            else:
                obj = Object(load_asset(cat, x=center_x, y=center_y, yaw=yaw, floor=floor,
                                        w=w, l=l, h=2.5 * MIN_COUNTER_Z, RANDOM_INSTANCE=True), name=cat)
            fixture['id'] = world.add_object(obj)
            center_z = stable_z(obj.body, floor)
            # center_x = 1-get_aabb_extent(get_aabb(fixture['id'].body))[0]/2
            center_x += 1 - get_aabb(obj.body)[1][0]
            fixture['pose'] = Pose(point=Point(x=center_x, y=center_y, z=center_z), euler=Euler(yaw=yaw))
            set_pose(obj.body, fixture['pose'])
            fixtures_cfg[cat] = fixture

    # oven_aabb = get_aabb(fixtures_cfg['oven']['id'])
    # fridge_aabb = get_aabb(fixtures_cfg['fridge']['id'])
    if fixtures[0] is not None:
        min_counter_y = get_aabb(fixtures_cfg[fixtures[0]]['id'])[1][1]
    else:
        min_counter_y = -2
    # if fixtures[3] is not None:
    #     max_counter_y = get_aabb(fixtures_cfg[fixtures[3]]['id'])[0][1]
    # else:
    #     max_counter_y = 2
    max_counter_y = 2
    min_counter_z = MIN_COUNTER_Z
    if fixtures[1] is not None:
        tmp_counter_z = get_aabb(fixtures_cfg[fixtures[1]]['id'])[1][2]
        if tmp_counter_z > min_counter_z:
            min_counter_z = tmp_counter_z
    if fixtures[2] is not None:
        tmp_counter_z = get_aabb(fixtures_cfg[fixtures[2]]['id'])[1][2]
        if tmp_counter_z > min_counter_z:
            min_counter_z = tmp_counter_z
    min_counter_z += 0.1

    ## add counter
    COUNTER_THICKNESS = 0.05
    COUNTER_WIDTH = 1
    COUNTER_LENGTH = max_counter_y - min_counter_y

    counter = world.add_object(
        Supporter(create_box(w=COUNTER_WIDTH, l=COUNTER_LENGTH, h=COUNTER_THICKNESS, color=FURNITURE_WHITE),
                  name='counter'),
        Pose(point=Point(x=0.5 * COUNTER_WIDTH, y=(max_counter_y + min_counter_y) / 2, z=min_counter_z)))

    ## add microwave
    # microwave = world.name_to_body('microwave')
    # world.put_on_surface(microwave, counter)
    microwave = counter.place_new_obj('microwave', scale=0.4 + 0.1 * random.random(), world=world)
    microwave.set_pose(Pose(point=microwave.get_pose()[0], euler=Euler(yaw=math.pi)), world=world)

    ## add pot
    pot = counter.place_new_obj('kitchenpot', scale=0.2, world=world)
    pot.set_pose(Pose(point=pot.get_pose()[0], euler=Euler(yaw=yaw)), world=world)

    ## add shelf
    SHELF_HEIGHT = 2.3
    SHELF_THICKNESS = 0.05
    SHELF_WIDTH = 0.5
    MIN_SHELF_LENGTH = 1.5
    min_shelf_y = min_counter_y + random.random() * (max_counter_y - min_counter_y - MIN_SHELF_LENGTH)
    max_shelf_y = max_counter_y - random.random() * (max_counter_y - min_shelf_y - MIN_SHELF_LENGTH)
    SHELF_LENGTH = max_shelf_y - min_shelf_y

    shelf = world.add_object(
        Supporter(create_box(w=SHELF_WIDTH, l=SHELF_LENGTH, h=SHELF_THICKNESS, color=FURNITURE_WHITE, collision=True),
                  name='shelf'),
        Pose(point=Point(x=0.5 * SHELF_WIDTH, y=(max_shelf_y + min_shelf_y) / 2, z=SHELF_HEIGHT)))

    ## add cabinet next to shelf
    if min_shelf_y > -0.5:
        ins = random.choice(['00001', '00002'])
        cabinet = world.add_object(
            Object(load_asset('CabinetTop', x=0, y=min_shelf_y, z=SHELF_HEIGHT, yaw=math.pi, RANDOM_INSTANCE=ins),
                   category='cabinet', name='cabinet')
            )
    else:  ## if max_shelf_y < 0.5:
        ins = random.choice(['00003'])
        cabinet = world.add_object(
            Object(load_asset('CabinetTop', x=0, y=max_shelf_y, z=SHELF_HEIGHT, yaw=math.pi, RANDOM_INSTANCE=ins),
                   category='cabinet', name='cabinet')
            )

    food_ids, bottle_ids, medicine_ids = load_counter_moveables(world, [counter, shelf], obstacles=[])

    # add camera
    camera_pose = Pose(point=Point(x=4.2, y=0, z=2.5), euler=Euler(roll=PI / 2 + PI / 8, pitch=0, yaw=-PI / 2))
    world.add_camera(camera_pose)
    # rgb, depth, segmented, view_pose, camera_matrix = world.camera.get_image()
    wait_unlocked()

    return food_ids, bottle_ids, medicine_ids


#################################################################


def load_counter_moveables(world, counters, d_x_min=None, obstacles=[],
                           verbose=False, reachability_check=True):
    categories = ['food', 'bottle', 'medicine']
    start = time.time()
    robot = world.robot
    state = State(world)
    size_matter = len(obstacles) > 0 and obstacles[-1].name == 'braiser_bottom'
    satisfied = []
    if isinstance(counters, list):
        counters = {k: counters for k in ['food', 'bottle', 'medicine', 'bowl', 'mug', 'pan']}
    if d_x_min is None:
        d_x_min = - 0.3
    instances = {k: None for k in counters}
    n_objects = {k: 2 for k in categories}

    if world.note in [31]:
        braiser_bottom = world.name_to_object('braiser_bottom')
        obstacles = [o for o in obstacles if o.pybullet_name != braiser_bottom.pybullet_name]
        move_lid_away(world, [world.name_to_object('floor')], epsilon=1.0)
    elif world.note in [11]:
        from_storage = random.choice(['minifridge', 'cabinettop'])
        counters['food'] = [world.name_to_object(f"{from_storage}_storage")]
    elif world.note in [551]:
        counters['food'] = [world.name_to_object(f"minifridge_storage")]
        counters['bottle'] = [world.name_to_object(f"sink_bottom")]
        counters['medicine'] = [world.name_to_object(f"cabinettop_storage")]
        from world_builder.partnet_scales import DONT_LOAD
        DONT_LOAD.append('VeggieZucchini')
    elif world.note in [552]:
        counters['food'] = [world.name_to_object(f"sink_bottom")]
        counters['bottle'] = [world.name_to_object(f"sink_counter_left"),
                              world.name_to_object(f"sink_counter_right"), ]
        counters['medicine'] = [world.name_to_object(f"cabinettop_storage")]
        from world_builder.partnet_scales import DONT_LOAD
        DONT_LOAD.append('VeggieZucchini')
    elif world.note in [553]:
        counters['food'] = [world.name_to_object(n) for n in \
                            ["counter#1", "ovencounter", "sink_counter_left", "sink_counter_right"]]
        instances['food'] = ['VeggieZucchini', 'VeggiePotato']
    elif world.note in [554]:
        counters['food'] = [world.name_to_object(n) for n in \
                            ["counter#1", "ovencounter", "sink_counter_left", "sink_counter_right"]]
        counters['bottle'] = [world.name_to_object(n) for n in \
                            ["counter#1", "ovencounter", "sink_counter_left", "sink_counter_right"]]
        instances['food'] = ['VeggieArtichoke', 'VeggiePotato']
        instances['bottle'] = ['3822', '3574']
    elif world.note in ['more_movables']:
        n_objects['food'] = 3
        n_objects['bottle'] = 3

    # # weiyu debug
    # counters["bottle"] = [world.name_to_object(n) for n in ["sink#1::sink_bottom"]]

    pprint(counters)
    if verbose:
        print('\nload_counter_moveables(obstacles={})\n'.format([o.name for o in obstacles]))

    def check_size_matter(obj):
        if size_matter and aabb_larger(obstacles[-1], obj):
            satisfied.append(obj)

    def place_on_counter(obj_name, category=None, counter_choices=None, ins=True):
        if counter_choices is None:
            counter_choices = counters[obj_name]
        counter = random.choice(counter_choices)
        obj = counter.place_new_obj(obj_name, category=category, RANDOM_INSTANCE=ins, world=world)
        if verbose:
            print(f'          placed {obj} on {counter.name}')
        if 'bottom' not in counter.name:
            adjust_for_reachability(obj, counter, d_x_min, world=world)
        return obj

    def ensure_cfree(obj, obstacles, obj_name, category=None, trials=10, **kwargs):
        def check_conditions(o):
            collision = collided(o, obstacles, verbose=verbose, world=world)
            unreachable = False
            if not collision and reachability_check:
                unreachable = not isinstance(o.supporting_surface, Space) and \
                              not robot.check_reachability(o, state, verbose=verbose, debug=debug)
            size = unreachable or ((obj_name == 'food' and size_matter and len(satisfied) == 0))
            if collision or unreachable or size:
                if verbose:
                    print(f'\t\tremove {o} because collision={collision}, unreachable={unreachable}, size={size}')
                return True
            return False

        debug = (obj_name == 'bottle')
        again = check_conditions(obj)
        while again:
            # set_camera_target_body(obj.body)
            # set_renderer(True)
            # wait_if_gui()

            world.remove_object(obj)
            obj = place_on_counter(obj_name, category, **kwargs)
            check_size_matter(obj)

            trials -= 1
            if trials == 0:
                sys.exit('Could not place object')
            again = check_conditions(obj)
        return obj

    ## add food items
    food_ids = []
    in_briaser = False
    for i in range(n_objects['food']):
        kwargs = dict()
        if world.note in [31] and not in_briaser:
            kwargs['counter_choices'] = [braiser_bottom]
        obj_cat = 'food'
        obj_category = 'edible'
        if instances['food'] is not None:
            kwargs['ins'] = instances['food'][i]
        obj = place_on_counter(obj_cat, category=obj_category, **kwargs)
        check_size_matter(obj)
        obj = ensure_cfree(obj, obstacles, obj_name=obj_cat, category=obj_category, **kwargs)
        in_briaser = in_briaser or 'braiser_bottom' in obj.supporting_surface.name
        food_ids.append(obj)
        obstacles.append(obj.body)

    ## add bottles
    bottle_ids = []
    for i in range(n_objects['bottle']):
        kwargs = dict()
        obj_cat = 'bottle'
        if instances['bottle'] is not None:
            kwargs['ins'] = instances['bottle'][i]
        obj = place_on_counter(obj_cat)
        obj = ensure_cfree(obj, obstacles, obj_name='bottle', **kwargs)
        bottle_ids.append(obj)
        obstacles.append(obj.body)

    ## add medicine
    medicine_ids = []
    for i in range(n_objects['medicine']):
        obj = place_on_counter('medicine')
        obj = ensure_cfree(obj, obstacles, obj_name='medicine')
        # state = State(copy.deepcopy(world), gripper=state.gripper)
        medicine_ids.append(obj)
        obstacles.append(obj.body)

    ## add bowl
    bowl_ids = []
    for i in range(1):
        obj = place_on_counter('bowl')
        obj = ensure_cfree(obj, obstacles, obj_name='bowl')
        # state = State(copy.deepcopy(world), gripper=state.gripper)
        bowl_ids.append(obj)
        obstacles.append(obj.body)

    ## add mug
    mug_ids = []
    for i in range(1):
        obj = place_on_counter('mug')
        obj = ensure_cfree(obj, obstacles, obj_name='mug')
        # state = State(copy.deepcopy(world), gripper=state.gripper)
        mug_ids.append(obj)
        obstacles.append(obj.body)

    ## add pan
    pan_ids = []
    for i in range(1):
        obj = place_on_counter('pan')
        obj = ensure_cfree(obj, obstacles, obj_name='pan')
        # state = State(copy.deepcopy(world), gripper=state.gripper)
        pan_ids.append(obj)
        obstacles.append(obj.body)

    if world.note in [3, 31]:
        put_lid_on_braiser(world)
    print('... finished loading moveables in {}s'.format(round(time.time() - start, 2)))
    # world.summarize_all_objects()
    # wait_unlocked()
    return food_ids, bottle_ids, medicine_ids, bowl_ids, mug_ids, pan_ids


def load_moveables(world, obj_dict, d_x_min=None, obstacles=[], verbose=False, reachability_check=True):

    start = time.time()
    robot = world.robot
    state = State(world)
    size_matter = len(obstacles) > 0 and obstacles[-1].name == 'braiser_bottom'
    satisfied = []
    if d_x_min is None:
        d_x_min = - 0.3

    # # weiyu debug
    # counters["bottle"] = [world.name_to_object(n) for n in ["sink#1::sink_bottom"]]

    print("\nloading moveables")
    pprint(obj_dict)
    if verbose:
        print('\nload_counter_moveables(obstacles={})\n'.format([o.name for o in obstacles]))

    def check_size_matter(obj):
        if size_matter and aabb_larger(obstacles[-1], obj):
            satisfied.append(obj)

    def place_on_counter(obj_name, counter_name, category=None, ins=True):
        # retrieve counter name
        counter = world.name_to_object(counter_name)
        obj = counter.place_new_obj(obj_name, category=category, RANDOM_INSTANCE=ins, world=world)
        if verbose:
            print(f'placed {obj} on {counter.name}')
        if 'bottom' not in counter.name:
            adjust_for_reachability(obj, counter, d_x_min, world=world)
        return obj

    def ensure_cfree(obj, loc, obstacles, obj_name, category=None, trials=10, **kwargs):
        def check_conditions(o):
            collision = collided(o, obstacles, verbose=verbose, world=world)
            unreachable = False
            if not collision and reachability_check:
                unreachable = not isinstance(o.supporting_surface, Space) and \
                              not robot.check_reachability(o, state, verbose=verbose, debug=debug)
            size = unreachable or ((obj_name == 'food' and size_matter and len(satisfied) == 0))
            if collision or unreachable or size:
                if verbose:
                    print(f'\t\tremove {o} because collision={collision}, unreachable={unreachable}, size={size}')
                return True
            return False

        debug = (obj_name == 'bottle')
        again = check_conditions(obj)
        while again:
            # set_camera_target_body(obj.body)
            # set_renderer(True)
            # wait_if_gui()

            world.remove_object(obj)
            obj = place_on_counter(obj_name, loc, category, **kwargs)
            check_size_matter(obj)

            trials -= 1
            if trials == 0:
                sys.exit('Could not place object')
            again = check_conditions(obj)
        return obj

    def assign_color(obj, color_name):
        # link=-1 won't work
        color = eval(color_name.upper())
        set_color(obj.body, color, link=None)
        return

    for oi in sorted(obj_dict.keys()):
        cls = obj_dict[oi]["class"]
        ins = obj_dict[oi]["instance"]
        loc = obj_dict[oi]["location"]
        color = obj_dict[oi]["color"]
        if cls == 'food':
            in_briaser = False
            kwargs = dict()
            obj_category = 'edible'
            if ins is not None:
                kwargs['ins'] = ins
            obj = place_on_counter(cls, loc, category=obj_category, **kwargs)
            check_size_matter(obj)
            obj = ensure_cfree(obj, loc, obstacles, obj_name=cls, category=obj_category, **kwargs)
            in_briaser = in_briaser or 'braiser_bottom' in obj.supporting_surface.name
            obj_dict[oi]["name"] = obj
            obstacles.append(obj.body)
            assign_color(obj, color)
        elif cls in ["medicine", "bowl", "mug", "pan", "bottle"]:
            kwargs = dict()
            if ins is not None:
                kwargs['ins'] = ins
            obj = place_on_counter(cls, loc)
            obj = ensure_cfree(obj, loc, obstacles, obj_name=cls, **kwargs)
            obj_dict[oi]["name"] = obj
            obstacles.append(obj.body)
            assign_color(obj, color)

    print('... finished loading moveables in {}s'.format(round(time.time() - start, 2)))
    # world.summarize_all_objects()
    # wait_unlocked()
    return obj_dict


def move_lid_away(world, counters, epsilon=1.0):
    lid = world.name_to_body('braiserlid')  ## , obstacles=moveables+obstacles
    counters_tmp = world.find_surfaces_for_placement(lid, counters)
    if len(counters_tmp) == 0:
        raise Exception('No counters found, try another seed')
    # world.add_highlighter(counters[0])

    if random.random() < epsilon:
        counters_tmp[0].place_obj(world.BODY_TO_OBJECT[lid], world=world)
    return counters_tmp[0]


def load_table_stationaries(world, w=6, l=6, h=0.9):
    """ a table with a tray and some stationaries to be put on top
    """
    # x, y = 3, 3
    # x += np.random.normal(4, 0.2)
    # y += np.random.normal(0, 0.2)
    camera_point = (3.5, 3, 3)
    target_point = (3, 3, 1)
    set_camera_pose(camera_point, target_point)

    categories = ['EyeGlasses', 'Camera', 'Stapler', 'Medicine', 'Bottle', 'Knife']
    random.shuffle(categories)

    floor = world.add_object(
        Floor(create_box(w=w, l=l, h=FLOOR_HEIGHT, color=TAN, collision=True)),
        Pose(point=Point(x=w/2, y=l/2, z=-2 * FLOOR_HEIGHT)))

    h = random.uniform(h-0.1, h+0.1)
    counter = world.add_object(Supporter(
        load_asset('KitchenCounter', x=w/2, y=l/2, yaw=math.pi, floor=floor, h=h,
                   RANDOM_INSTANCE=True, verbose=False), category='supporter', name='counter'))
    # set_camera_target_body(counter, dx=3, dy=0, dz=2)
    aabb_ext = get_aabb_extent(get_aabb(counter.body))
    y = l/2 - aabb_ext[1] / 2 + aabb_ext[0] / 2
    tray = world.add_object(Object(
        load_asset('Tray', x=w/2, y=y, yaw=math.pi, floor=counter,
                   RANDOM_INSTANCE=True, verbose=False), name='tray'))
    tray_bottom = world.add_surface_by_keyword(tray, 'tray_bottom')

    ## --- add cabage on an external table
    items = []
    for cat in categories:
        RI = '103104' if cat == 'Stapler' else True
        item = world.add_object(Moveable(
            load_asset(cat, x=0, y=0, yaw=random.uniform(-math.pi, math.pi),
                       RANDOM_INSTANCE=RI), category=cat
        ))
        world.put_on_surface(item, 'counter')
        if collided(item.body, [tray]+items, verbose=False, tag='ensure item cfree'):
            world.put_on_surface(item, 'counter')
        items.append(item)

    distractors = []
    k = 1 ## random.choice(range(2))
    for item in items[:k]:
        world.put_on_surface(item, 'tray_bottom')
        if collided(item.body, [tray]+distractors, verbose=False, tag='ensure distractor cfree'):
            world.put_on_surface(item, 'tray_bottom')
        distractors.append(item)

    items = items[k:]

    ## --- almost top-down view
    camera_point = (4, 3, 3)
    target_point = (3, 3, 1)
    world.visualize_image(rgb=True, camera_point=camera_point, target_point=target_point)
    wait_unlocked()

    return items, distractors, tray_bottom, counter


#################################################################


def place_faucet_by_sink(faucet_obj, sink_obj, gap=0.01, world=None):
    body = faucet_obj.body
    base_link = get_partnet_links_by_type(faucet_obj.path, body, 'switch')[0]
    x = get_link_pose(body, base_link)[0][0]
    aabb = get_aabb(body, base_link)
    lx = get_aabb_extent(aabb)[0]
    x_new = sink_obj.aabb().lower[0] - lx / 2 - gap
    faucet_obj.adjust_pose(dx=x_new - x, world=world)


WALL_HEIGHT = 2.5
WALL_WIDTH = 0.1
COUNTER_THICKNESS = 0.05
diswasher = 'DishwasherBox'


def sample_kitchen_sink(world, floor=None, x=0.0, y=1.0, verbose=True, random_scale=1.0):

    if floor is None:
        floor = create_house_floor(world, w=2, l=2, x=0, y=1)
        x = 0

    ins = True
    if world.note in [551, 552]:
        ins = '45305'
    base = world.add_object(Object(
        load_asset('SinkBase', x=x, y=y, yaw=math.pi, floor=floor,
                   RANDOM_INSTANCE=ins, verbose=verbose, random_scale=random_scale), name='sinkbase'))
    dx = base.lx / 2
    base.adjust_pose(dx=dx, world=world)
    x += dx
    if base.instance_name == 'partnet_5b112266c93a711b824662341ce2b233':  ##'46481'
        x += 0.1 * random_scale

    ins = True
    if world.note in [551, 552]:
        ins = '00005'
    if world.note in [554]:
        ins = '00004'
    # stack on base, since floor is base.body, based on aabb not exact geometry
    sink = world.add_object(Object(
        load_asset('Sink', x=x, y=y, yaw=math.pi, floor=base.body,
                   RANDOM_INSTANCE=ins, verbose=verbose, random_scale=random_scale), name='sink'))
    # TODO: instead of 0.05, make it random
    # align the front of the sink with the front of the base in the x direction
    dx = (base.aabb().upper[0] - sink.aabb().upper[0]) - 0.05 * random_scale
    # align the center of the sink with the center of the base in the y direction
    dy = sink.ly / 2 - (sink.aabb().upper[1] - y)
    sink.adjust_pose(dx=dx, dy=dy, dz=0, world=world)
    sink.adjust_pose(dx=0, dy=0, dz=-sink.height + COUNTER_THICKNESS, world=world)
    # print('sink.get_pose()', sink.get_pose())
    if sink.instance_name == 'partnet_u82f2a1a8-3a4b-4dd0-8fac-6679970a9b29':  ##'100685'
        x += 0.2 * random_scale
    if sink.instance_name == 'partnet_549813be-3bd8-47dd-9a49-b51432b2f14c':  ##'100685'
        x -= 0.06 * random_scale

    ins = True
    if world.note in [551, 552]:
        ins = '14'
    faucet = world.add_object(Object(
        load_asset('Faucet', x=x, y=y, yaw=math.pi, floor=base.body,
                   RANDOM_INSTANCE=ins, verbose=verbose, random_scale=random_scale), name='faucet'))
    # adjust placement of faucet to be behind the sink
    place_faucet_by_sink(faucet, sink, world=world, gap=0.01 * random_scale)
    faucet.adjust_pose(dz=COUNTER_THICKNESS, world=world)

    xa, ya, _ = base.aabb().lower
    xb, yb, _ = sink.aabb().lower
    xc, yc, z = sink.aabb().upper
    xd, yd, _ = base.aabb().upper
    z -= COUNTER_THICKNESS/2

    padding, color = {
        'sink_003': (0.05, FURNITURE_WHITE),
        'sink_005': (0.02, FURNITURE_WHITE),
        'partnet_3ac64751-e075-488d-9938-9123dc88b2b6-0': (0.02, WHITE),
        'partnet_e60bf49d6449cf37c5924a1a5d3043b0': (0.027, FURNITURE_WHITE),  ## 101176
        'partnet_u82f2a1a8-3a4b-4dd0-8fac-6679970a9b29': (0.027, FURNITURE_GREY),  ## 100685
        'partnet_553c586e128708ae7649cce35db393a1': (0.03, WHITE),  ## 100501
        'partnet_549813be-3bd8-47dd-9a49-b51432b2f14c': (0.03, FURNITURE_YELLOW),  ## 100191
    }[sink.instance_name]

    xb += padding
    xc -= padding
    yb += padding
    yc -= padding

    counter_x = (xd+xa)/2
    counter_w = xd-xa
    left_counter = create_box(w=counter_w, l=yb-ya, h=COUNTER_THICKNESS, color=color)
    left_counter = world.add_object(Supporter(left_counter, name='sink_counter_left'),
                     Pose(point=Point(x=counter_x, y=(yb+ya)/2, z=z)))

    right_counter = create_box(w=counter_w, l=yd-yc, h=COUNTER_THICKNESS, color=color)
    right_counter = world.add_object(Supporter(right_counter, name='sink_counter_right'),
                     Pose(point=Point(x=counter_x, y=(yd+yc)/2, z=z)))

    front_counter = create_box(w=xd-xc, l=yc-yb, h=COUNTER_THICKNESS, color=color)
    world.add_object(Object(front_counter, name='sink_counter_front', category='filler'),
                     Pose(point=Point(x=(xd+xc)/2, y=(yc+yb)/2, z=z)))

    back_counter = create_box(w=xb-xa, l=yc-yb, h=COUNTER_THICKNESS, color=color)
    world.add_object(Object(back_counter, name='sink_counter_back', category='filler'),
                     Pose(point=Point(x=(xb+xa)/2, y=(yc+yb)/2, z=z)))

    set_camera_target_body(sink)

    return floor, base, counter_x, counter_w, z, color, [left_counter, right_counter]


def sample_kitchen_furniture_ordering(all_necessary=True):
    END = 'Wall'
    T = {
        'SinkBase': ['CabinetLower', diswasher, END],
        'CabinetLower': ['CabinetLower', diswasher, 'OvenCounter',
                         'CabinetTall', 'MiniFridge', END],  ## 'MicrowaveHanging', 
        diswasher: ['CabinetLower', 'CabinetTall', 'MiniFridge', END], ## 'MicrowaveHanging',
        'OvenCounter': ['CabinetLower'],
        'CabinetTall': ['MiniFridge', END],
        # 'MicrowaveHanging': ['MiniFridge', END],
        'MiniFridge': ['CabinetTall', END],
    }
    ordering = ['OvenCounter']
    left = right = 0
    necessary = [diswasher, 'MiniFridge', 'SinkBase']
    multiple = ['CabinetLower']
    optional = ['CabinetTall', 'MicrowaveHanging', END]
    while len(necessary) > 0:
        if ordering[left] == END and ordering[right] == END:
            if all_necessary:
                for each in necessary:
                    index = random.choice(range(1, len(ordering)-1))
                    ordering.insert(index, each)
            break
        else:
            if (random.random() > 0.5 and ordering[right] != END) or ordering[left] == END:
                possibile = [m for m in T[ordering[right]] if m in necessary + multiple + optional]
                chosen = random.choice(possibile)
                ordering.insert(right+1, chosen)
            else:
                possibile = [m for m in T[ordering[left]] if m in necessary + multiple + optional]
                chosen = random.choice(possibile)
                ordering.insert(0, chosen)
            right += 1
            if chosen in necessary:
                necessary.remove(chosen)
    print(ordering)
    return ordering[1:-1]


def load_full_kitchen_upper_cabinets(world, counters, x_min, y_min, y_max, dz=0.8, others=[],
                                     obstacles=[], verbose=False, random_scale=1.0):
    cabinets, shelves = [], []
    cabi_type = 'CabinetTop' if random.random() < 0.5 else 'CabinetUpper'
    if world.note in [1, 21, 31, 11, 4, 41, 991, 551, 552]:
        cabi_type = 'CabinetTop'
    colors = {
        '45526': HEX_to_RGB('#EDC580'),
        '45621': HEX_to_RGB('#1F0C01'),
        '46889': HEX_to_RGB('#A9704F'),
        '46744': HEX_to_RGB('#160207'),
        '48381': HEX_to_RGB('#F3D5C1'),
        '46108': HEX_to_RGB('#EFB064'),
        '49182': HEX_to_RGB('#FFFFFF'),
    }

    def add_wall_fillings(cabinet, color=FURNITURE_WHITE):
        if cabinet.mobility_id in colors:
            color = colors[cabinet.mobility_id]
        xa = x_min  ## counter.aabb().lower[0]
        xb, ya, za = cabinet.aabb().lower
        _, yb, zb = cabinet.aabb().upper
        xb += 0.003
        zb -= 0.003
        ya += 0.003
        yb -= 0.003

        filler = create_box(w=xb - xa, l=yb - ya, h=zb - za, color=color)
        world.add_object(Object(filler, name=f'{cabi_type}_filler', category='filler'),
                         Pose(point=Point(x=(xa + xb) / 2, y=(ya + yb) / 2, z=(za + zb) / 2)))
        world.add_ignored_pair((cabinet.body, filler))
        return color

    def place_cabinet(selected_counters, cabi_type=cabi_type, **kwargs):
        counter = random.choice(selected_counters)
        cabinet = world.add_object(
            Object(load_asset(cabi_type, yaw=math.pi, verbose=verbose, random_scale=random_scale, **kwargs),
                   category=cabi_type, name=cabi_type)
        )
        cabinet.adjust_next_to(counter, direction='+z', align='+x', dz=dz)
        cabinet.adjust_pose(dx=0.3 * random_scale, world=world)
        return cabinet, counter

    def ensure_cfree(obj, obstacles, selected_counters, **kwargs):
        counter = None
        trials = 5
        while collided(obj, obstacles, verbose=verbose, world=world) or \
                obj.aabb().upper[1] > y_max or obj.aabb().lower[1] < y_min:
            world.remove_object(obj)
            trials -= 1
            if trials < 0:
                return None
            obj, counter = place_cabinet(selected_counters, **kwargs)
        return obj, counter

    def add_cabinets(selected_counters, obstacles=[], **kwargs):
        color = FURNITURE_WHITE
        blend = []  ## cabinet overflowed to the next counter
        # for num in range(random.choice([1, 2])):
        # Debug: only add one cabinet for now
        for num in range(1):
            cabinet, counter = place_cabinet(selected_counters, **kwargs)
            result = ensure_cfree(cabinet, obstacles, selected_counters, **kwargs)
            if result is None:
                continue
            cabinet, counter2 = result
            counter = counter2 if counter2 is not None else counter
            selected_counters.remove(counter)
            color = add_wall_fillings(cabinet)
            cabinets.append(cabinet)
            obstacles.append(cabinet)
            kwargs['RANDOM_INSTANCE'] = cabinet.mobility_id
            if cabinet.aabb().upper[1] > counter.aabb().upper[1]:
                blend.append(cabinet.aabb().upper[1])
        return obstacles, color, selected_counters, blend

    def add_shelves(counters, color, bled=[], obstacles=[], **kwargs):
        new_counters = []
        last_left = counters[0].aabb().lower[1]
        last_right = counters[0].aabb().upper[1]
        for i in range(1, len(counters)):
            if last_right != counters[i].aabb().lower[1]:
                new_counters.append((last_left, last_right))
                last_left = counters[i].aabb().lower[1]
            last_right = counters[i].aabb().upper[1]
        new_counters.append((last_left, last_right))

        ## sort new counters by length
        new_counters = sorted(new_counters, key=lambda x: x[1] - x[0], reverse=True)
        xa = x_min
        xb = counters[0].aabb().upper[0]
        ya, yb = new_counters[0]
        for b in bled:
            if ya < b < yb:
                ya = b
        za = counters[0].aabb().upper[2] + dz
        zb = za + COUNTER_THICKNESS
        if (yb-ya) > 0.2:
            shelf = world.add_object(
                Supporter(create_box(w=(xb-xa), l=(yb-ya), h=COUNTER_THICKNESS, color=color),
                          name='shelf_lower'),
                Pose(point=Point(x=(xb+xa)/2, y=(yb+ya)/2, z=(zb+za)/2)))
            shelves.append(shelf)
        return shelves

    # ## load ultra-wide CabinetTop
    # wide_counters = [c for c in counters if c.ly > 1.149]
    # if len(wide_counters) > 0:
    #     add_cabinets_shelves(wide_counters, cabi_type, ['00003'])
    #
    # ## load wide CabinetTop
    # wide_counters = [c for c in counters if 1.149 > c.ly > 0.768]
    # if len(wide_counters) > 0:
    #     add_cabinets_shelves(wide_counters, cabi_type, ['00001', '00002'])

    ## sort counters by aabb().lower[1]
    counters = copy.deepcopy(counters)
    counters = sorted(counters, key=lambda x: x.aabb().lower[1])

    ## load cabinets
    ins = world.note not in [11, 4, 41]
    obstacles, color, counters, bled = add_cabinets(counters, obstacles=obstacles,
                                                    cabi_type=cabi_type, RANDOM_INSTANCE=ins)
    ## then load shelves
    add_shelves(counters+others, color, bled=bled, obstacles=obstacles)
    set_camera_target_body(cabinets[0])

    return cabinets, shelves


def load_braiser(world, supporter, x_min=None, verbose=True):
    ins = True
    if world.note in [551, 552]:
        ins = random.choice(['100038', '100023'])  ## larger braisers
    elif world.note in [553]:
        ins = random.choice(['100015'])  ## shallower braisers big enough for zucchini, ,'100693'
    braiser = supporter.place_new_obj('BraiserBody', RANDOM_INSTANCE=ins, verbose=verbose, world=world)
    braiser.adjust_pose(theta=PI, world=world)
    if supporter.mobility_id == '102044':
        aabb = supporter.aabb()
        y = aabb.lower[1] + 2/5 * supporter.ly
        braiser.adjust_pose(y=y, world=world)
    set_camera_target_body(braiser)

    if x_min is None:
        x_min = supporter.aabb().upper[0] - 0.3
    adjust_for_reachability(braiser, supporter, x_min, world=world)
    set_camera_target_body(braiser)

    lid = braiser.place_new_obj('BraiserLid', category='moveable', name='BraiserLid', max_trial=1,
                                RANDOM_INSTANCE=braiser.mobility_id, verbose=verbose, world=world)
    world.make_transparent(lid)
    put_lid_on_braiser(world, lid, braiser)

    braiser_bottom = world.add_surface_by_keyword(braiser, 'braiser_bottom')
    return braiser, braiser_bottom


def put_lid_on_braiser(world, lid=None, braiser=None):
    if lid is None:
        lid = world.name_to_object('BraiserLid')
    if braiser is None:
        braiser = world.name_to_object('BraiserBody')
    point, quat = get_pose(braiser)
    r, p, y = euler_from_quat(quat)
    lid.set_pose((point, quat_from_euler((r, p, y + PI / 4))), world=world)
    braiser.attach_obj(lid, world=world)


def sample_full_kitchen(world, w=3, l=8, verbose=True, pause=True, reachability_check=True):
    h_lower_cabinets = 1
    dh_cabinets = 0.8
    h_upper_cabinets = 0.768
    wall_height = h_lower_cabinets + dh_cabinets + h_upper_cabinets + COUNTER_THICKNESS

    floor = create_house_floor(world, w=w, l=l, x=w/2, y=l/2)

    ordering = sample_kitchen_furniture_ordering()
    while 'SinkBase' not in ordering:
        ordering = sample_kitchen_furniture_ordering()

    """ step 1: sample a sink """
    start = ordering.index('SinkBase')
    sink_y = l * start / len(ordering) + np.random.normal(0, 0.5)
    floor, base, counter_x, counter_w, counter_z, color, counters = \
        sample_kitchen_sink(world, floor=floor, y=sink_y)

    under_counter = ['SinkBase', 'CabinetLower', 'DishwasherBox']
    on_base = ['MicrowaveHanging', 'MiniFridge']
    full_body = ['CabinetTall', 'Fridge', 'OvenCounter']
    tall_body = ['CabinetTall', 'Fridge', 'MiniFridge']

    def update_x_lower(obj, x_lower):
        if obj.aabb().lower[0] < x_lower:
            x_lower = obj.aabb().lower[0]
        return x_lower

    def load_furniture(category):
        ins = True
        if world.note in [551, 552]:
            if category == 'MiniFridge':
                ins = random.choice(['11178', '11231'])  ## two doors
            if category == 'CabinetTop':
                ins = random.choice(['00003'])  ## two doors
            if category == 'Sink':
                ins = random.choice(['00003'])  ## two doors
        if world.note in [553]:
            if category == 'OvenCounter':
                ins = random.choice(['101921'])  ## two doors
        if world.note in [555]:
            if category == 'MiniFridge':
                ins = random.choice(['11709'])  ## two doors
        return world.add_object(Object(
            load_asset(category, yaw=math.pi, floor=floor, RANDOM_INSTANCE=ins, verbose=True),
            name=category, category=category))

    def load_furniture_base(furniture):
        return world.add_object(Object(
            load_asset('MiniFridgeBase', l=furniture.ly, yaw=math.pi, floor=floor,
                       RANDOM_INSTANCE=True, verbose=True),
            name=f'{furniture.category}Base', category=f'{furniture.category}Base'))

    counter_regions = []
    tall_obstacles = []
    right_counter_lower = right_counter_upper = base.aabb().upper[1]
    left_counter_lower = left_counter_upper = base.aabb().lower[1]
    x_lower = base.aabb().lower[0]

    adjust_y = {}

    """ step 2: on the left and right of sink base, along with the extended counter """
    for direction in ['+y', '-y']:
        if direction == '+y':
            categories = [c for c in ordering[start+1:]]
        else:
            categories = [c for c in ordering[:start]][::-1]
        current = base
        for category in categories:
            adjust = {}  ## doors bump into neighbors and counters
            furniture = load_furniture(category)
            if category in tall_body:
                tall_obstacles.append(furniture)

            if category in full_body + on_base:
                if direction == '+y' and right_counter_lower != right_counter_upper:
                    counter_regions.append([right_counter_lower, right_counter_upper])
                elif direction == '-y' and left_counter_lower != left_counter_upper:
                    counter_regions.append([left_counter_lower, left_counter_upper])

            ## x_lower aligns with the counter and under the counter
            if category in under_counter + full_body:
                adjust = furniture.adjust_next_to(current, direction=direction, align='+x')

            ## put a cabinetlower with the same y_extent as the object
            elif category in on_base:
                if furniture.mobility_id not in ['11709']:
                    furniture_base = load_furniture_base(furniture)
                    adjust = furniture_base.adjust_next_to(current, direction=direction, align='+x')
                    furniture.adjust_next_to(furniture_base, direction='+z', align='+x')
                    x_lower = update_x_lower(furniture_base, x_lower)
                else:
                    adjust = furniture.adjust_next_to(current, direction=direction, align='+x')
                    # counters.append(furniture)
                    world.add_to_cat(furniture.body, 'supporter')
                x_lower = update_x_lower(furniture, x_lower)

            adjust_y.update(adjust)

            if direction == '+y':
                right_counter_upper = furniture.aabb().upper[1]
            else:
                left_counter_lower = furniture.aabb().lower[1]

            x_lower = update_x_lower(furniture, x_lower)
            current = furniture
            if category in full_body + on_base:
                if direction == '+y':
                    right_counter_lower = right_counter_upper
                else:
                    left_counter_upper = left_counter_lower
            # if direction == '+y':
            #     right_most = furniture.aabb().upper[1]
            # else:
            #     left_most = furniture.aabb().lower[1]
    if right_counter_lower != right_counter_upper:
        counter_regions.append([right_counter_lower, right_counter_upper])
    if left_counter_lower != left_counter_upper:
        counter_regions.append([left_counter_lower, left_counter_upper])

    ## adjust counter regions
    new_counter_regions = []
    for lower, upper in counter_regions:
        original = [lower, upper]
        if lower in adjust_y:
            lower = adjust_y[lower]
        if upper in adjust_y:
            upper = adjust_y[upper]
        new_counter_regions.append([lower, upper])
    counter_regions = new_counter_regions

    ## make doors easier to open
    world.name_to_object('minifridge').adjust_pose(dx=0.2, world=world)

    ## make wall
    l = right_counter_upper - left_counter_lower
    y = (right_counter_upper + left_counter_lower) / 2
    x = x_lower - WALL_WIDTH / 2
    wall = world.add_object(
        Supporter(create_box(w=WALL_WIDTH, l=l, h=wall_height, color=color), name='wall'),
        Pose(point=Point(x=x, y=y, z=wall_height/2)))
    floor.adjust_pose(dx=x_lower - WALL_WIDTH, world=world)

    """ step 3: make all the counters """
    sink_left = world.name_to_object('sink_counter_left')
    sink_right = world.name_to_object('sink_counter_right')

    def could_connect(y1, y2, adjust_y):
        if equal(y1, y2):
            return True
        result1 = in_list(y1, adjust_y)
        if result1 is not None:
            if equal(adjust_y[result1], y2):
                return True
        result2 = in_list(y2, adjust_y)
        if result2 is not None:
            if equal(adjust_y[result2], y1):
                return True
        return False

    for lower, upper in counter_regions:
        name = 'counter'
        if could_connect(lower, sink_right.aabb().upper[1], adjust_y):
            name = 'sink_counter_right'
            lower = sink_right.aabb().lower[1]
            counters.remove(sink_right)
            world.remove_object(sink_right)
        elif could_connect(upper, sink_left.aabb().lower[1], adjust_y):
            name = 'sink_counter_left'
            upper = sink_left.aabb().upper[1]
            counters.remove(sink_left)
            world.remove_object(sink_left)
        counters.append(world.add_object(
            Supporter(create_box(w=counter_w, l=upper-lower,
                                 h=COUNTER_THICKNESS, color=color), name=name),
            Pose(point=Point(x=counter_x, y=(upper + lower) / 2, z=counter_z))))
        # print('lower, upper', (round(lower, 2), round(upper, 2)))

    ## to cover up the wide objects at the back
    if x_lower < base.aabb().lower[0]:
        x_upper = base.aabb().lower[0]
        x = (x_upper+x_lower)/2
        counter_regions.append([base.aabb().lower[1], base.aabb().upper[1]])

        ## merge those could be merged
        counter_regions = sorted(counter_regions, key=lambda x: x[0])
        merged_counter_regions = [counter_regions[0]]
        for i in range(1, len(counter_regions)):
            if could_connect(counter_regions[i][0], merged_counter_regions[-1][1], adjust_y):
                merged_counter_regions[-1][1] = counter_regions[i][1]
            else:
                merged_counter_regions.append(counter_regions[i])

        for lower, upper in merged_counter_regions:
            world.add_object(
                Object(create_box(w=x_upper - x_lower, l=upper - lower,
                                  h=COUNTER_THICKNESS, color=color),
                       name='counter_back', category='filler'),
                Pose(point=Point(x=x, y=(upper + lower) / 2, z=counter_z)))
            # print('lower, upper', (round(lower, 2), round(upper, 2)))

    """ step 4: put upper cabinets and shelves """
    oven = world.name_to_object('OvenCounter')
    cabinets, shelves = load_full_kitchen_upper_cabinets(world, counters, x_lower, left_counter_lower,
                                                         right_counter_upper, others=[oven],
                                                         dz=dh_cabinets, obstacles=tall_obstacles)

    """ step 5: add additional surfaces in furniture """
    sink = world.name_to_object('sink')
    sink_bottom = world.add_surface_by_keyword(sink, 'sink_bottom')

    """ step 5: place electronics and cooking appliances on counters """
    only_counters = [c for c in counters]
    obstacles = []
    microwave = None
    if 'MicrowaveHanging' not in ordering:
        wide_counters = [c for c in counters if c.ly > 0.66]
        if len(wide_counters) > 0:
            counter = wide_counters[0]
            microwave = counter.place_new_obj('microwave', scale=0.4 + 0.1 * random.random(),
                                              RANDOM_INSTANCE=True, verbose=True, world=world)
            microwave.set_pose(Pose(point=microwave.get_pose()[0], euler=Euler(yaw=math.pi)), world=world)
            obstacles.append(microwave)
    else:
        microwave = world.name_to_object('MicrowaveHanging')
    # if microwave is not None:
    #     counters.append(microwave)
    #     world.add_to_cat(microwave.body, 'supporter')

    x_food_min = base.aabb().upper[0] - 0.3
    braiser, braiser_bottom = load_braiser(world, oven, x_min=x_food_min)
    obstacles.extend([braiser, braiser_bottom])

    """ step 5: place movables on counters """
    all_counters = {
        'food': counters,
        'bottle': counters, ##  + [sink_bottom],
        'medicine': shelves + [microwave],
    }
    possible = []
    for v in all_counters.values():
        possible.extend(v)

    ## draw boundary of surfaces
    drawn = []
    for c in possible:
        if c in drawn: continue
        mx, my, z = c.aabb().upper
        aabb = AABB(lower=(x_food_min, c.aabb().lower[1], z), upper=(mx, my, z + 0.1))
        draw_aabb(aabb)
        drawn.append(str(c))

    ## probility of each door being open
    world.make_doors_transparent()
    epsilon = 0
    load_storage_mechanism(world, world.name_to_object('minifridge'), epsilon=epsilon)
    for cabi_type in ['cabinettop', 'cabinetupper']:
        cabi = world.cat_to_objects(cabi_type)
        if len(cabi) > 0:
            cabi = world.name_to_object(cabi_type)
            load_storage_mechanism(world, cabi, epsilon=epsilon)

    ## load objects into reachable places
    food_ids, bottle_ids, medicine_ids = \
        load_counter_moveables(world, all_counters, d_x_min=0.3, obstacles=obstacles,
                               reachability_check=reachability_check)
    moveables = food_ids + bottle_ids + medicine_ids

    """ step 6: take an image """
    set_camera_pose((4, 4, 3), (0, 4, 0))

    # pause = True
    if pause:
        wait_unlocked()
    return moveables, cabinets, only_counters, obstacles, x_food_min


def make_sure_obstacles(world, case, moveables, counters, objects, food=None):
    assert case in [
        2, ## to_sink
        3, ## to_braiser
        992, ## to_sink (no obstacle)
        993, ## to_braiser (no obstacle)
        21, ## sink_to_storage
        31, ## braiser_to_storage
    ]
    cammies = {
        2: ('sink', 'sink_bottom', [0.1, 0.0, 0.8]),
        3: ('braiserbody', 'braiser_bottom', [0.2, 0.0, 1.3])
    }

    if '2' in str(case):
        obj_name, surface_name, d = cammies[2]
    elif '3' in str(case):
        obj_name, surface_name, d = cammies[3]

    """ add camera to the from or to region """
    obj = world.name_to_body(obj_name)
    obj_bottom = world.name_to_body(surface_name)
    set_camera_target_body(obj, dx=d[0], dy=d[1], dz=d[2])
    world.planning_config['camera_zoomins'].append(
        {'name': world.BODY_TO_OBJECT[obj].name, 'd': d}
    )

    """ add obstacles to the from or to region """
    bottom_obj = world.BODY_TO_OBJECT[obj_bottom]
    obstacles = bottom_obj.supported_objects
    start = time.time()
    time_allowed = 4
    while (case in [2, 21] and len(obstacles) <= 2 and random.random() < 1) \
            or (case in [3] and len(obstacles) == 0 and random.random() < 1):
        all_to_move = []

        """ add one goal unrelated obstacles """
        existing = [o.body for o in obstacles]
        something = None
        if case in [2]:
            something = [m for m in moveables if m.body not in existing]
        elif case in [3]:
            something = world.cat_to_objects('edible') + world.cat_to_objects('medicine')
            something = [m for m in something if m.body not in existing]
        if food is not None:
            something = [m for m in something if m.body != food]

        """ may add some goal related objects """
        if case in [2, 3] and something is not None:
            all_to_move = random.sample(something, 2)
        elif case in [21]:
            all_to_move = []
            all_to_move += random.sample(world.cat_to_objects('edible'), 2 if random.random() < 0.5 else 1)
            all_to_move += random.sample(world.cat_to_objects('bottle'), 2 if random.random() < 0.5 else 1)

        """ move objects to the clustered region """
        for something in all_to_move:
            pkwargs = dict(max_trial=5, world=world, obstacles=obstacles)
            result = bottom_obj.place_obj(something, **pkwargs)
            if result is not None:
                obstacles.append(something)
                obstacles += bottom_obj.supported_objects
            ## move objects away if there is no space
            else:
                counters_tmp = world.find_surfaces_for_placement(something, counters)
                counters_tmp[0].place_obj(something, **pkwargs)

        if time.time() - start > time_allowed:
            sys.exit()

    """ make sure obstacles can be moved away """
    for o in obstacles:
        world.add_to_cat(o, 'moveable')
        # skeleton.extend([(k, arm, o) for k in pick_place_actions])
        # goals.append(('On', o.body, random.choice(sink_counters)))
        # goals = [('Holding', arm, o.body)]
        objects.append(o)
    # skeleton.extend([(k, arm, food) for k in pick_place_actions])

    """ choose the object to rearrange """
    foods = world.cat_to_bodies('edible')
    if food is None:
        food = foods[0]

    """ maybe move the lid away """
    if case in [3, 31]:
        """ make sure food is smaller than the braiser """
        epsilon = 0 if case in [3] else 0.3
        if case in [3]:
            count = 0
            while not aabb_larger(obj_bottom, food):
                count += 1
                food = foods[count]
        lid = world.name_to_body('braiserlid')
        objects += [food, lid]
        world.add_to_cat(lid, 'moveable')
        counters_tmp = move_lid_away(world, counters, epsilon=epsilon)
        if counters_tmp is not None:
            objects += [counters_tmp.pybullet_name]
    elif case in [993]:
        world.remove_object(world.name_to_object('braiserlid'))

    return food, obj_bottom, objects


######################################################################################


def sample_table_plates(world, verbose=True):
    """ a table facing the kitchen counters, with four plates on it """
    x = random.uniform(3, 3.5)
    y = random.uniform(2, 6)
    table = sample_table(world, x=x, y=y, verbose=verbose)
    plates = load_plates_on_table(world, table, verbose)
    return table, plates


def sample_two_tables_plates(world, verbose=True):
    """ two tables side by side facing the kitchen counters, with four plates on each """
    x = random.uniform(3, 3.5)
    y1 = random.uniform(1, 3)

    table1 = sample_table(world, x=x, y=y1, verbose=verbose)
    plates = load_plates_on_table(world, table1, verbose)

    table2 = sample_table(world, x=x, y=y1, verbose=verbose)
    table2.adjust_next_to(table1, direction='+y', align='+x')
    plates += load_plates_on_table(world, table2, verbose)

    return [table1, table2], plates


def sample_table(world, RANDOM_INSTANCE=True, **kwargs):
    """ a table facing the kitchen counters, x < 4 """
    floor = world.name_to_body('floor')
    table = world.add_object(Supporter(
        load_asset('DiningTable', yaw=0, floor=floor, RANDOM_INSTANCE=RANDOM_INSTANCE, **kwargs)))
    if table.aabb().upper[0] > 4:
        table.adjust_pose(x=4 - table.xmax2x)
    return table


def load_plates_on_table(world, table, verbose):
    """ sample a series of poses for plates """
    plate = world.add_object(Supporter(load_asset('Plate', yaw=0, verbose=verbose, floor=table.body)))

    ## dist from edge of table to center of plates
    x_gap = random.uniform(0.05, 0.1)
    x = table.aabb().lower[0] + (plate.x2xmin + x_gap)
    y_min = table.aabb()[0][1]
    # draw_aabb(table.aabb())

    width = table.ly
    num = min(4, int(width / plate.ly))
    fixed = [(i+0.5) * (width / num) + y_min for i in range(num)]
    plate.adjust_pose(x=x, y=fixed[0])
    # draw_aabb(plate.aabb())

    plates = [plate]
    for i in range(1, len(fixed)):
        plate = world.add_object(Supporter(load_asset('Plate', yaw=0, verbose=verbose, floor=table.body)))
        plate.adjust_pose(x=x, y=fixed[i])
        plates.append(plate)
    return plates


######################################################################################


if __name__ == '__main__':
    ordering = sample_kitchen_furniture_ordering()
