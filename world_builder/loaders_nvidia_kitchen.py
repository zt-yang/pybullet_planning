import pickle
import random
from collections import defaultdict

from os.path import join

import numpy as np

from pybullet_tools.pr2_primitives import Conf, get_group_joints
from pybullet_tools.utils import invert, get_name, pairwise_collision, sample_placement_on_aabb, \
    get_link_pose, get_pose, set_pose, sample_placement, aabb_from_extent_center, RED, BLUE, YELLOW, GREEN
from pybullet_tools.pose_utils import sample_pose, xyzyaw_to_pose, sample_center_top_surface
from pybullet_tools.bullet_utils import nice, collided, equal
from pybullet_tools.logging_utils import print_dict
from pybullet_tools.general_streams import Position

from world_builder.paths import DATABASES_PATH
from world_builder.world_utils import sort_body_indices, RIGHT, LEFT, ABOVE, BELOW
from world_builder.loaders import *

part_names = {
    'sektion': 'cabinet',
    'chewie_door_left_joint': 'cabinet left door',
    'chewie_door_right_joint': 'cabinet right door',
    'indigo_drawer_top': 'top drawer space',
    'indigo_drawer_top_joint': 'top drawer',
    'indigo_tmp': 'counter top on the right',
    'hitman_countertop': 'counter top on the left',
    'braiserlid': 'pot lid',
    'braiserbody': 'pot body',
    'braiser_bottom': 'pot bottom',
    'front_left_stove': 'stove on the left',
    'front_right_stove': 'stove on the right',
    'knob_joint_2': 'stove knob on the right',
    'knob_joint_3': 'stove knob on the left',
    'shelf_top': 'fridge shelf',
    'joint_faucet_0': 'faucet handle',
    'basin_bottom': 'kitchen sink'
}

## used by make_camera_image_with_object_labels() in world_builder.world_utils.py
object_label_locations = {
    'pot lid': RIGHT,
    'pot body': BELOW,
    'top drawer': BELOW,
    'cabinet right door': BELOW,
    'counter top on the left': ABOVE,
    'counter top on the right': ABOVE,
    'stove knob on the left': ABOVE,
    'stove knob on the right': RIGHT,
    'stove on the left': LEFT,
    'stove on the right': ABOVE,
    'chicken leg': BELOW,
    'fridge shelf': ABOVE,
    'salt shaker': BELOW,
    'pepper shaker': ABOVE,
    'robot': ABOVE,
    'faucet handle': ABOVE,
    'kitchen sink': BELOW
}

###############################################################################

default_supports = [
    # ['appliance', 'microwave', True, 'microwave', 'hitman_tmp'],
    ['food', 'MeatTurkeyLeg', True, 'chicken-leg', 'shelf_top'],
    ['food', 'VeggieCabbage', True, 'cabbage', 'upper_shelf'],  ## ['shelf_bottom', 'indigo_drawer_top]
    ['condiment', 'Salter', '3934', 'salt-shaker', 'sektion'],
    ['condiment', 'Salter', '5861', 'pepper-shaker', 'sektion'],
    # ['utensil', 'PotBody', True, 'pot', 'indigo_tmp'],
    ['utensil', 'KitchenFork', True, 'fork', 'upper_shelf'],  ## ['indigo_drawer_top]
    # ['utensil', 'KitchenKnife', True, 'knife', 'indigo_drawer_top'],
]

saved_joints = [
    ('counter', ['chewie_door_left_joint', 'chewie_door_right_joint'], 1.4, 'sektion'),
    # ('counter', ['dagger_door_left_joint', 'dagger_door_right_joint'], 1, 'dagger'),
    # ('counter', ['indigo_door_left_joint', 'indigo_door_right_joint'], 1, 'indigo_tmp'),
    ('counter', ['indigo_drawer_top'], 1, 'indigo_drawer_top'),
    ('fridge', ['fridge_door'], 0.7, 'shelf_bottom'),  ## 0.5 if you want to close it
    ('fridge', ['fridge_door'], 0.7, 'shelf_top'),  ## 0.5 if you want to close it
    ('dishwasher', ['dishwasher_door'], 1, 'upper_shelf'),
]


def in_case_hyphened_names(tuple_to_pose):
    added = {}
    for key, value in tuple_to_pose.items():
        obj = key[0]
        if '-' in obj:
            new_key = tuple([obj.replace('-', '')] + list(key[1:]))
            added[new_key] = value
    tuple_to_pose.update(added)


saved_relposes = {
    ('fork', 'indigo_drawer_top'): ((0.141, -0.012, -0.033), (0.0, 0.0, 0.94, 0.34)),
    ('fork', 'upper_shelf'): ((1.051, 6.288, 0.42), (0.0, 0.0, 0.94, 0.338)),
    ('cabbage', 'upper_shelf'): ((-0.062, 0.182, -0.256), (-0.64, 0.301, 0.301, 0.64)),
    ('cabbage', 'indigo_drawer_top'): ((0.115, -0.172, 0.004), (0.0, 0.0, 0.173, 0.985)),
    ('braiserbody', 'front_left_stove'): ((0.0, 0.0, 0.0921), (0.0, 0.0, 0.7071, 0.7071)),
    ('braiserbody', 'front_right_stove'): ((0.0, 0.0, 0.0921), (0.0, 0.0, 0.7071, 0.7071)),
}
in_case_hyphened_names(saved_relposes)

saved_poses = {
    ('pot', 'indigo_tmp'): ((0.63, 8.88, 0.11), (0.0, 0.0, -0.68, 0.73)),
    ('microwave', 'hitman_tmp'): ((0.43, 6.38, 1.0), (0.0, 0.0, 1.0, 0)),
    ('vinegar-bottle', 'sektion'): ((0.75, 7.41, 1.24), (0.0, 0.0, 0.0, 1.0)), ## ((0.75, 7.3, 1.24), (0, 0, 0, 1)),
    ('vinegar-bottle', 'dagger'): ((0.45, 8.83, 1.54), (0.0, 0.0, 0.0, 1.0)),
    ('vinegar-bottle', 'indigo_tmp'): ((0.59, 8.88, 0.16), (0.0, 0.0, 0.0, 1.0)),
    ('vinegar-bottle', 'shelf_bottom'): ((0.64, 4.88, 0.901), (0.0, 0.0, 0.0, 1.0)),
    # ('chicken-leg', 'indigo_tmp'): ((0.717, 8.714, 0.849), (0.0, -0.0, -0.99987, 0.0163)), ## grasp the meat end
    ('chicken-leg', 'indigo_tmp'): ((0.787, 8.841, 0.849), (0.0, 0.0, 0.239, 0.971)), ## grasp the bone end
    # ('chicken-leg', 'shelf_bottom'): ((0.654, 5.062, 0.797), (0.0, 0.0, 0.97, 0.25)),
    ('chicken-leg', 'shelf_bottom'): ((0.654, 4.846, 0.794), (0.0, 0.0, -0.083, 0.997)),  ## regrasp side pose
    ('chicken-leg', 'shelf_top'): ((0.654, 4.846, 1.384), (-0.0, 0.0, -0.182, 0.983)),
    ('cabbage', 'shelf_bottom'): ((0.668, 4.862, 0.83), (0, 0, 0.747, 0.665)),
    # ('cabbage', 'upper_shelf'): ((1.006, 6.295, 0.461), (0.0, 0.0, 0.941, 0.338)),
    # ('cabbage', 'indigo_drawer_top'): ((1.12, 8.671, 0.726), (0.0, 0.0, 0.173, 0.985)),
    ('salt-shaker', 'sektion'): ((0.771, 7.071, 1.152), (0.0, 0.0, 1.0, 0)),
    ('pepper-shaker', 'sektion'): ((0.764, 7.303, 1.164), (0.0, 0.0, 1.0, 0)),
    ('fork', 'indigo_tmp'): ((0.767, 8.565, 0.842), (0.0, 0.0, 0.543415, 0.8395)),
    ('braiserbody', 'basin_bottom'): ((0.538, 5.652, 0.876), (0.0, 0.0, 0.993, 0.12)),
}
in_case_hyphened_names(saved_poses)

saved_base_confs = {
    ('cabbage', 'shelf_bottom', 'left'): [
        (1.54, 4.693, 0.49, 2.081),
        ((1.353, 4.55, 0.651, 1.732), {0: 1.353, 1: 4.55, 2: 1.732, 17: 0.651, 61: 0.574, 62: 0.996, 63: -0.63, 65: -0.822, 66: -2.804, 68: -1.318, 69: -2.06}),
        ((1.374, 4.857, 0.367, -3.29), {0: 1.374, 1: 4.857, 2: -3.29, 17: 0.367, 61: 0.18, 62: 0.478, 63: 0.819, 65: -0.862, 66: 2.424, 68: -1.732, 69: 4.031}),
        ((1.43, 5.035, 0.442, -3.284), {0: 1.43, 1: 5.035, 2: -3.284, 17: 0.442, 61: 0.195, 62: 0.147, 63: 2.962, 65: -0.34, 66: 0.2, 68: -1.089, 69: 3.011}),
        ((1.369, 5.083, 0.366, 2.76), {0: 1.369, 1: 5.083, 2: 2.76, 17: 0.366, 61: 0.547, 62: -0.309, 63: 3.004, 65: -1.228, 66: 0.213, 68: -0.666, 69: -3.304}),
    ],
    ('cabbage', 'upper_shelf', 'left'): [
        ((1.2067, 5.65, 0.0253, 1.35), {0: 1.2067001768469119, 1: 5.649899504044555, 2: 1.34932216824778, 17: 0.025273033185228437, 61: 0.5622022164634521, 62: -0.11050738689251488, 63: 2.3065452971538147, 65: -1.382707387923262, 66: 1.2637544829338638, 68: -0.8836125960523644, 69: 2.9453111543097945}),
        ((1.166, 5.488, 0.222, 0.475), {0: 1.166, 1: 5.488, 2: 0.475, 17: 0.222, 61: 1.106, 62: 0.172, 63: 3.107, 65: -0.943, 66: 0.077, 68: -0.457, 69: -2.503}),
        ((1.684, 6.371, 0.328, 1.767), {0: 1.684, 1: 6.371, 2: 1.767, 17: 0.328, 61: 1.422, 62: 0.361, 63: 3.127, 65: -1.221, 66: 2.251, 68: -0.018, 69: 1.634}),
        ((1.392, 5.57, 0.141, 0.726), {0: 1.392, 1: 5.57, 2: 0.726, 17: 0.141, 61: 1.261, 62: -0.043, 63: 2.932, 65: -1.239, 66: 0.526, 68: -0.427, 69: 0.612}),
        ((1.737, 6.033, 0.279, -5.162), {0: 1.737, 1: 6.033, 2: -5.162, 17: 0.279, 61: 1.966, 62: 0.37, 63: 2.515, 65: -1.125, 66: -4.711, 68: -0.579, 69: 2.463}),
        ((1.666, 6.347, 0.404, 1.66), {0: 1.666, 1: 6.347, 2: 1.66, 17: 0.404, 61: 1.22, 62: 0.656, 63: 3.653, 65: -0.948, 66: -1.806, 68: -0.41, 69: 0.355}),
    ],
    # ('fork', 'indigo_drawer_top'): [
    #     ((1.718, 8.893, 0.0, -2.825), {0: 1.718, 1: 8.893, 2: -2.825, 17: 0.0, 61: -0.669, 62: 0.358, 63: -0.757, 65: -0.769, 66: 3.234, 68: -0.274, 69: -0.961}),
    #     ((1.718, 8.324, 0.439, 1.462), {0: 1.718, 1: 8.324, 2: 1.462, 17: 0.439, 61: 1.081, 62: 0.608, 63: 1.493, 65: -0.693, 66: 2.039, 68: -1.16, 69: -2.243}),
    #     ((1.702, 9.162, 0.002, -2.329), {0: 1.702, 1: 9.162, 2: -2.329, 17: 0.002, 61: -0.59, 62: 0.137, 63: 0.951, 65: -0.15, 66: 0.185, 68: -0.112, 69: 0.449}),
    # ],
    # ('fork', 'indigo_tmp', 'left'): [
    #     ((1.273, 8.334, 0.15, 0.881), {0: 1.273, 1: 8.334, 2: 0.881, 17: 0.15, 61: 1.009, 62: 1.396, 63: 0.382, 65: -1.986, 66: -1.594, 68: -1.654, 69: -2.563}),
    #     ((1.493, 8.18, 0.039, 1.83), {0: 1.493, 1: 8.18, 2: 1.83, 17: 0.039, 61: 0.19, 62: 0.083, 63: 2.022, 65: -0.357, 66: 2.77, 68: -1.454, 69: 2.908}),
    #     ((1.837, 8.537, 0.0, 1.649), {0: 1.837, 1: 8.537, 2: 1.649, 17: 0.0, 61: 0.586, 62: 0.407, 63: -0.782, 65: -0.921, 66: -1.597, 68: -0.398, 69: -2.335}),
    #     ((1.307, 8.365, 0.293, 1.613), {0: 1.307, 1: 8.365, 2: 1.613, 17: 0.293, 61: 1.063, 62: 0.265, 63: 0.077, 65: -0.412, 66: -3.217, 68: -1.716, 69: -2.743}),
    # ],
    # ('chicken-leg', 'indigo_tmp', 'left'):  [  ## grasp the meat end
    #     ((1.785, 8.656, 0.467, 0.816), {0: 1.785, 1: 8.656, 2: 0.816, 17: 0.467, 61: 2.277, 62: 0.716, 63: -0.8, 65: -0.399, 66: 1.156, 68: -0.468, 69: -0.476}),
    #     ((1.277, 8.072, 0.507, 1.023), {0: 1.277, 1: 8.072, 2: 1.023, 17: 0.507, 61: 0.939, 62: 0.347, 63: 2.877, 65: -0.893, 66: 2.076, 68: -1.644, 69: 0.396}),
    # ],
    ('chicken-leg', 'indigo_tmp', 'right'):  [  ## grasp the bone end
        ((1.951, 8.831, 0.122, -1.558), {0: 1.951, 1: 8.831, 2: -1.558, 17: 0.122, 40: -1.71, 41: 0.38, 42: 0.027, 44: -0.706, 45: -2.012, 47: -0.696, 48: -1.049}),
        ((1.835, 8.625, 0.088, -2.017), {0: 1.835, 1: 8.625, 2: -2.017, 17: 0.088, 40: -1.725, 41: -0.281, 42: -2.296, 44: -0.99, 45: 1.503, 47: -0.511, 48: -2.462}),
    ],
    ('chicken-leg', 'indigo_tmp', 'left'):  [  ## grasp the bone end
        ((1.557, 9.315, 0.0, -2.741), {0: 1.557, 1: 9.315, 2: -2.741, 17: 0.0, 61: -0.364, 62: 0.293, 63: -0.567, 65: -1.113, 66: -2.77, 68: -0.665, 69: 0.258}),
    ],
    ('chicken-leg', 'shelf_bottom', 'left'):  [
        ((1.473, 4.787, 0.502, 2.237), {0: 1.473, 1: 4.787, 2: 2.237, 17: 0.502, 61: 0.232, 62: 0.601, 63: 3.858, 65: -0.465, 66: 2.487, 68: -0.943, 69: -0.248}),
    ],
    ('pepper-shaker', 'sektion', 'left'): [
        ((1.619, 7.741, 0.458, -3.348), {0: 1.619, 1: 7.741, 2: -3.348, 17: 0.458, 61: 0.926, 62: 0.187, 63: 1.498, 65: -0.974, 66: 3.51, 68: -0.257, 69: 1.392}),
    ],
    ('braiserbody', 'basin_bottom', 'left'): [
        ((1.0, 5.62, 0.796, 1.599), {0: 1.0, 1: 5.62, 2: 1.599, 17: 0.796, 61: 2.117, 62: 0.964, 63: 0.968, 65: -0.752, 66: 2.609, 68: -1.182, 69: -1.651}),
        ((1.066, 5.78, 0.479, 2.249), {0: 1.066, 1: 5.78, 2: 2.249, 17: 0.479, 61: 1.08, 62: 0.573, 63: -0.321, 65: -0.87, 66: -2.864, 68: -1.835, 69: -1.13}),
        ((1.24, 5.551, 0.417, 1.287), {0: 1.24, 1: 5.551, 2: 1.287, 17: 0.417, 61: 2.065, 62: 0.084, 63: 1.985, 65: -0.086, 66: 1.165, 68: -1.453, 69: 4.844}),
    ],
}
in_case_hyphened_names(saved_base_confs)

FRONT_CAMERA_POINT = (3.9, 7, 1.3)
DOWNWARD_CAMERA_POINT = (2.9, 7, 3.3)

#######################################################################################################


def load_kitchen_floor_plan(world, spaces=None, surfaces=None, **kwargs):
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
                'faucet_platform': ['Faucet']  ##
            }
        }
    return load_floor_plan(world, spaces=spaces, surfaces=surfaces, **kwargs)


def load_full_kitchen(world, load_cabbage=True, **kwargs):
    world.set_skip_joints()

    if world.robot is None:
        custom_limits = ((0, 4), (4, 13))
        robot = create_pr2_robot(world, base_q=(1.79, 6, PI / 2 + PI / 2),
                                 custom_limits=custom_limits, use_torso=True)

    floor = load_kitchen_floor_plan(world, plan_name='kitchen_v2.svg', **kwargs)
    world.remove_object(floor)
    world.set_camera_points(FRONT_CAMERA_POINT, DOWNWARD_CAMERA_POINT)

    lid = world.name_to_body('braiserlid')
    world.put_on_surface(lid, 'braiserbody')

    if load_cabbage:
        cabbage = load_experiment_objects(world, CABBAGE_ONLY=True)
        counter = world.name_to_object('hitman_countertop')
        counter.place_obj(cabbage)
        (_, y, z), _ = cabbage.get_pose()
        cabbage.set_pose(Pose(point=Point(x=0.85, y=y, z=z)))
        return cabbage
    return None

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
    world.add_to_cat(lid, 'movable')

    return bottom, lid


def load_basin_faucet(world):
    from .actions import ChangeLinkColorEvent, CreateCylinderEvent
    cold_blue = RGBA(0.537254902, 0.811764706, 0.941176471, 1.)

    name_to_body = world.name_to_body
    name_to_object = world.name_to_object

    basin = world.add_surface_by_keyword('basin', 'basin_bottom')
    set_color(basin.body, GREY, basin.link)

    faucet = name_to_body('faucet')
    handles = world.add_joints_by_keyword('faucet', 'joint_faucet', 'knob')

    # world.summarize_all_objects()

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
    world.remove_body_from_planning(name_to_body('hitman_countertop'))

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


def load_feg_kitchen_dishwasher(world):
    surfaces = {
        'counter': {
            'front_left_stove': [],
            'front_right_stove': ['BraiserBody'],
            # 'back_left_stove': [],
            # 'back_right_stove': [],
            'hitman_countertop': [],
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
    floor = load_kitchen_floor_plan(world, plan_name='kitchen_v3.svg', surfaces=surfaces, spaces=spaces)
    world.remove_object(floor)
    load_kitchen_mechanism(world)
    # load_kitchen_mechanism_stove(world)
    dishwasher_door = world.add_joints_by_keyword('dishwasher', 'dishwasher_door')[0]

    cabbage = world.name_to_body('cabbage')
    chicken = world.name_to_body('turkey')
    for ingredient in [cabbage, chicken]:
        world.add_to_cat(ingredient, 'edible')
        world.add_to_cat(ingredient, 'movable')
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
            'hitman_countertop': [],
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
    floor = load_kitchen_floor_plan(world, plan_name='kitchen_v3.svg', surfaces=surfaces)
    world.remove_object(floor)
    load_kitchen_mechanism(world)
    # load_kitchen_mechanism_stove(world)

    cabbage = world.name_to_body('cabbage')
    turkey = world.name_to_body('turkey')
    for ingredient in [cabbage, turkey]:
        world.add_to_cat(ingredient, 'edible')
        world.add_to_cat(ingredient, 'movable')
    world.put_on_surface(cabbage, 'shelf_bottom')
    world.put_on_surface(turkey, 'indigo_tmp')

    lid = world.name_to_body('lid')
    world.open_joint_by_name('fridge_door', pstn=1.5)
    # world.put_on_surface(lid, 'indigo_tmp')

    world.add_to_cat(turkey, 'cleaned')

    return cabbage, turkey, lid


def studio(args):
    """
    for testing fridge: plan_name = 'fridge.svg', robot pose: x=1.79, y=4.5
    for testing fridge: plan_name = 'kitchen.svg', robot pose: x=1.79, y=8 | 5.5
    for testing planning: plan_name = 'studio0.svg', robot pose: x=1.79, y=8
    """
    world = World(time_step=args.time_step, camera=args.camera, segment=args.segment)

    # floor = load_floor_plan(world, plan_name='studio0.svg')  ## studio1
    floor = load_kitchen_floor_plan(world, plan_name='kitchen.svg')
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


# --------------------------------------------------------------------------------

def load_cooking_mechanism(world):
    load_braiser_bottom(world)
    load_stove_knobs(world)
    define_seasoning(world)
    # load_dishwasher(world)


def load_braiser_bottom(world):
    braiser = world.name_to_body('braiserbody')
    world.add_object(Surface(braiser, link_from_name(braiser, 'braiser_bottom')))
    world.add_to_cat('braiserlid', 'movable')
    world.add_to_cat('braiserbody', 'surface')
    world.add_to_cat('braiserbody', 'region')
    world.add_to_cat('braiserbody', 'space')


braiser_quat = quat_from_euler(Euler(yaw=PI/2))


def fix_braiser_orientation(world):
    braiser = world.name_to_object('braiserbody')
    point, quat = braiser.get_pose()
    braiser.set_pose((point, braiser_quat))


def load_stove_knobs(world, knobs=('knob_joint_2', 'knob_joint_3'), color_code_surfaces=True, draw_label=True):
    colors = [RED, YELLOW, BLUE, GREEN] if color_code_surfaces else [GREY] * 4
    # knobs = ['knob_joint_1', 'knob_joint_2']
    surfaces = ['back_right_stove', 'front_right_stove', 'front_left_stove', 'back_left_stove']

    oven = world.name_to_body('oven')
    for name in knobs:
        i = int(name.split('_')[-1]) - 1
        world.add_joints_by_keyword('oven', name, category='knob')
        knob = world.name_to_object(name)
        # knob.handle_link = link_from_name(oven, name.replace('joint', 'link'))
        if draw_label:
            knob.draw()
        set_color(knob.body, colors[i], link=knob.handle_link)

    braiser = world.name_to_body('braiserbody')
    counter = world.name_to_body('counter')
    for i, name in enumerate(surfaces):
        surface = world.name_to_object(name)
        if surface is None:
            continue
        if draw_label:
            surface.draw()
        set_color(counter, colors[i], link=link_from_name(counter, name))

        # ## consider removing braiserbody when the knob is blocked
        # corresponding_knob = world.name_to_body(f'knob_joint_{i+1}')
        # if corresponding_knob is not None:
        #     world.add_to_relevant_objects(corresponding_knob, braiser)
        #     # world.add_to_init(('HeatableOnSurfaceWhenTurnedOn', braiser, surface.pybullet_name, corresponding_knob))

    ## consider removing braiserbody when the knob is blocked
    if world.name_to_body('braiserbody') is not None:
        world.add_to_cat('braiserbody', 'movable')
        fix_braiser_orientation(world)


def define_seasoning(world):
    for name in ['salt-shaker', 'pepper-shaker']:
        world.add_to_cat(name, 'sprinkler')


def load_dishwasher(world):
    dishwasher_door = world.add_joints_by_keyword('dishwasher', 'dishwasher_door')[0]


# --------------------------------------------------------------------------------


def get_objects_for_open_kitchen(world, difficulty, verbose=False):
    object_names = ['chicken-leg', 'fridge', 'fridge_door', 'shelf_top',
                    'braiserbody', 'braiserlid', 'braiser_bottom',
                    'indigo_tmp', 'hitman_countertop',
                    'sektion', 'chewie_door_left_joint', 'chewie_door_right_joint',
                    'salt-shaker', 'pepper-shaker',
                    'front_left_stove', 'front_right_stove', 'knob_joint_2', 'knob_joint_3',
                    'joint_faucet_0', 'basin_bottom']  ## 'fork', 'indigo_drawer_top', 'indigo_drawer_top_joint',
    if difficulty in [20]:
        for k in ['sektion', 'chewie_door_left_joint', 'chewie_door_right_joint',
                  'indigo_drawer_top', 'indigo_drawer_top_joint',
                  'front_left_stove', 'knob_joint_3']:
            object_names.remove(k)
    objects = [world.name_to_body(name) for name in object_names]
    objects = sort_body_indices(objects)
    world.set_english_names(part_names)
    world.remove_bodies_from_planning([], exceptions=objects)

    if verbose:
        print('reduce_objects_for_open_kitchen')
        print(f"\t{len(objects)} objects provided (A):\t {objects}")
        world_objects = sort_body_indices(list(world.BODY_TO_OBJECT.keys()))
        in_a_not_in_b = [o for o in objects if o not in world_objects]
        in_b_not_in_a = [o for o in world_objects if o not in objects]
        print(f"\t\t in A not in B ({len(in_a_not_in_b)}):\t {in_a_not_in_b}")
        print(f"\t{len(world.BODY_TO_OBJECT.keys())} objects in world (B):\t {world_objects}")
        print(f"\t\t in B not in A ({len(in_b_not_in_a)}):\t {in_b_not_in_a}")
    return objects


def prevent_funny_placements(world, verbose=False):
    """ need to be automated by LLMs """

    movables = world.cat_to_bodies('movable')
    food = world.cat_to_bodies('food')
    condiments = world.cat_to_bodies('condiment')

    ## only the lid can be placed on braiserbody or the front left stove
    ## only food can be placed on braiser bottom, or inside braiserbody
    cabinet = world.name_to_body('sektion')
    braiserbody = world.name_to_body('braiserbody')
    braiserlid = world.name_to_body('braiserlid')

    left_counter = world.name_to_body('hitman_countertop')
    left_stove = world.name_to_body('front_left_stove')
    right_stove = world.name_to_body('front_right_stove')
    braiser_bottom = world.name_to_body('braiser_bottom')
    basin_bottom = world.name_to_body('basin_bottom')
    shelf_top = world.name_to_body('shelf_top')

    for o in movables:
        ## nothing should be moved there during planning
        world.add_not_stackable(o, shelf_top)
        # world.add_not_containable(o, cabinet)

        if o not in food:  ##  + condiments
            world.add_not_stackable(o, braiser_bottom)
            world.add_not_containable(o, braiserbody)

        if o not in food + [braiserlid]:  ##  + condiments
            world.add_not_stackable(o, basin_bottom)

        if o not in condiments + [braiserlid]:
            world.add_not_stackable(o, left_counter)

        if o != braiserlid:
            world.add_not_stackable(o, braiserbody)
            world.add_not_stackable(o, left_stove)
            world.add_not_stackable(o, right_stove)
        else:
            world.add_not_stackable(o, right_stove)

    if verbose:
        world.summarize_forbidden_placements()

    world.add_ignored_pair((braiserbody, braiserlid))


def load_open_problem_kitchen(world, reduce_objects=False, difficulty=1, open_doors_for=[],
                              randomize_joint_positions=True):
    """
    difficulty 20:  seasoning on surfaces, knob is already turned on
    difficulty 0:   all objects in containers, joints all open
    difficulty 1:   all objects in containers, joints all closed
    """
    spaces = {
        'counter': {
            'sektion': [],
            'indigo_drawer_top': [],
        },
        'dishwasher': {
            'upper_shelf': []
        }
    }
    surfaces = {
        'counter': {
            'front_left_stove': [],
            'front_right_stove': ['BraiserLid'],
            'hitman_countertop': [],
            'indigo_tmp': ['BraiserBody'],
        },
        'Basin': {
            'faucet_platform': ['Faucet'],
            'basin_bottom': []
        }
    }
    custom_supports = {
        'cabbage': 'shelf_bottom',
        'fork': 'indigo_drawer_top'
    }
    load_full_kitchen(world, surfaces=surfaces, spaces=spaces, load_cabbage=False)
    movables, movable_to_doors = load_nvidia_kitchen_movables(world, open_doors_for=open_doors_for,
                                                              custom_supports=custom_supports)
    load_cooking_mechanism(world)
    load_basin_faucet(world)
    prevent_funny_placements(world)

    world.set_learned_pose_list_gen(learned_nvidia_pickled_pose_list_gen)
    world.set_learned_bconf_list_gen(learned_nvidia_pickled_bconf_list_gen)
    world.set_learned_position_list_gen(learned_nvidia_pickled_position_list_gen)

    objects = None
    if reduce_objects:
        objects = get_objects_for_open_kitchen(world, difficulty)

    if difficulty == 0:
        for door in world.cat_to_objects('door'):
            extent = 0.6 if 'fridge' in door.name else 0.8
            if randomize_joint_positions:
                extent += (random.random() - 0.5) * 0.1
            world.open_joint(door.body, joint=door.joint, extent=extent, verbose=True)
        world.name_to_object('front_left_stove').place_obj(world.name_to_object('braiserlid'))
        # world.name_to_object('hitman_countertop').place_obj(world.name_to_object('braiserlid'))

    ## seasoning on surfaces, knob is already turned on
    elif difficulty == 20:
        for obj in world.cat_to_objects('sprinkler'):
            world.name_to_object('hitman_countertop').place_obj(obj)
        knob = world.name_to_object('knob_joint_2')
        world.open_joint(knob.body, joint=knob.joint, extent=1, verbose=True)

    return objects, movables


##########################################################


def get_nvidia_kitchen_hacky_pose(obj, supporter_name):
    """ return a pose and whether to interactively adjust """
    if obj is None:
        return None, True
    world = obj.world
    if isinstance(supporter_name, tuple):
        o, _, l = supporter_name
        supporter_name = get_link_name(o, l)
        # supporter_name = world.get_name(supporter_name)

    key = (obj.shorter_name, supporter_name)
    supporter = world.name_to_object(supporter_name)
    if key in saved_relposes:
        link_pose = get_link_pose(supporter.body, supporter.link)
        return multiply(link_pose, saved_relposes[key]), False

    if key in saved_poses:
        return saved_poses[key], False

    """ given surface name may be the full name """
    for kk, pose in saved_poses.items():
        if kk[0].lower == key[0].lower and kk[1] in key[1]:
            return pose, False

    """ same spot on the surface to start with """
    for kk, pose in saved_relposes.items():
        if kk[1] in key[1]:
            link_pose = get_link_pose(supporter.body, supporter.link)
            return multiply(link_pose, pose), True
    for kk, pose in saved_poses.items():
        if kk[1] in key[1]:
            return pose, True

    return None, True


def place_in_nvidia_kitchen_space(obj, supporter_name, interactive=False, doors=[]):
    """ place an object on a supporter in the nvidia kitchen using saved poses for debugging """
    world = obj.world
    supporter = world.name_to_object(supporter_name)
    pose, interactive2 = get_nvidia_kitchen_hacky_pose(obj, supporter_name)
    interactive = interactive or interactive2
    if pose is not None:
        obj.set_pose(pose)
        supporter.attach_obj(obj)
    else:
        ## initialize the object pose
        supporter.place_obj(obj)

    ## adjust the pose by pressing keyboard
    if pose is None or interactive:

        title = f' ==> Putting {obj.shorter_name} on {supporter_name}'

        ## open all doors that may get in the way
        set_camera_target_body(supporter.body, link=supporter.link, dx=1, dy=0, dz=0.5)
        for door, extent in doors:
            world.open_joint(door, extent=extent)

        ## enter the interactive program
        print(f'\n{title} starting at pose\t', nice(get_pose(obj), keep_quat=True))
        obj.change_pose_interactive()
        link_pose = get_link_pose(supporter.body, supporter.link)
        pose_relative = multiply(invert(link_pose), get_pose(obj))
        print(f'{title} ending at relpose\t{nice(pose_relative, keep_quat=True)}\n')

        ## close all doors that have been opened
        for door, extent in doors:
            world.close_joint(door)

    ## check if the object is in collision with the surface
    collided(obj.body, [world.name_to_object(supporter_name).body],
             world=world, verbose=True, tag='place_in_nvidia_kitchen_space', log_collisions=False)


def load_nvidia_kitchen_movables(world: World, open_doors_for: list = [], custom_supports: dict = {}):

    for elems in default_supports:
        if elems[-2] in custom_supports:
            elems[-1] = custom_supports[elems[-2]]

    """ load joints """
    supporter_to_doors = load_nvidia_kitchen_joints(world)

    """ add surfaces """
    for body_name, surface_name in [
        ('fridge', 'shelf_bottom'), ('fridge', 'shelf_top'),
    ]:
        body = world.name_to_body(body_name)
        shelf = world.add_object(Surface(
            body, link=link_from_name(body, surface_name), name=surface_name,
        ))

    # ## left half of the kitchen
    # set_camera_pose(camera_point=[3.5, 9.5, 2], target_point=[1, 7, 1])

    """ load movables """
    movables = {}
    movable_to_doors = {}
    for category, asset_name, rand_ins, name, supporter_name in default_supports:
        object_class = Movable if category not in ['appliance'] else Object
        movable = world.add_object(object_class(
            load_asset(asset_name, x=0, y=0, yaw=random.uniform(-math.pi, math.pi), random_instance=rand_ins),
            category=category, name=name
        ))
        supporting_surface = world.name_to_object(supporter_name)
        if supporting_surface is None:
            print("load_nvidia_kitchen_movables | no supporting surface", supporter_name)
            continue
        movable.supporting_surface = supporting_surface

        doors = supporter_to_doors[supporter_name] if supporter_name in supporter_to_doors else []
        if name in open_doors_for:
            for door, extent in doors:
                world.open_joint(door, extent=extent)

        interactive = name in []  ## 'cabbage'
        place_in_nvidia_kitchen_space(movable, supporter_name, interactive=interactive, doors=doors)

        movables[name] = movable.body
        movable_to_doors[name] = doors

    """ load some smarter samplers for those movables """
    world.set_learned_bconf_list_gen(learned_nvidia_bconf_list_gen)
    world.set_learned_pose_list_gen(learned_nvidia_pose_list_gen)

    return movables, movable_to_doors


def load_nvidia_kitchen_joints(world: World, open_doors: bool = False):

    """ load joints """
    supporter_to_doors = {}
    for body_name, door_names, pstn, supporter_name in saved_joints:
        doors = []
        for door_name in door_names:
            door = world.add_joints_by_keyword(body_name, door_name)[0]
            doors.append((door, pstn))
        supporter_to_doors[supporter_name] = doors

    return supporter_to_doors


#####################################################################################################


def learned_nvidia_pose_list_gen(world, body, surfaces, num_samples=30, obstacles=[], verbose=True):
    if isinstance(surfaces, tuple):
        surfaces = [surfaces]

    ## --------- Special case for plates -------------
    results = check_plate_placement(world, body, surfaces, num_samples=num_samples, obstacles=obstacles)

    name = world.BODY_TO_OBJECT[body].shorter_name
    for surface in surfaces:

        results += check_on_plate_placement(body, surface, world)

        if isinstance(surface, tuple):
            body, _, link = surface
            surface_name = get_link_name(body, link)
        else:
            surface_name = world.BODY_TO_OBJECT[surface].shorter_name
        key = (name, surface_name)

        if key in saved_relposes:
            results.append(saved_relposes[key])
            if verbose:
                print('learned_nvidia_pose_list_gen | found relpose for', key)
        if key in saved_poses:
            results.append(saved_poses[key])
            if verbose:
                print('learned_nvidia_pose_list_gen | found pose for', key)
    return results


def check_plate_placement(world, body, surfaces, obstacles=[], num_samples=30, num_trials=30):
    from pybullet_tools.pr2_primitives import Pose
    surface = random.choice(surfaces)
    poses = []
    trials = 0

    if 'plate-fat' in get_name(body):
        while trials < num_trials:
            y = random.uniform(8.58, 9)
            body_pose = ((0.84, y, 0.88), quat_from_euler((0, math.pi / 2, 0)))
            p = Pose(body, body_pose, surface)
            p.assign()
            if not any(pairwise_collision(body, obst) for obst in obstacles if obst not in {body, surface}):
                poses.append(p)
                # for roll in [-math.pi/2, math.pi/2, math.pi]:
                #     body_pose = (p.value[0], quat_from_euler((roll, math.pi / 2, 0)))
                #     poses.append(Pose(body, body_pose, surface))

                if len(poses) >= num_samples:
                    return [(p,) for p in poses]
            trials += 1
        return []

    if isinstance(surface, int) and 'plate-fat' in get_name(surface):
        aabb = get_aabb(surface)
        while trials < num_trials:
            body_pose = xyzyaw_to_pose(sample_pose(body, aabb))
            p = Pose(body, body_pose, surface)
            p.assign()
            if not any(pairwise_collision(body, obst) for obst in obstacles if obst not in {body, surface}):
                poses.append(p)
                if len(poses) >= num_samples:
                    return [(p,) for p in poses]
            trials += 1

    return []


def check_on_plate_placement(body, surface, world=None, num_samples=30):
    surface_name = world.get_name(surface)
    if surface_name.startswith('counter') and surface_name.endswith('stove'):
        aabb = get_aabb(surface[0], link=surface[-1])
        center = get_aabb_center(aabb)
        w, l, h = get_aabb_extent(aabb)
        enlarged_aabb = aabb_from_extent_center((w*3, l*3, h), center)
        z = sample_placement_on_aabb(body, enlarged_aabb)[0][2]
        return sample_center_top_surface(body, surface, k=num_samples, _z=z, dy=-0.05)
    if 'square_plate' in surface_name:
        return sample_center_top_surface(body, surface, k=num_samples)
    return []


def load_plate_on_counter(world, counter_name='indigo_tmp'):
    plate = world.add_object(Movable(load_asset('plate')))
    world.add_to_cat(plate, 'surface')

    counter = world.name_to_object(counter_name)
    counter.place_obj(plate)
    return plate.body


#####################################################################################################


def learned_nvidia_bconf_list_gen(world, inputs, verbose=True, num_samples=5):
    a, o, p = inputs[:3]
    robot = world.robot
    joints = robot.get_group_joints('base-torso')
    movable = world.BODY_TO_OBJECT[o].shorter_name

    to_return = []
    for key, bqs in saved_base_confs.items():
        if key[0] == movable and key[2] == a:
            results = []
            random.shuffle(bqs)
            for bq in bqs:
                joint_state = None
                if isinstance(bq, tuple) and isinstance(bq[1], dict):
                    bq, joint_state = bq
                results.append(Conf(robot.body, joints, bq, joint_state=joint_state))

            if key in saved_poses and equal(p.value[0], saved_poses[key][0]):
                to_return = results
            if key in saved_relposes:
                relpose = saved_relposes[key]
                if len(inputs) == 4:
                    supporter_pose = world.name_to_object(key[1], include_removed=True).get_pose()
                    if equal(p.value[0], multiply(supporter_pose, relpose)[0]):
                        to_return = results
                else:
                    if equal(p.value[0], relpose[0]):
                        to_return = results
            if len(to_return) > 0:
                if verbose:
                    print('learned_nvidia_bconf_list_gen | found', len(to_return), 'base confs for', key)
                break
    return to_return[:num_samples]


#####################################################################################################


def load_database(world, name):
    dual_arm = '' if not world.robot.dual_arm else 'dual_arm_'
    pickled_path = join(DATABASES_PATH, f'nvidia_kitchen_{dual_arm}{name}.pickle')
    return pickle.load(open(pickled_path, 'rb'))


def learned_nvidia_pickled_position_list_gen(world, joint, p1, num_samples=30, verbose=False):
    if world.learned_position_database is None:
        world.learned_position_database = load_database(world, name='j_to_p')

    results = None
    key = (str(joint), p1.value)
    if key in world.learned_position_database:
        positions = world.learned_position_database[key]
        results = random.sample(positions, min(num_samples, len(positions)))
    elif verbose:
        print(f'learned_nvidia_pickled_position_list_gen({key}) not found in database')
    return results


def learned_nvidia_pickled_pose_list_gen(world, body, surfaces, num_samples=30, obstacles=[], verbose=False):
    if world.learned_pose_database is None:
        world.learned_pose_database = load_database(world, name='or_to_p')
    results = None
    key = (str(body), str(surfaces[0]))
    if key in world.learned_pose_database:
        poses = world.learned_pose_database[key]
        results = random.sample(poses, min(num_samples, len(poses)))
        results = [(f[:3], quat_from_euler(f[3:])) for f in results]
    elif verbose:
        print(f'learned_nvidia_pickled_pose_list_gen({key}) not found in database')
    return results


def learned_nvidia_pickled_bconf_list_gen(world, inputs, num_samples=30, verbose=False):
    from pybullet_tools.pr2_primitives import Pose

    if world.learned_bconf_database is None:
        database_names = ['ajg_to_p_to_q', 'aog_to_p_to_q']
        world.learned_bconf_database = {
            p: load_database(world, name=p) for p in database_names
        }

    robot = world.robot
    joints = robot.get_group_joints('base-torso')
    results = []

    a, o, p, g = inputs[:4]
    if isinstance(g, tuple):
        print(f'learned_nvidia_pickled_bconf_list_gen(g)\t {g}')
    key = (a, str(o), nice(g.value))

    if isinstance(p, Position):
        database = world.learned_bconf_database['ajg_to_p_to_q']

        if key in database:
            p_to_q = database[key]
            if p.value in p_to_q:
                results = p_to_q[p.value]
            elif verbose:
                print(f'learned_nvidia_pickled_ajg_to_p_to_q({p.value}) not found in database')

    if isinstance(p, Pose):
        aog_to_p_to_q = world.learned_bconf_database['aog_to_p_to_q']
        key = (a, str(o), nice(g.value))
        if key in aog_to_p_to_q:
            p_to_q = aog_to_p_to_q[key]
            key2 = nice(p.value)
            if key2 in p_to_q:
                results = p_to_q[key2]
            else:
                keys = np.asarray(list(p_to_q.keys()))
                diff = keys - np.asarray(key2)
                min_index = np.argmin(np.sum(diff*diff, axis=1))
                min_diff = diff[min_index]
                if np.max(np.abs(min_diff)) < 0.1:
                    key_found = list(p_to_q.keys())[min_index]
                    results = p_to_q[key_found]
                elif verbose:
                    print(f'learned_nvidia_pickled_aog_to_p_to_q({key2}) not found in database')

    results = random.sample(results, min(num_samples, len(results)))
    results = [Conf(robot.body, joints, bq) for bq in results]
    return results


## ----------------------------------------------------------------------------------------
## haven't fixed
## ----------------------------------------------------------------------------------------


def load_gripper_test_scene(world):
    surfaces = {
        'counter': {
            'front_left_stove': [],
            'front_right_stove': ['BraiserBody'],
            'hitman_countertop': [],
            'indigo_tmp': ['BraiserLid', 'MeatTurkeyLeg'],  ## , 'VeggieCabbage'
        }
    }

    floor = load_floor_plan(world, plan_name='counter.svg', surfaces=surfaces)
    world.remove_object(floor)
    pot, lid = load_pot_lid(world)
    set_camera_target_body(lid, dx=1.5, dy=0, dz=0.7)

    turkey = world.name_to_body('turkey')
    counter = world.name_to_body('indigo_tmp')

    world.add_to_cat(turkey, 'movable')
    world.add_to_cat(lid, 'movable')

    camera_pose = ((1.7, 6.1, 1.5), (0.5, 0.5, -0.5, -0.5))
    world.add_camera(camera_pose)

    return pot, lid, turkey, counter


def load_cabinet_test_scene(world, random_instance=False, MORE_MOVABLE=False, verbose=True):
    surfaces = {
        'counter': {
            'front_left_stove': [],
            'front_right_stove': ['BraiserBody'],
            'hitman_countertop': [],
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
        surfaces['counter']['hitman_countertop'].append('VeggieCabbage')

    floor = load_floor_plan(world, plan_name='counter.svg', debug=True, verbose=verbose,
                            surfaces=surfaces, spaces=spaces, random_instance=random_instance)
    world.remove_object(floor)
    pot, lid = load_pot_lid(world)

    lid = world.name_to_body('lid')
    pot = world.name_to_body('braiser_bottom')
    turkey = world.name_to_body('turkey')
    counter = world.name_to_body('indigo_tmp')
    oil = world.name_to_body('bottle')
    vinegar = world.name_to_body('vinegarbottle')

    world.add_to_cat(oil, 'movable')
    world.add_to_cat(lid, 'movable')
    world.add_joints_by_keyword('counter', 'chewie_door')
    world.add_joints_by_keyword('counter', 'dagger_door')

    ### ------- more objects
    if MORE_MOVABLE:
        world.add_to_cat(turkey, 'movable')

        veggie = world.name_to_body('veggiecabbage')
        world.add_to_cat(veggie, 'movable')
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
            # 'hitman_countertop': [],
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
    world.add_to_cat(vinegar, 'movable')
    world.add_to_cat(oil, 'movable')
    world.add_to_cat(lid, 'movable')
    world.add_to_cat(turkey, 'movable')
    world.add_to_cat(veggie, 'movable')
    world.add_to_cat(turkey, 'edible')
    world.add_to_cat(veggie, 'edible')

    world.add_joints_by_keyword('counter', 'chewie_door')
    world.add_joints_by_keyword('counter', 'dagger_door')

    return pot, lid, turkey, veggie, counter, oil, vinegar
