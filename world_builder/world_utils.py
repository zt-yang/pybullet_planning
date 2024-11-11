from __future__ import annotations

import time
import numpy as np
import untangle
from numpy import inf
from os.path import join, isdir, isfile, dirname, abspath, basename, split
from os import listdir
import json
import random
import math
import inspect

from pybullet_tools.utils import unit_pose, get_aabb_extent, is_darwin, RED, sample_placement_on_aabb, wait_unlocked, \
    set_pose, get_aabb_center, aabb_from_extent_center, aabb_overlap, sample_placement, \
    CameraImage, dump_joint, dump_body, PoseSaver, get_aabb, add_text, GREEN, AABB, remove_body, HideOutput, \
    stable_z, Pose, Point, create_box, load_model, get_joints, set_joint_position, BROWN, Euler, PI, \
    set_camera_pose, TAN, RGBA, get_color, get_min_limit, get_max_limit, set_color, WHITE, get_links, \
    get_link_name, get_link_pose, euler_from_quat, get_collision_data, get_joint_name, get_joint_position, \
    set_renderer, link_from_name, parent_joint_from_link, set_random_seed, set_numpy_seed
from pybullet_tools.bullet_utils import dist, get_fine_rainbow_colors, open_joint, toggle_joint
from pybullet_tools.camera_utils import get_segmask, get_obj_keys_for_segmentation
from pybullet_tools.logging_utils import dump_json

from lisdf_tools.image_utils import get_seg_foreground_given_obj_keys, get_mask_bb, RAINBOW_COLORS, DARKER_COLORS

from world_builder.asset_constants import DONT_LOAD
from world_builder.paths import ASSET_PATH, DATABASES_PATH


FURNITURE_WHITE = RGBA(0.85, 0.85, 0.85, 1)
FURNITURE_GREY = RGBA(0.4, 0.4, 0.4, 1)
FURNITURE_YELLOW = RGBA(221/255, 187/255, 123/255, 1)

LIGHT_GREY = RGBA(0.5, 0.5, 0.5, 0.6)
DARK_GREEN = RGBA(35/255, 66/255, 0, 1)
FLOOR_HEIGHT = 1e-3
WALL_HEIGHT = 0.5

JOINT = 'joint'
DOOR = 'door'
DRAWER = 'drawer'
KNOB = 'knob'
MOVABLE = 'movable'
GRASPABLE = 'graspable'
SURFACE = 'surface'
SPACE = 'space'

GRASPABLES = ['BraiserLid', 'Egg', 'VeggieCabbage', 'MeatTurkeyLeg', 'VeggieGreenPepper', 'VeggieArtichoke',
              'VeggieTomato', 'VeggieZucchini', 'VeggiePotato', 'VeggieCauliflower', 'MeatChicken']
GRASPABLES = [o.lower() for o in GRASPABLES]


SCALE_DB = abspath(join(dirname(__file__), 'model_scales.json'))

SAMPLER_DB = abspath(join(dirname(__file__), 'sampling_distributions.json'))
SAMPLER_KEY = "{x}&{y}"

Z_CORRECTION_FILE = join(dirname(__file__), '..', 'databases', 'pose_z_correction.json')
SCENE_CONFIG_PATH = abspath(join(dirname(__file__), '..', 'data_generator', 'configs'))


def parse_yaml(path, verbose=True):
    from pybullet_tools.logging_utils import myprint as print
    import yaml
    import pprint
    from pathlib import Path
    from argparse import Namespace
    conf = yaml.safe_load(Path(path).read_text())
    if verbose:
        print(f'-------------- {abspath(path)} --------------')
        print(pprint.pformat(conf))
        print('------------------------------------\n')
    conf = Namespace(**conf)
    conf.sim = Namespace(**conf.sim)
    conf.data = Namespace(**conf.data)
    conf.world = Namespace(**conf.world)
    conf.robot = Namespace(**conf.robot)
    conf.planner = Namespace(**conf.planner)
    return conf


def add_walls_given_rooms_doors(objects):
    wall_thickness = 30
    wall_offsets = {
        "east": (0, 1),
        "south": (1, 0),
        "west": (0, -1),
        "north": (-1, 0)
    }
    rooms = []
    wall_objects = {}
    for k, v in objects.items():
        if v['category'] == 'room':
            rooms.append(k)
            index = k.replace('room_', '').replace('_room', '')
            walls = []
            for direction, (ix, iy) in wall_offsets.items():
                wall_name = f"wall_{index}_{direction}"
                length = abs(iy * v['w'] + ix * v['l']) + wall_thickness * 2
                yaw = 90 if ix == 0 else 0
                wall_objects[wall_name] = {'x': v['x'] + ix * (v['w']+wall_thickness)/2,
                                           'y': v['y'] + iy * (v['l']+wall_thickness)/2,
                                           'yaw': yaw, 'w': length, 'l': wall_thickness, 'category': 'wall'}
                walls.append(wall_name)

            door_name = f"doorframe_{index}"
            if door_name in objects:
                door_attr = objects[door_name]
                for wall_name in walls:
                    wall_attr = wall_objects[wall_name]
                    if abs(wall_attr['x'] - door_attr['x']) <= door_attr['w'] / 2:
                        left = wall_attr['y'] - wall_attr['w'] / 2
                        right = wall_attr['y'] + wall_attr['w'] / 2
                        door_left = door_attr['y'] - door_attr['w'] / 2
                        door_right = door_attr['y'] + door_attr['w'] / 2
                        seg_index = 0
                        for l, r in [(left, door_left), (door_right, right)]:
                            wall_objects[f"{wall_name}_{seg_index}"] = {
                                'x': wall_attr['x'], 'y': (l + r) / 2, 'w': r - l, 'l': wall_attr['l'],
                                'yaw': wall_attr['yaw'], 'category': 'wall'}
                            seg_index += 1
                    elif abs(wall_attr['y'] - door_attr['y']) <= door_attr['w'] / 2:
                        top = wall_attr['x'] - wall_attr['w'] / 2
                        bottom = wall_attr['x'] + wall_attr['w'] / 2
                        door_top = door_attr['x'] - door_attr['w'] / 2
                        door_bottom = door_attr['x'] + door_attr['w'] / 2
                        seg_index = 0
                        for l, r in [(top, door_top), (door_bottom, bottom)]:
                            wall_objects[f"{wall_name}_{seg_index}"] = {
                                'x': (l + r) / 2, 'y': wall_attr['y'], 'w': r - l, 'l': wall_attr['l'],
                                'yaw': wall_attr['yaw'], 'category': 'wall'}
                            seg_index += 1
    objects.update(wall_objects)
    for k in rooms:
        del objects[k]
    return objects


def get_domain_constants(pddl_path):
    lines = open(pddl_path, 'r').readlines()
    constants = []
    start = False
    for i in range(len(lines)):
        line = lines[i]
        if '(:constants' in line:
            start = True
        elif start and ')' in line:
            break
        elif start:
            constants.extend(line.strip().split(' '))
    constants = [c.strip() for c in constants if c.strip()]
    return constants


ALLOWED_ROBOT_NAMES = ['pr2', 'rummy', 'spot', 'robot']


def read_xml(plan_name, asset_path=ASSET_PATH, auto_walls=False):
    """ load a svg file representing the floor plan in asset path
            treating the initial pose of robot as world origin
        return a dictionary of object name: (category, pose) as well as world dimensions
        plan_name: svg file, boxes need to be single-stroke, no fill, no rounded corner
        auto_walls: automatically create walls with door space for "office_xxx" or "xxx_room" in the layout
    """
    plan_path = abspath(join(asset_path, 'floorplans', plan_name))

    X_OFFSET, Y_OFFSET, SCALING = None, None, None
    FLOOR_X_MIN, FLOOR_X_MAX = inf, -inf
    FLOOR_Y_MIN, FLOOR_Y_MAX = inf, -inf
    objects = {}
    objects_by_category = {}
    content = untangle.parse(plan_path).svg.g.g.g
    for object in content:
        name = None
        rect = object.rect[0]
        w = float(rect['height'])
        h = float(rect['width'])
        x = float(rect['y']) + w / 2
        y = float(rect['x']) + h / 2

        text = ''.join([t.cdata for t in object.text.tspan]).lower()

        if text in ALLOWED_ROBOT_NAMES:
            SCALING = w / 0.8
            X_OFFSET, Y_OFFSET = x, y
            continue

        if '/' in text:
            category, yaw = text.split('/')
            yaw = int(yaw)
            if yaw in [90, 270]:
                w = float(rect['width'])
                h = float(rect['height'])

        elif text.startswith('.'):
            category = 'doorframe'
            if len(text) > 1:
                name = f"{category}_{text[1:]}"
            if w > h:
                yaw = 270
            else:
                yaw = 180

        elif 'floor' in text or 'room' in text:
            category = 'floor'
            if 'floor' not in text and auto_walls:
                category = 'room'
            yaw = 0
            name = text
            if float(rect['y']) < FLOOR_X_MIN:
                FLOOR_X_MIN = float(rect['y'])
            if float(rect['x']) < FLOOR_Y_MIN:
                FLOOR_Y_MIN = float(rect['x'])
            if float(rect['y']) + w > FLOOR_X_MAX:
                FLOOR_X_MAX = float(rect['y']) + w
            if float(rect['x']) + w > FLOOR_Y_MAX:
                FLOOR_Y_MAX = float(rect['x']) + h

        elif 'office' in text:
            category = 'office'
            yaw = 0
            name = text

        else:
            print('what is this')

        if category not in objects_by_category:
            objects_by_category[category] = []
        if name is None:
            next = len(objects_by_category[category])
            name = f"{category}#{next + 1}"
        objects_by_category[category].append(name)

        objects[name] = {'x': x, 'y': y, 'yaw': yaw, 'w': w, 'l': h, 'category': category}

    ## associate rooms with doors for computing walls
    if auto_walls:
        objects = add_walls_given_rooms_doors(objects)

    return objects, X_OFFSET, Y_OFFSET, SCALING, FLOOR_X_MIN, FLOOR_X_MAX, FLOOR_Y_MIN, FLOOR_Y_MAX


def get_model_scale(file, l=None, w=None, h=None, scale=1, category=None):
    scale_db = {}
    # if isfile(SCALE_DB):
    #     with open(SCALE_DB, "r") as read_file:
    #         scale_db = json.loads(read_file.read())
        # if file in scale_db:
        #     return scale_db[file]

    # if 'oven' in file.lower() or (category != None and category.lower() == 'oven'):
    #     print(file, l, w)

    ## ------- Case 1: no restrictions or directly given scale
    if w is None and l is None and h is None:
        return scale

    ## --- load and adjust
    with HideOutput():
        if isdir(file):
            file = join(file, 'mobility.urdf')
        body = load_model(file, scale=scale, fixed_base=True)
    aabb = get_aabb(body)
    extent = get_aabb_extent(aabb)

    ## ------- Case 2: given width and length of object
    if w is not None or l is not None:
        width, length = extent[:2]
        if w is None:
            scale *= l / length
        elif l is None:
            scale *= w / width
        else:
            scale *= min(l / length, w / width) ## unable to reconstruct

    ## ------ Case 3: given height of object
    elif h is not None:
        height = extent[2]
        scale *= h / height

    ## ------ Case N: exceptions
    if category is not None:
        if 'door' == category.lower():
            set_joint_position(body, get_joints(body)[1], -0.8)
        if 'dishwasher' == category.lower():
            set_joint_position(body, get_joints(body)[3], -0.66)
        if 'door' == category.lower():
            scale = (l / length + w / width) / 2

    # scale_db[file] = scale
    # with open(SCALE_DB, 'w') as f:
    #     json.dump(scale_db, f)
    remove_body(body)

    return scale


#######################################################################


def get_partnet_semantics(path):
    if path.endswith('urdf'):
        path = path[:path.rfind('/')]
    file = join(path, 'semantics.txt')
    lines = []
    with open(file, 'r') as f:
        for line in f.readlines():
            lines.append(line.replace('\n', '').split(' '))
    return lines


def get_partnet_doors(path, body):
    body_joints = {}
    for line in get_partnet_semantics(path):
        link_name, part_type, part_name = line
        if part_type == 'hinge' and part_name in ['door', 'rotation_door']:
            link = link_from_name(body, link_name)
            joint = parent_joint_from_link(link)
            joint_name = '--'.join(line)
            body_joint = (body, joint)
            body_joints[body_joint] = joint_name
    return body_joints


def get_partnet_spaces(path, body):
    space_links = {}
    for line in get_partnet_semantics(path):
        link_name, part_type, part_name = line
        if '_body' in part_name:
            link = link_from_name(body, link_name)
            space_links[(body, None, link)] = part_name
    return space_links


def get_partnet_links_by_type(path, body, keyward):
    links = []
    for line in get_partnet_semantics(path):
        if line[-1] == keyward:
            links.append(link_from_name(body, line[0]))
    return links


def get_grasp_link(path, body):
    link = None
    for line in get_partnet_semantics(path):
        if line[-1] == 'grasp_link':
            link = link_from_name(body, line[0])
    return link


#######################################################################


def partnet_id_from_path(path):
    id = path.replace('/mobility.urdf', '')
    return id[id.rfind('/') + 1:]


def get_sampled_file(sampling, category, ids):
    from world_builder.entities import Object
    dists = json.load(open(SAMPLER_DB, 'r'))

    ## conditioned on the instance distribution of another object
    dist = None
    if isinstance(sampling, Object):
        key = SAMPLER_KEY.format(x=sampling.category.lower(), y=category.lower())
        id = partnet_id_from_path(sampling.path)
        if id in dists[key]:
            dist = dists[key][id]
    elif category.lower() in dists:
        key = category.lower()
        dist = dists[category.lower()]

    if dist is not None:
        p = []
        for i in ids:
            if i in dist:
                p.append(dist[i])
            else:
                p.append(0.25)
        if sum(p) > 1.2:
            p = [x / sum(p) for x in p]
        if sum(p) != 1:
            p[-1] = 1-sum(p[:-1])
        id = np.random.choice(ids, p=p)
        print(f'world_builder.utils.get_sampled_file({key}, {category}) chose {id}')
        return str(id)
    return None


def get_file_by_category(category : str, random_instance : str | bool = False, sampling : str | bool = False):
    """ category:   name of the object category in assets/models/
            see https://github.com/zt-yang/kitchen-models
        random_instance:    sample an asset instance randomly, instead of using the first one.
                            if random_instance is given a string value, use that instance
        sampling:   sample an asset instance in a way that balances instance distribution
                    or conditioned on the instance distribution of other categories
    """
    file = None

    ## correct the capitalization because Ubuntu cares about it
    cats = [c for c in listdir(join(ASSET_PATH, 'models')) if c.lower() == category.lower()]
    if len(cats) > 0:
        category = cats[0]

    ## sometimes the category name is a subcategory and there's only one instance of it, e.g. models/Food/VeggieCabbage
    asset_root = join(ASSET_PATH, 'models', category)
    if isdir(asset_root):
        ids = [f for f in listdir(join(asset_root)) if isdir(join(asset_root, f)) and not f.startswith('_')]
        files = [join(asset_root, f) for f in listdir(join(asset_root)) if 'DS_Store' not in f and not f.startswith('_')]

        if len(ids) == 0:
            print(f'get_file_by_category({category})')
        assert len(ids) > 0

        ## for partnet-mobility objects, all files in the category directory are directories
        if len(ids) == len(files):
            paths = [join(asset_root, p) for p in ids]
            paths.sort()
            if random_instance:
                paths = [p for p in paths if basename(p) not in DONT_LOAD]
                if isinstance(random_instance, str):
                    paths = [join(asset_root, random_instance)]
                else:
                    sampled = False
                    if sampling:
                        result = get_sampled_file(sampling, category, ids)
                        if isinstance(result, str):
                            paths = [join(asset_root, result)]
                            sampled = True
                    if not sampled:
                        random.shuffle(paths)
            file = join(paths[0], 'mobility.urdf')

        ## for loading the digital twin of nvidia kitchen
        elif category == 'counter':
            file = join(ASSET_PATH, 'models', 'counter', 'urdf', 'kitchen_part_right_gen_convex.urdf')

    else:
        parent = get_parent_category(category)
        assert parent

        ## find the subcategory
        category = [c for c in listdir(join(ASSET_PATH, 'models', parent)) if c.lower() == category.lower()][0]
        file = join(ASSET_PATH, 'models', parent, category, 'mobility.urdf')

    return file


def get_scale_by_category(file=None, category=None, scale=1):
    from world_builder.asset_constants import MODEL_HEIGHTS, MODEL_SCALES, OBJ_SCALES

    cat = category.lower()

    ## general category-level
    if category is not None:
        if cat in OBJ_SCALES:
            scale = OBJ_SCALES[cat]

    ## specific instance-level
    if file is not None:
        if category in MODEL_HEIGHTS:
            height = MODEL_HEIGHTS[category]['height']
            # print('bullet_utils.get_scale_by_category', file)
            scale = get_model_scale(file, h=height)
        elif category in MODEL_SCALES:
            f = file.lower()
            f = f[f.index(cat) + len(cat) + 1:]
            id = f[:f.index('/')]
            if id in MODEL_SCALES[category]:
                scale = MODEL_SCALES[category][id]
        else:
            parent = get_parent_category(category)
            # if parent is None:
            #     print('\tcant find model scale', category, 'using default 1')
            if parent in MODEL_SCALES:
                scale = MODEL_SCALES[parent][category]

    return scale


def get_parent_category(category):
    from world_builder.asset_constants import MODEL_SCALES
    for k, v in MODEL_SCALES.items():
        for vv in v:
            if vv == category:
                for m in [k, k.capitalize(), k.lower()]:
                    if isdir(join(ASSET_PATH, 'models', m)):
                        return k
    return None


def adjust_scale(body, category, file, w, l):
    """ reload with the correct scale """
    aabb = get_aabb(body)
    width = aabb.upper[0] - aabb.lower[0]
    length = aabb.upper[1] - aabb.lower[1]
    height = aabb.upper[2] - aabb.lower[2]

    if 'door' == category.lower():
        set_joint_position(body, get_joints(body)[1], -0.8)
    if 'dishwasher' == category.lower():
        set_joint_position(body, get_joints(body)[3], -0.66)

    if 'door' == category.lower():
        scale = (l / length + w / width) / 2  ##
    else:
        scale = min(l / length, w / width)  ##
    remove_body(body)
    with HideOutput():
        body = load_model(file, scale=scale, fixed_base=True)

    return body


def load_asset(category, x=0, y=0, yaw=0, floor=None, z=None, w=None, l=None, h=None,
               scale=1, verbose=False, random_instance=False, sampling=False, random_scale=1.0):
    """ returns body, file, and scale used to initiate an entity Object """

    """ ============= load body by category ============= """
    file = get_file_by_category(category, random_instance=random_instance, sampling=sampling)
    if verbose and file is not None:
        print(f"Loading ...... {abspath(file)}", end='\r')

    scale = get_scale_by_category(file=file, category=category, scale=scale)
    if file is not None:
        scale = get_model_scale(file, l, w, h, scale, category)

        scale = random_scale * scale

        with HideOutput():
            body = load_model(file, scale=scale, fixed_base=True)
    else:
        body = create_box(w=w, l=l, h=1, color=BROWN, collision=True)

    """ ============= place object z on a surface or floor ============= """
    if z is None:
        if category.lower() in ['oven']:
            aabb = get_aabb(body)
            height = aabb.upper[2] - aabb.lower[2]
            z = height / 2
        elif isinstance(floor, tuple):
            z = stable_z(body, floor[0], floor[1])
        elif isinstance(floor, int):
            z = stable_z(body, floor)
        elif hasattr(floor, 'body') and isinstance(floor.body, int):
            z = stable_z(body, floor.body)
        else:
            z = 0
    pose = Pose(point=Point(x=x, y=y, z=z), euler=Euler(yaw=yaw))
    set_pose(body, pose)

    """ ============= category-specific modification ============= """
    if category.lower() == 'veggieleaf':
        set_color(body, DARK_GREEN, 0)
    elif category.lower() == 'veggiestem':
        set_color(body, WHITE, 0)
    elif category.lower() == 'facetbase':
        from pybullet_tools.bullet_utils import open_doors_drawers
        open_doors_drawers(body)

    """ ============= create an Object ============= """
    # if movable:
    #     object = Movable(body, category=category)
    # elif category.lower() == 'food':
    #     # index = file.replace('/mobility.urdf', '')
    #     # index = index[index.index('models/')+7:]
    #     # index = index[index.index('/')+1:]
    #     index = partnet_id_from_path(file)
    #     return body, file, scale, index
    # else:
    #     object = Object(body, category=category)

    # if category.lower() == 'food':
    #     # index = file.replace('/mobility.urdf', '')
    #     # index = index[index.index('models/')+7:]
    #     # index = index[index.index('/')+1:]
    #     index = partnet_id_from_path(file)
    #     return body, file, scale, index
    return body, file, scale


def world_of_models(floor_width=5, floor_length = 5):
    from world_builder.entities import Floor
    from pybullet_tools.bullet_utils import add_body
    set_camera_pose(camera_point=[floor_width, 0., 5], target_point=[1., 0, 1.])
    floor = add_body(Floor(create_box(w=floor_width*2, l=floor_length*2, h=FLOOR_HEIGHT, color=TAN, collision=True)),
                     Pose(point=Point(z=-2*FLOOR_HEIGHT)))

    box = load_asset('Box', x=-0.7*floor_width, y= -0.7 * floor_length, yaw=PI, floor=floor)
    cart = load_asset('Cart', x=-0.7*floor_width, y= 0.7 * floor_length, yaw=PI, floor=floor)
    chair = load_asset('Chair', x=-0.7*floor_width, y= -0.3 * floor_length, yaw=PI, floor=floor)
    dishwasher = load_asset('Dishwasher', x=-0.7*floor_width, y= 0.3 * floor_length, yaw=PI, floor=floor)

    door = load_asset('Door', x=-0.3*floor_width, y= 0.3 * floor_length, yaw=PI, floor=floor)
    faucet = load_asset('Faucet', x=-0.3*floor_width, y= -0.3 * floor_length, yaw=PI, floor=floor)
    kettle = load_asset('Kettle', x=-0.3*floor_width, y= 0.7 * floor_length, yaw=PI, floor=floor)
    kitchenPot = load_asset('KitchenPot', x=-0.3*floor_width, y= -0.7 * floor_length, yaw=PI, floor=floor)

    microwave = load_asset('Microwave', x=0.3*floor_width, y= -0.7 * floor_length, yaw=PI, floor=floor)
    oven = load_asset('Oven', x=0.3*floor_width, y= 0.7 * floor_length, yaw=PI, floor=floor)
    refrigerator = load_asset('Fridge', x=0.3*floor_width, y= -0.3 * floor_length, yaw=PI, floor=floor)
    table = load_asset('Table', x=0.3*floor_width, y= 0.3 * floor_length, yaw=PI, floor=floor)

    toaster = load_asset('Toaster', x=0.7*floor_width, y= 0.3 * floor_length, yaw=PI, floor=floor)
    trashCan = load_asset('TrashCan', x=0.7*floor_width, y= -0.3 * floor_length, yaw=PI, floor=floor)
    washingMachine = load_asset('WashingMachine', x=0.7*floor_width, y= 0.7 * floor_length, yaw=PI, floor=floor)
    window = load_asset('Window', x=0.7*floor_width, y= -0.7 * floor_length, yaw=PI, floor=floor)

    return floor


def visualize_point(point, world):
    from .entities import Movable
    z = 0
    if len(point) == 3:
        x, y, z = point
    else:
        x, y = point
    world.add_object(
        Movable(create_box(.05, .05, .05, mass=1, color=(1, 0, 0, 1)), category='marker'),
        Pose(point=Point(x, y, z)))


def sort_instances(category, instances, get_all=False):
    keys = list(instances.keys())

    if isinstance(keys[0], tuple):
        instances = instances
    else:
        if not get_all:
            cat_dir = join(ASSET_PATH, 'models', category)
            if isdir(cat_dir) and len(listdir(cat_dir)) > 0:
                keys = [k for k in keys if isdir(join(cat_dir, k))]
        if not keys[0].isdigit():
            keys = list(set([k for k in keys]))
            instances = {k: instances[k] for k in keys}
            instances = dict(sorted(instances.items()))
        else:
            instances = {k: instances[k] for k in keys}
            instances = dict(sorted(instances.items()))
    return instances


def get_instances(category, **kwargs):
    from world_builder.asset_constants import MODEL_SCALES, MODEL_HEIGHTS, OBJ_SCALES
    if category in MODEL_SCALES:
        instances = MODEL_SCALES[category]
    elif category in MODEL_HEIGHTS:
        instances = MODEL_HEIGHTS[category]['models']
        instances = {k: 1 for k in instances}
    else:
        parent = get_parent_category(category)
        if parent is not None and parent in MODEL_SCALES:
            instances = {(parent, k): v for k, v in MODEL_SCALES[parent].items() if k == category}
        elif category.lower() in OBJ_SCALES:
            scale = OBJ_SCALES[category.lower()]
            category = [c for c in listdir(join(ASSET_PATH, 'models')) if c.lower() == category.lower()][0]
            asset_root = join(ASSET_PATH, 'models', category)
            indices = [f for f in listdir(join(asset_root)) if isdir(join(asset_root, f))]
            instances = {k: scale for k in indices}
        else:
            instances = []
            print(f'world_builder.utils.get_instances({category}) didnt find any models')
            assert NotImplementedError()
    return sort_instances(category, instances, **kwargs)


def get_instance_name(path):
    if not isfile(path): return None
    rows = open(path, 'r').readlines()
    if len(rows) > 50: rows = rows[:50]

    def from_line(r):
        r = r.replace('\n', '')[13:]
        return r[:r.index('"')]

    name = [from_line(r) for r in rows if '<robot name="' in r]
    if len(name) == 1:
        return name[0]
    return None


def get_mobility_id(path):
    if path is None or not path.endswith('mobility.urdf'):
        return None
    idx = dirname(path)
    idx = idx.replace(dirname(idx), '')
    return idx[1:]


def get_mobility_category(path):
    if path is None or not path.endswith('mobility.urdf'):
        return None
    idx = dirname(path)
    idx = idx.replace(dirname(dirname(idx)), '').replace(basename(idx), '')
    return idx[1:-1]


def get_mobility_identifier(path):
    if path is None or not path.endswith('mobility.urdf'):
        return None
    idx = dirname(path)
    idx = idx.replace(dirname(dirname(idx)), '')
    return idx[1:]


def HEX_to_RGB(color):
    color = color.lstrip('#')
    rgb = [int(color[i: i+2], 16)/255 for i in (0, 2, 4)] + [1]
    return tuple(rgb)


def adjust_for_reachability(obj, counter, d_x_min=0.3, body_pose=None, return_pose=False):
    x_min = counter.aabb().upper[0] - d_x_min
    x_max = counter.aabb().upper[0]
    if body_pose is None:
        body_pose = obj.get_pose()
    if obj.aabb().lower[0] > x_min and obj.aabb().upper[0] < x_max:
        if return_pose:
            return body_pose
        return

    (x, y, z), r = body_pose
    x_min += (y - obj.aabb().lower[1])
    ## scale the x to a reachable range
    x_max = counter.aabb().upper[0] - (obj.aabb().upper[0] - x)
    x_new = x_max - (x_max - x_min) * (x_max - x) / counter.lx
    if return_pose:
        return (x_new, y, z), r
    obj.set_pose(((x_new, y, z), r))
    counter.attach_obj(obj)


def smarter_sample_placement(body, surface, world, **kwargs):
    from world_builder.world import World
    if isinstance(world, World):
        obj = world.BODY_TO_OBJECT[body]
        surface = world.BODY_TO_OBJECT[surface]
        xa, ya, za = surface.aabb().lower
        xb, yb, zb = surface.aabb().upper
        obstacles = surface.supported_objects
        regions = [(ya, yb)]
        for o in obstacles:
            y1 = o.get_aabb().lower[1]
            y2 = o.get_aabb().upper[1]
            new_regions = []
            for y1_, y2_ in regions:
                if y1_ < y1 < y2 < y2_:
                    new_regions.append((y1_, y1))
                    new_regions.append((y2, y2_))
                elif y1_ < y1 < y2_:
                    new_regions.append((y1_, y1))
                elif y1_ < y2 < y2_:
                    new_regions.append((y2, y2_))
                else:
                    new_regions.append((y1_, y2_))
            regions = new_regions

        regions = sorted(regions, key=lambda r: r[1] - r[0], reverse=True)
        body_pose = None
        for y1, y2 in regions:
            if y2 - y1 < obj.ly:
                continue
            aabb = AABB(lower=(xa, y1, za), upper=(xb, y2, zb))
            # draw_aabb(aabb, color=RED)
            # wait_unlocked()
            body_pose = sample_placement_on_aabb(body, aabb, **kwargs)
            if body_pose is not None:
                body_pose = adjust_for_reachability(obj, surface, body_pose=body_pose, return_pose=True)
                break
    else:
        body_pose = sample_placement(body, surface, **kwargs)
    return body_pose


def get_camera_zoom_in(run_dir):
    config_file = join(run_dir, 'planning_config.json')
    if isfile(config_file):
        config = json.load(open(config_file, 'r'))
        if 'camera_zoomins' in config:
            camera_zoomins = config['camera_zoomins']
            if len(camera_zoomins) > 0:
                return camera_zoomins[-1]
    return None


def reduce_model_scale(txt_file, scale_down=10, new_scale_file='new_scale.txt'):
    new_lines = []
    for line in open(txt_file, 'r').readlines():
        nums = line.strip().split(' ')
        prefix = [n for n in nums if len(n) <= 1]
        nums = [n for n in nums if len(n) > 1]
        nums = prefix + [str(round(eval(n)/scale_down, 4)) for n in nums]
        new_lines.append(' '.join(nums)+'\n')
    with open(join(dirname(txt_file), new_scale_file), 'w') as f:
        f.writelines(new_lines)


def get_lisdf_name(body, name, joint=None, link=None):
    LINK_STR = '::'
    if joint is not None or link is not None:
        body_name = name.split(LINK_STR)[0]
        if joint is None and link is not None:
            return f"{body_name}{LINK_STR}{get_link_name(body, link)}"
        elif joint is not None and link is None:
            return f"{body_name}{LINK_STR}{get_joint_name(body, joint)}"
    return name


def get_placement_z():
    z_correction = json.load(open(Z_CORRECTION_FILE, 'r'))
    z_correction.update({k.lower(): {kk.lower(): vv for kk, vv in v.items()} for k, v in z_correction.items()})
    return z_correction


#########################################################


def get_camera_image(camera, include_rgb=False, include_depth=False, include_segment=False):
    rgb, depth, seg, pose, matrix = camera.get_image(segment=include_segment, segment_links=False)
    if not include_rgb:
        rgb = None
    if not include_depth:
        depth = None
    if not include_segment:
        seg = None
    return CameraImage(rgb, depth, seg, pose, matrix)


def make_camera_image(rgbd_matrix, figsize=None, name_to_loc={},
                      fontsize=12, lineheight=18, alpha=0.7, padding=4,
                      show_annotation=True, show=False, output_path='observation.png', verbose=False):
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(rgbd_matrix)

    ax.axis('off')
    fig.tight_layout()

    if figsize is not None:
        colors = DARKER_COLORS[:-1]
        for i, (text, (textx, texty, left, top, width, height)) in enumerate(name_to_loc.items()):
            if show_annotation:
                color = colors[i % len(colors)]
                ax.add_patch(mpatches.Rectangle(
                    (left + padding, top + padding), width - 2 * padding, height - 2 * padding,
                    fill=False, edgecolor=color, alpha=alpha, linewidth=2
                ))
                plt.text(textx, texty, text, fontsize=fontsize, color='white',
                         bbox=dict(fill=True, color=color, alpha=alpha, linewidth=0))

                if show:
                    left, top, width, height = _get_rect_of_text_box(text, textx, texty, fontsize, lineheight, padding)
                    ax.add_patch(mpatches.Rectangle(
                        (left, top), width, height, fill=False, edgecolor='white', linewidth=1
                    ))

            else:
                plt.text(textx, texty, text, fontsize=fontsize)

    if show:
        fig.show()
        if is_darwin():
            wait_unlocked()
    else:
        fig.savefig(output_path, dpi=100)

    if verbose:
        print(f'world_utils.make_camera_image(rgbd_matrix) \t {output_path}')


def _get_rect_of_text_box(text, textx, texty, fontsize, lineheight, padding):
    left = textx-padding
    top = texty-lineheight/2-padding*0.6
    width = _get_text_width(text, fontsize)
    height = lineheight
    return left, top, width, height


def _get_bb_of_text_box(text, textx, texty, fontsize, lineheight, padding):
    left, top, width, height = _get_rect_of_text_box(text, textx, texty, fontsize, lineheight, padding)
    return aabb_from_extent_center((width, height), (left+width/2, top+height/2))


def _get_text_width(text, fontsize):
    narrower = 'iflj'
    widths = [0.3 if c in narrower else 0.6 for c in text+'  ']
    return sum(widths) * fontsize


def make_camera_collage(camera_images, output_path='observation.png', verbose=False):
    import matplotlib.pyplot as plt
    images = [camera_image.rgbPixels for camera_image in camera_images]
    fig, axes = plt.subplots(1, len(images), figsize=(5 * len(images), 4))
    for i, rgb in enumerate(images):
        axes[i].imshow(rgb)
        axes[i].axis('off')
    fig.tight_layout()
    fig.savefig(output_path)
    if verbose:
        print(f'world_utils.make_camera_collage(images) len = {len(images)} \t {output_path}')


##############################################################################

ABOVE = 0
BELOW = 1
RIGHT = 2
LEFT = 3


def make_camera_image_with_object_labels(camera, body_to_english_names, object_label_locations={},
                                         verbose=False, fontsize=8, lineheight=18, padding=4, **kwargs):
    """ for creating image for VLM query """

    ## reorg to label the special cases first
    body_to_english_names = {k: v for k, v in body_to_english_names.items() if not v.endswith('space')}
    body_dict = {k: v for k, v in body_to_english_names.items() if v in object_label_locations}
    body_dict.update({k: v for k, v in body_to_english_names.items() if v not in object_label_locations})

    camera_image = camera.get_image(segment=True, segment_links=True)
    rgb = camera_image.rgbPixels[:, :, :3]
    seg = camera_image.segmentationMaskBuffer
    unique = get_segmask(seg)
    obj_keys = get_obj_keys_for_segmentation(body_to_english_names, unique)

    heights = []
    name_to_loc = {}
    for k, name in body_dict.items():
        keys = obj_keys[name]
        mask = get_seg_foreground_given_obj_keys(rgb, keys, unique)
        bb = get_mask_bb(mask)
        if bb is not None:
            center = get_aabb_center(bb)
            width, height = get_aabb_extent(bb)
            left, top = bb.lower
            x, y = center.tolist()

            box_area = (width - padding) * (height - padding)
            if box_area <= 500:
                left -= 2 * padding
                top -= 2 * padding
                width += 4 * padding
                height += 4 * padding
                box_area = (width - padding) * (height - padding)

            _, _, textbox_width, textbox_height = _get_rect_of_text_box(name, x, y, fontsize, lineheight, padding)
            text_box_area = textbox_width * textbox_height
            is_small_box = name in object_label_locations  # box_area < 2.4 * text_box_area
            if verbose:
                print(f'{name}\tis_small_box = {is_small_box}\t box_area = {box_area}\t (top, left) = {(top, left)}'
                      f'\t textbox_width = {round(textbox_width, 2)}\t text_box_area = {round(text_box_area, 2)}')

            ## adjust the x of label to align it to the center of object parts
            x = x + padding / 2 - textbox_width / 2
            y += padding / 2

            ## adjust for specific cases
            if is_small_box:
                x, y = _adjust_xy_for_small_box(x, y, top, left, width, height, padding,
                                                name, textbox_width, textbox_height, object_label_locations)

            else:
                ## adjust the y of label to ensure not overlapping
                yp = _adjust_y_to_prevent_overlap(y, heights, lineheight, bb, name, x, fontsize, padding)
                if yp is not None:
                    y = yp
            heights.append(y)

            name_to_loc[name] = [x, y, left, top, width, height]

    figsize = (8, 6)  ## (800px by 600px)
    name_to_loc = dict(sorted(name_to_loc.items(), key=lambda item: item[1][0]))
    make_camera_image(rgb, figsize=figsize, name_to_loc=name_to_loc,
                      fontsize=fontsize, padding=padding, lineheight=lineheight, **kwargs)


def _adjust_xy_for_small_box(x, y, top, left, width, height, padding, text, textbox_width, textbox_height,
                             object_label_locations):
    case = 0
    if text in object_label_locations:
        case = object_label_locations[text]
    ## above box
    if case == 0:
        y = top - textbox_height / 2 + padding * 2
    ## below box
    if case == 1:
        y = top + height - padding + textbox_height / 2
    ## right of box
    if case == 2:
        x = left + width
    ## left of box
    if case == 3:
        x = left - textbox_width + padding
    return x, y


def _adjust_y_to_prevent_overlap(y, heights, lineheight, bb, name, x, fontsize, padding, verbose=False):
    for yy in heights:
        if abs(yy - y) < lineheight:
            found = False
            for dy in [1, -1]:
                yyy = y + lineheight * dy

                ## first criteria to check: label not overlapping with other boxes
                overlap_with_others = False
                for yyyy in heights:
                    if yyyy != yy and abs(yyyy - yyy) < lineheight:
                        overlap_with_others = True

                ## second criteria to check: label not overlapping with other boxes
                new_bb = _get_bb_of_text_box(name, x, yyy, fontsize, lineheight, padding)
                overlap_with_bb = aabb_overlap(bb, new_bb)

                if not overlap_with_others and overlap_with_bb:
                    found = True
                    if verbose:
                        print(f'\t{y} -> {yyy}')
                    y = yyy
                    break
            if not found:
                return None
    return y

##############################################################################


def get_objs_in_camera_images(camera_images, world=None, show=False, save=False, verbose=False):
    import matplotlib.pyplot as plt

    objs = []
    images = []
    colors = get_fine_rainbow_colors(math.ceil(len(world.all_objects)/7))

    for camera_image in camera_images:

        rgb = camera_image.rgbPixels
        depth = camera_image.depthPixels
        seg = camera_image.segmentationMaskBuffer

        ## create segmentation images
        unique = get_segmask(seg)
        seg = np.zeros_like(rgb[:, :, :4])
        for (body, link), pixels in unique.items():
            c, r = zip(*pixels)
            color = colors[body]
            if verbose and world is not None:
                print('\t', world.get_name(body))
            seg[(np.asarray(c), np.asarray(r))] = color

        objs += [b for b, l in unique.keys() if b not in objs]
        images.append((rgb, depth, seg, len(unique)))

    ## show color, depth, and segmentation images
    if show or save:

        fig, axes = plt.subplots(len(images), 3, figsize=(15, 5*len(images)))
        fig.suptitle('Camera Image', fontsize=24)

        for i, (rgb, depth, seg, n_obj) in enumerate(images):
            camera = f'Camera {i} | '

            axes[i, 0].imshow(rgb)
            axes[i, 0].set_title(f'{camera}RGB Image')

            axes[i, 1].imshow(depth)
            axes[i, 1].set_title(f'{camera}Depth Image')

            axes[i, 2].imshow(seg)
            axes[i, 2].set_title(f'{camera}Segmentation Image ({n_obj} obj)')

        fig.tight_layout()
        if show:
            fig.show()
        else:
            fig.savefig('observation.png')

    return objs


def sort_body_indices(lst):
    """ given a list of int (whole object) and tuples (joints or links), return a sorted list """
    bodies = [o for o in lst if isinstance(o, int)]
    joints = [o for o in lst if isinstance(o, tuple) and len(o) == 2]
    links = [o for o in lst if isinstance(o, tuple) and len(o) == 3]
    all_bodies = list(set(bodies + [o[0] for o in joints + links]))
    all_bodies.sort()

    sorted_lst = []
    for body in all_bodies:
        if body in bodies:
            sorted_lst.append(body)

        found_joints = [o[-1] for o in joints if o[0] == body]
        found_joints.sort()
        sorted_lst.extend([(body, j) for j in found_joints])

        found_links = [o[-1] for o in links if o[0] == body]
        found_links.sort()
        sorted_lst.extend([(body, None, l) for l in found_links])
    return sorted_lst


def draw_body_label(body, text, link=None, offset=(0, -0.05, 0.05), **kwargs):
    with PoseSaver(body):
        if link is None and get_color(body) == GREEN:
            set_pose(body, unit_pose())
        lower, upper = get_aabb(body, link=link)
    position = ((lower[0] + upper[0]) / 2, (lower[1] + upper[1]) / 2, upper[2])
    position = [position[i]+offset[i] for i in range(len(position))]
    return add_text(text, position=position, **kwargs)  # , lifetime=0 , parent=body


## ----------------------------------------------------------------------------------


def save_world_aabbs_to_json(world, json_name='world_shapes.json'):
    """ used by test_replay to make world_aabbs and find camera point """
    shapes = {}
    for body, obj in world.BODY_TO_OBJECT.items():
        if isinstance(body, int):
            aabb = get_aabb(body)
            shapes[obj.name] = (aabb.lower, aabb.upper)

    json_path = join(DATABASES_PATH, json_name)
    print(f'[world.save_world_aabbs_to_json]\t{json_path}')
    with open(json_path, 'w') as f:
        json.dump(shapes, f, indent=2, sort_keys=False)


def select_door_closer_to_body(door_links, movable):
    door_centers = {key: get_aabb_center(get_aabb(body, link=link)) for key, (body, link) in door_links.items()}
    movable_center = get_aabb_center(get_aabb(movable))
    return min(door_centers, key=lambda x: np.linalg.norm(np.array(door_centers[x]) - movable_center))


if __name__ == "__main__":
    reduce_model_scale(join(ASSET_PATH, 'models/Food/MeatTurkeyLeg/old_scale.txt'), scale_down=10)
