import numpy as np
import untangle
from numpy import inf
from os.path import join, isdir, isfile, dirname, abspath
from os import listdir
import json
import random

from .entities import Object, Floor, Moveable
from pybullet_tools.utils import unit_pose, get_aabb_extent, \
    set_pose, get_movable_joints, draw_pose, pose_from_pose2d, set_velocity, set_joint_states, get_bodies, \
    flatten, INF, inf_generator, get_time_step, get_all_links, get_visual_data, pose2d_from_pose, multiply, invert, \
    get_sample_fn, pairwise_collisions, sample_placement, is_placement, aabb_contains_point, point_from_pose, \
    aabb2d_from_aabb, is_center_stable, aabb_contains_aabb, get_model_info, get_name, get_pose, dump_link, \
    dump_joint, dump_body, PoseSaver, get_aabb, add_text, GREEN, AABB, remove_body, HideOutput, \
    stable_z, Pose, Point, create_box, load_model, get_joints, set_joint_position, BROWN, Euler, PI, \
    set_camera_pose, TAN, RGBA, sample_aabb, get_min_limit, get_max_limit, set_color, WHITE, get_links, \
    get_link_name, get_link_pose, euler_from_quat, get_collision_data, get_joint_name, get_joint_position
from pybullet_tools.bullet_utils import get_scale_by_category
from pybullet_tools.logging import dump_json
from world_builder.paths import ASSET_PATH

LIGHT_GREY = RGBA(0.5, 0.5, 0.5, 0.6)
DARK_GREEN = RGBA(35/255, 66/255, 0, 1)
FLOOR_HEIGHT = 1e-3
WALL_HEIGHT = 0.5

GRASPABLES = ['BraiserLid', 'Egg', 'VeggieCabbage', 'MeatTurkeyLeg', 'VeggieGreenPepper', 'VeggieArtichoke',
              'VeggieTomato', 'VeggieZucchini', 'VeggiePotato', 'VeggieCauliflower', 'MeatChicken']
GRASPABLES = [o.lower() for o in GRASPABLES]


SCALE_DB = abspath(join(dirname(__file__), 'model_scales.json'))


def read_xml(plan_name, asset_path=ASSET_PATH):
    X_OFFSET, Y_OFFSET, SCALING = None, None, None
    FLOOR_X_MIN, FLOOR_X_MAX = inf, -inf
    FLOOR_Y_MIN, FLOOR_Y_MAX = inf, -inf
    objects = {}
    objects_by_category = {}
    content = untangle.parse(join(asset_path, 'floorplans', plan_name)).svg.g.g.g
    for object in content:
        name = None
        rect = object.rect[0]
        w = float(rect['height'])
        h = float(rect['width'])
        x = float(rect['y']) + w / 2
        y = float(rect['x']) + h / 2

        text = ''.join([t.cdata for t in object.text.tspan])
        if 'pr2' in text:
            SCALING = w / 0.8
            X_OFFSET, Y_OFFSET = x, y
            continue
        elif '/' in text:
            category, yaw = text.split('/')
        elif '.' in text:
            category = 'door'
            if w > h:
                yaw = 270
            else:
                yaw = 180
        else:
            category = 'floor'
            yaw = 0
            if float(rect['y']) < FLOOR_X_MIN:
                FLOOR_X_MIN = float(rect['y'])
            if float(rect['x']) < FLOOR_Y_MIN:
                FLOOR_Y_MIN = float(rect['x'])
            if float(rect['y']) + w > FLOOR_X_MAX:
                FLOOR_X_MAX = float(rect['y']) + w
            if float(rect['x']) + w > FLOOR_Y_MAX:
                FLOOR_Y_MAX = float(rect['x']) + h
            name = text

        if category not in objects_by_category:
            objects_by_category[category] = []
        if name == None:
            next = len(objects_by_category[category])
            name = f"{category}#{next + 1}"
        objects_by_category[category].append(name)

        objects[name] = {'x': x, 'y': y, 'yaw': int(yaw), 'w': w, 'l': h, 'category': category}

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
    if w is None and h is None:
        return scale

    ## --- load and adjust
    with HideOutput():
        body = load_model(file, scale=scale, fixed_base=True)
    aabb = get_aabb(body)
    extent = get_aabb_extent(aabb)

    ## ------- Case 2: given width and length of object
    if w is not None:
        width, length = extent[:2]
        scale = min(l / length, w / width) ## unable to construct

    ## ------ Case 3: given height of object
    elif h is not None:
        height = extent[2]
        scale = h/height

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


def get_file_by_category(category, RANDOM_INSTANCE=False):
    ## correct the capitalization because Ubuntu cares about it
    category = [c for c in listdir(join(ASSET_PATH, 'models')) if c.lower() == category.lower()][0]

    asset_root = join(ASSET_PATH, 'models', category)  ## ROOT_DIR
    if isdir(asset_root):
        asset_root = join(ASSET_PATH, 'models', category)

        paths = [f for f in listdir(join(asset_root)) if isdir(join(asset_root, f))]
        files = [join(asset_root, f) for f in listdir(join(asset_root)) if 'DS_Store' not in f]

        if len(paths) == len(files):  ## mobility objects
            paths = [join(asset_root, p) for p in paths if not p.startswith('_')]
            paths.sort()
            if RANDOM_INSTANCE:
                random.shuffle(paths)
            file = join(paths[0], 'mobility.urdf')

        elif category == 'counter':
            file = join(ASSET_PATH, 'models', 'counter', 'urdf', 'kitchen_part_right_gen_convex.urdf')

        else:  ## bookshelf
            file = join(asset_root, 'model.sdf')
            if not isfile(file):
                file = join(asset_root, f'{category}.sdf')
        return file
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


def load_asset(category, x=0, y=0, yaw=0, floor=None, z=None, w=None, l=None, h=None, scale=1,
               verbose=False, maybe=False, moveable=False, RANDOM_INSTANCE=False):

    if verbose: print(f"\nLoading ... {category}", end='\r')

    """ ============= load body by category ============= """
    file = get_file_by_category(category, RANDOM_INSTANCE=RANDOM_INSTANCE)
    scale = get_scale_by_category(file=file, category=category, scale=scale)
    if file != None:
        if verbose: print(f"Loading ...... {abspath(file)}")
        scale = get_model_scale(file, l, w, h, scale, category)
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
        elif isinstance(floor, int) or (hasattr(floor, 'body') and isinstance(floor.body, int)):
            z = stable_z(body, floor)
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
    if moveable:
        object = Moveable(body, category=category)
    elif category.lower() == 'food':
        index = file.replace('/mobility.urdf', '')
        index = index[index.index('models/')+7:]
        index = index[index.index('./')+1:]
        object = Object(body, category=category, name=index)
    else:
        object = Object(body, category=category)

    return body, file, scale

# def load_asset(category, x, y, yaw, floor=None, z=None, w=None, l=None, scale=1,
#                verbose=False, maybe=False, moveable=False):
#
#     if verbose: print(f"\nLoading ... {category}")
#     height = 0
#
#     # ROOT_DIR = abspath(join(dirname(__file__), os.pardir))
#     asset_root = join(ASSET_PATH, 'models', category)  ## ROOT_DIR
#     if isdir(asset_root):
#         paths = [join(asset_root, f) for f in listdir(join(asset_root)) if isdir(join(asset_root, f))]
#         files = [join(asset_root, f) for f in listdir(join(asset_root)) if 'DS_Store' not in f]
#         with HideOutput(enable=True):
#             if len(paths) == len(files):  ## mobility objects
#                 paths.sort()
#                 file = join(paths[0], 'mobility.urdf')
#
#             elif category == 'counter':
#                 file = join(ASSET_PATH, 'models', 'counter', 'urdf', 'kitchen_part_right_gen_convex.urdf')
#
#             else:  ## bookshelf
#                 file = join(asset_root, 'model.sdf')
#                 if not isfile(file):
#                     file = join(asset_root, f'{category}.sdf')
#
#             if verbose: print(f"Loading ...... {file}")
#             body = load_model(file, scale=scale, fixed_base=True)
#             aabb = get_aabb(body)
#             width = aabb.upper[0] - aabb.lower[0]
#             length = aabb.upper[1] - aabb.lower[1]
#             height = aabb.upper[2] - aabb.lower[2]
#
#             ## reload with the correct scale
#             if w != None:
#                 if 'door' == category.lower():
#                     set_joint_position(body, get_joints(body)[1], -0.8)
#                 if 'dishwasher' == category.lower():
#                     set_joint_position(body, get_joints(body)[3], -0.66)
#
#                 if 'door' == category.lower():
#                     scale = (l / length + w / width) / 2  ##
#                 else:
#                     scale = min(l/length, w/width)  ##
#                 remove_body(body)
#                 body = load_model(file, scale=scale, fixed_base=True)
#
#     else:
#         body = create_box(w=w, l=l, h=1, color=BROWN, collision=True)
#
#     ## PLACE OBJECT
#     if z == None:
#         if category.lower() in ['oven']:
#             z = height / 2
#         elif isinstance(floor, tuple):
#             z = stable_z(body, floor[0], floor[1])
#         else:
#             z = stable_z(body, floor)
#     pose = Pose(point=Point(x=x, y=y, z=z), euler=Euler(yaw=yaw))
#     set_pose(body, pose)
#     if not maybe:
#         if category.lower() == 'veggieleaf':
#             set_color(body, DARK_GREEN, 0)
#         elif category.lower() == 'veggiestem':
#             set_color(body, WHITE, 0)
#         elif category.lower() == 'facetbase':
#             from pybullet_tools.bullet_utils import open_doors_drawers
#             open_doors_drawers(body)
#
#         if moveable:
#             object = Moveable(body, category=category)
#         else:
#             object = Object(body, category=category)
#     return body

def world_of_models(floor_width=5, floor_length = 5):
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

def find_point_for_single_push(body):
    (x_min, y_min, z_min), (x_max, y_max, z_max) = get_aabb(body)
    x_c = (x_max + x_min) / 2
    y_c = (y_max + y_min) / 2
    pts = [(x_c, y_min, z_max), (x_c, y_max, z_max), (x_min, y_c, z_max), (x_max, y_c, z_max)]

    poses = []
    for link in get_links(body):
        if '_4' not in get_link_name(body, link):
            poses.append(list(get_link_pose(body, link)[0])[:2])
    wheel_c = np.sum(np.asarray(poses), axis=0) / len(poses)

    max_dist = -np.inf
    max_pt = None
    for (x,y,z) in pts:
        dist = np.linalg.norm(np.asarray([x,y])-wheel_c)
        if dist > max_dist:
            max_dist = dist
            max_pt = (x,y,z)

    return max_pt


def visualize_point(point, world):
    z = 0
    if len(point) == 3:
        x, y, z = point
    else:
        x, y = point
    world.add_object(
        Moveable(create_box(.05, .05, .05, mass=1, color=(1, 0, 0, 1)), category='marker'),
        Pose(point=Point(x, y, z)))


def get_instances(category):
    from world_builder.partnet_scales import MODEL_SCALES, MODEL_HEIGHTS, OBJ_SCALES
    if category in MODEL_SCALES:
        return MODEL_SCALES[category]
    elif category in MODEL_HEIGHTS:
        return MODEL_HEIGHTS[category]['models']
    elif category.lower() in OBJ_SCALES:
        scale = OBJ_SCALES[category.lower()]
        category = [c for c in listdir(join(ASSET_PATH, 'models')) if c.lower() == category.lower()][0]
        asset_root = join(ASSET_PATH, 'models', category)
        indices = [f for f in listdir(join(asset_root)) if isdir(join(asset_root, f))]
        return {k : scale for k in indices}
    else:
        print(f'world_builder.utils.get_instances({category}) didnt find any models')
        assert NotImplementedError()
