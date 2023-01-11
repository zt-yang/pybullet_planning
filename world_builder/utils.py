import time

import numpy as np
import untangle
from numpy import inf
from os.path import join, isdir, isfile, dirname, abspath
from os import listdir
import json
import random

from pybullet_tools.utils import unit_pose, get_aabb_extent, draw_aabb, RED, sample_placement_on_aabb, wait_unlocked, \
    set_pose, get_movable_joints, draw_pose, pose_from_pose2d, set_velocity, set_joint_states, get_bodies, \
    flatten, INF, inf_generator, get_time_step, get_all_links, get_visual_data, pose2d_from_pose, multiply, invert, \
    get_sample_fn, pairwise_collisions, sample_placement, is_placement, aabb_contains_point, point_from_pose, \
    aabb2d_from_aabb, is_center_stable, aabb_contains_aabb, get_model_info, get_name, get_pose, dump_link, \
    dump_joint, dump_body, PoseSaver, get_aabb, add_text, GREEN, AABB, remove_body, HideOutput, \
    stable_z, Pose, Point, create_box, load_model, get_joints, set_joint_position, BROWN, Euler, PI, \
    set_camera_pose, TAN, RGBA, sample_aabb, get_min_limit, get_max_limit, set_color, WHITE, get_links, \
    get_link_name, get_link_pose, euler_from_quat, get_collision_data, get_joint_name, get_joint_position
from pybullet_tools.bullet_utils import get_partnet_links_by_type
from pybullet_tools.logging import dump_json
from world_builder.paths import ASSET_PATH


FURNITURE_WHITE = RGBA(0.85, 0.85, 0.85, 1)
FURNITURE_GREY = RGBA(0.4, 0.4, 0.4, 1)
FURNITURE_YELLOW = RGBA(221/255, 187/255, 123/255, 1)

LIGHT_GREY = RGBA(0.5, 0.5, 0.5, 0.6)
DARK_GREEN = RGBA(35/255, 66/255, 0, 1)
FLOOR_HEIGHT = 1e-3
WALL_HEIGHT = 0.5

GRASPABLES = ['BraiserLid', 'Egg', 'VeggieCabbage', 'MeatTurkeyLeg', 'VeggieGreenPepper', 'VeggieArtichoke',
              'VeggieTomato', 'VeggieZucchini', 'VeggiePotato', 'VeggieCauliflower', 'MeatChicken']
GRASPABLES = [o.lower() for o in GRASPABLES]


SCALE_DB = abspath(join(dirname(__file__), 'model_scales.json'))

SAMPLER_DB = abspath(join(dirname(__file__), 'sampling_distributions.json'))
SAMPLER_KEY = "{x}&{y}"


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


def partnet_id_from_path(path):
    id = path.replace('/mobility.urdf', '')
    return id[id.rfind('/') + 1:]


def get_sampled_file(SAMPLING, category, ids):
    from world_builder.entities import Object
    dists = json.load(open(SAMPLER_DB, 'r'))
    dist = None
    if isinstance(SAMPLING, Object):
        key = SAMPLER_KEY.format(x=SAMPLING.category.lower(), y=category.lower())
        id = partnet_id_from_path(SAMPLING.path)
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


def get_file_by_category(category, RANDOM_INSTANCE=False, SAMPLING=False):
    file = None

    ## correct the capitalization because Ubuntu cares about it
    cats = [c for c in listdir(join(ASSET_PATH, 'models')) if c.lower() == category.lower()]
    if len(cats) > 0:
        category = cats[0]
    asset_root = join(ASSET_PATH, 'models', category)  ## ROOT_DIR
    if isdir(asset_root):
        ids = [f for f in listdir(join(asset_root))
               if isdir(join(asset_root, f)) and not f.startswith('_')]
        files = [join(asset_root, f) for f in listdir(join(asset_root))
                 if 'DS_Store' not in f and not f.startswith('_')]

        # if 'minifridge' in asset_root.lower():
        #     print('asset_root', asset_root)
        if len(ids) == len(files):  ## mobility objects
            paths = [join(asset_root, p) for p in ids]
            paths.sort()
            if RANDOM_INSTANCE:
                if isinstance(RANDOM_INSTANCE, str):
                    paths = [join(asset_root, RANDOM_INSTANCE)]
                else:
                    sampled = False
                    if SAMPLING:
                        result = get_sampled_file(SAMPLING, category, ids)
                        if isinstance(result, str):
                            paths = [join(asset_root, result)]
                            sampled = True
                    if not sampled:
                        random.shuffle(paths)
            file = join(paths[0], 'mobility.urdf')

        elif category == 'counter':
            file = join(ASSET_PATH, 'models', 'counter', 'urdf', 'kitchen_part_right_gen_convex.urdf')

    else:
        parent = get_parent_category(category)
        if parent is None:
            print('cant', category)
        cats = [c for c in listdir(join(ASSET_PATH, 'models', parent)) if c.lower() == category.lower()]
        if len(cats) > 0:
            category = cats[0]

        if parent is not None:
            file = join(ASSET_PATH, 'models', parent, category, 'mobility.urdf')

        else:  ## bookshelf
            file = join(asset_root, 'model.sdf')
            if not isfile(file):
                file = join(asset_root, f'{category}.sdf')
    return file


def get_scale_by_category(file=None, category=None, scale=1):
    from world_builder.partnet_scales import MODEL_HEIGHTS, MODEL_SCALES, OBJ_SCALES

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
            if parent is None:
                print('cant', category)
            if parent in MODEL_SCALES:
                scale = MODEL_SCALES[parent][category]

    return scale


def get_parent_category(category):
    from world_builder.partnet_scales import MODEL_SCALES
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
               scale=1, verbose=False, maybe=False, moveable=False,
               RANDOM_INSTANCE=False, SAMPLING=False):

    if verbose: print(f"\nLoading ... {category}", end='\r')

    """ ============= load body by category ============= """
    file = get_file_by_category(category, RANDOM_INSTANCE=RANDOM_INSTANCE, SAMPLING=SAMPLING)
    if verbose and file is not None:
        print(f"Loading ...... {abspath(file)}")

    scale = get_scale_by_category(file=file, category=category, scale=scale)
    if file is not None:
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
    # if moveable:
    #     object = Moveable(body, category=category)
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
    from .entities import Moveable
    z = 0
    if len(point) == 3:
        x, y, z = point
    else:
        x, y = point
    world.add_object(
        Moveable(create_box(.05, .05, .05, mass=1, color=(1, 0, 0, 1)), category='marker'),
        Pose(point=Point(x, y, z)))


def sort_instances(category, instances, get_all=False):
    keys = list(instances.keys())
    if not get_all:
        cat_dir = join(ASSET_PATH, 'models', category)
        if not isdir(cat_dir):
            return {}
        elif len(listdir(cat_dir)) > 0:
            keys = [k for k in keys if isdir(join(cat_dir, k))]
    if isinstance(keys[0], tuple):
        instances = instances
    elif not keys[0].isdigit():
        keys = list(set([k for k in keys]))
        instances = {k: instances[k] for k in keys}
        instances = dict(sorted(instances.items()))
    else:
        instances = {k: instances[k] for k in keys}
        instances = dict(sorted(instances.items()))
    return instances


def get_instances(category, **kwargs):
    from world_builder.partnet_scales import MODEL_SCALES, MODEL_HEIGHTS, OBJ_SCALES
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
            instances = {k : scale for k in indices}
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
    if not path.endswith('mobility.urdf'):
        return None
    idx = dirname(path)
    idx = idx.replace(dirname(idx), '')
    return idx[1:]


def HEX_to_RGB(color):
    color = color.lstrip('#')
    rgb = [int(color[i: i+2], 16)/255 for i in (0, 2, 4)] + [1]
    return tuple(rgb)


def adjust_for_reachability(obj, counter, x_min=None, body_pose=None, return_pose=False):
    if x_min is None:
        x_min = counter.aabb().upper[0] - 0.3
    if body_pose is None:
        body_pose = obj.get_pose()
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
        obtacles = surface.supported_objects
        regions = [(ya, yb)]
        for o in obtacles:
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


if __name__ == "__main__":
    reduce_model_scale('/home/yang/Documents/jupyter-worlds/assets/models/Food/MeatTurkeyLeg/old_scale.txt',
                       scale_down=10)