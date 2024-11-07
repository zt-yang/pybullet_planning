import numpy as np
import math
import string

from pybullet_tools.utils import PI, create_box, TAN, Point, get_movable_joints, aabb_overlap, \
    BLACK, RGBA, YELLOW, set_all_static, set_color, get_aabb, get_link_name, get_links, link_from_name, \
    AABB, INF, clip, aabb_union, get_aabb_center, Pose, Euler, set_joint_position, \
    get_box_geometry, get_aabb_extent, multiply, GREY, create_shape_array, create_body, STATIC_MASS, \
    set_renderer, quat_from_euler, get_joint_limits, get_aabb, draw_aabb, dump_joint, body_collision, \
    get_pose, set_pose
from pybullet_tools.camera_utils import set_camera_target_body

from world_builder.world_utils import LIGHT_GREY, read_xml, load_asset, FLOOR_HEIGHT, WALL_HEIGHT, ASSET_PATH
from world_builder.world import World, State
from world_builder.entities import Object, Location, Floor, Supporter, Surface, Movable, \
    Space, Steerable, Door

from robot_builder.robot_builders import create_pr2_robot

GRASPABLES = ['BraiserLid', 'Egg', 'VeggieCabbage', 'MeatTurkeyLeg', 'VeggieGreenPepper', 'VeggieArtichoke',
              'VeggieTomato', 'VeggieZucchini', 'VeggiePotato', 'VeggieCauliflower', 'MeatChicken']
GRASPABLES = [o.lower() for o in GRASPABLES]

NOT_MOVABLES = ['microwave']

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

    ## create a box
    if movable_category is None:
        z = get_aabb(supporter).upper[2]
        return world.add_box(
            Movable(create_box(.06, .06, .1, color=(0, 1, 0, 1)),
                    category=category, name=name),
            Pose(point=Point(x=xy[0], y=xy[1], z=z + .1 / 2)))

    return world.add_object(
        Movable(load_asset(movable_category, x=xy[0], y=xy[1], yaw=0, floor=supporter),
                category=category, name=name))


def create_house_floor(world, w=6, l=6, x=0.0, y=0.0):
    return world.add_object(
        Floor(create_box(w=w, l=l, h=FLOOR_HEIGHT, color=TAN, collision=True)),
        Pose(point=Point(x=x, y=y, z=-2 * FLOOR_HEIGHT)))


def create_floor_covering_base_limits(world, x_min=0):
    limits = world.robot.custom_limits
    _, x_max = limits[0]
    y_min, y_max = limits[1]
    w = x_max - x_min
    l = y_max - y_min
    x = (x_min + x_max) / 2
    y = (y_min + y_max) / 2
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


def get_room_boxes_given_room_door(room_region, door, height=0.4, thickness=0.2):
    door_aabb = get_aabb(door)
    dx = 0.2  ## door offset due to strange door aabb

    room_aabb = get_aabb(room_region)
    x0, y0 = room_aabb.lower[:2]
    width, length, _ = get_aabb_extent(room_aabb)
    walls = []
    for i, (x1, x2, y1, y2) in enumerate([
        [0, width, 0, 0+thickness], [0, width, length - thickness, length],
        [0, 0+thickness, 0, length], [width-thickness, width, 0, length],
    ]):
        box = Box(x1=x0+x1, x2=x0+x2, y1=y0+y1, y2=y0+y2, z1=0, z2=height)
        if aabb_overlap(box, door_aabb):
            (xx1, yy1, _), (xx2, yy2, _) = get_aabb(door)
            if i == 0:
                walls += [
                    Box(x1=x0+x1, x2=xx1, y1=yy2-thickness-dx, y2=yy2-dx, z1=0, z2=height),
                    Box(x1=xx2, x2=x0+x2, y1=yy2-thickness-dx, y2=yy2-dx, z1=0, z2=height),
                ]
            elif i == 1:
                walls += [
                    Box(x1=x0+x1, x2=xx1, y1=yy1+dx, y2=yy1+thickness+dx, z1=0, z2=height),
                    Box(x1=xx2, x2=x0+x2, y1=yy1+dx, y2=yy1+thickness+dx, z1=0, z2=height),
                ]
            elif i == 2:
                walls += [
                    Box(x1=xx2-thickness-dx, x2=xx2-dx, y1=y0+y1, y2=yy1, z1=0, z2=height),
                    Box(x1=xx2-thickness-dx, x2=xx2-dx, y1=yy2, y2=y0+y2, z1=0, z2=height),
                ]
            else:
                walls += [
                    Box(x1=xx1+dx, x2=xx1+thickness+dx, y1=y0+y1, y2=yy1, z1=0, z2=height),
                    Box(x1=xx1+dx, x2=xx1+thickness+dx, y1=yy2, y2=y0+y2, z1=0, z2=height),
                ]
        else:
            walls += [box]
    aabb = aabb_union(walls)
    lower, _ = aabb
    center = get_aabb_center(aabb)
    base_center = np.append(center[:2], lower[2:])
    # TODO: transform OOBB into a centered form
    return [(get_box_geometry(*get_aabb_extent(box)),
             Pose(get_aabb_center(box)-base_center)) for box in walls]


def create_room_given_room_door(room_region, door, color=GREY, **kwargs):
    shapes = get_room_boxes_given_room_door(room_region, door, **kwargs)
    geoms, poses = zip(*shapes)
    colors = len(shapes)*[color]
    collision_id, visual_id = create_shape_array(geoms, poses, colors)
    return create_body(collision_id, visual_id, mass=STATIC_MASS)


#######################################################


def load_experiment_objects(world, w=.5, h=.7, wb=.07, hb=.1, mass=1, EXIST_PLATE=True,
                            CABBAGE_ONLY=True, name='cabbage', color=(0, 1, 0, 1)) -> Object:

    if not CABBAGE_ONLY:
        table = world.add_object(
            Object(create_box(w, w, h, color=(.75, .75, .75, 1)), category='supporter', name='table'),
            Pose(point=Point(x=4, y=6, z=h / 2)))

    # cabbage = world.add_object(
    #     Movable(create_box(wb, wb, hb, mass=mass, color=color), name=name),
    #     Pose(point=Point(x=4, y=6, z=h + .1/2)))
    cabbage = world.add_object(Movable(load_asset('VeggieCabbage', x=2, y=0, yaw=0), name=name))
    cabbage.set_pose(Pose(point=Point(x=3, y=6, z=0.9 + .1/2)))

    if not CABBAGE_ONLY:
        sink = world.add_object(
            Object(create_box(w/4, w/4, h, color=(.25, .25, .75, 1)), category='supporter', name='sink'),
            Pose(point=Point(x=2, y=5, z=h / 2)))
        if EXIST_PLATE:
            plate = world.add_object(
                Movable(create_box(.07, .07, .1, mass=mass, color=(1, 1, 1, 1)), name='plate'),
                Pose(point=Point(x=2, y=5, z=h + .1 / 2)))

    return cabbage


def load_floor_plan(world, plan_name='studio1.svg', asset_renaming=None, debug=False, spaces=None, surfaces=None,
                    asset_path=ASSET_PATH, random_instance=False, load_movables=True, verbose=False, auto_walls=False):
    print(f'\nloading floor plan {plan_name}...')
    world.floorplan = plan_name
    if asset_renaming is not None:
        asset_renaming = {k.lower(): v.lower() for k, v in asset_renaming.items()}

    regions = []
    if spaces is not None:
        spaces = {k.lower(): v for k, v in spaces.items()}
        regions += list(spaces.keys())
    if surfaces is not None:
        surfaces = {k.lower(): v for k, v in surfaces.items()}
        regions += list(surfaces.keys())

    ## read xml file
    objects, X_OFFSET, Y_OFFSET, SCALING, FLOOR_X_MIN, FLOOR_X_MAX, FLOOR_Y_MIN, FLOOR_Y_MAX = \
        read_xml(plan_name, asset_path=asset_path, auto_walls=auto_walls)

    ## add reference floor used to ground objects z, will be removed afterward
    w = (FLOOR_X_MAX - FLOOR_X_MIN) / SCALING
    l = (FLOOR_Y_MAX - FLOOR_Y_MIN) / SCALING
    x = ((FLOOR_X_MIN + FLOOR_X_MAX) / 2 - X_OFFSET) / SCALING
    y = ((FLOOR_Y_MIN + FLOOR_Y_MAX) / 2 - Y_OFFSET) / SCALING
    floor = world.add_object(
        Floor(create_box(w=round(w, 1), l=round(l, 1), h=FLOOR_HEIGHT, color=TAN, collision=True)),
        Pose(point=Point(x=round(x, 1), y=round(y, 1), z=-2 * FLOOR_HEIGHT)))

    ## add each static object
    for name, o in objects.items():
        cat = o['category'].lower()
        if asset_renaming is not None and cat in asset_renaming:
            cat = asset_renaming[cat]

        x = (o['x'] - X_OFFSET) / SCALING
        y = (o['y'] - Y_OFFSET) / SCALING
        w = o['w'] / SCALING
        l = o['l'] / SCALING

        if cat in ['floor', 'office']:
            color = {'floor': TAN, 'office': GREY}[cat]
            z = {'floor': -2 * FLOOR_HEIGHT, 'office': -2 * FLOOR_HEIGHT + 0.03}[cat]
            world.add_box(
                Floor(create_box(w=round(w, 1), l=round(l, 1), h=FLOOR_HEIGHT, color=color, collision=True), name=name),
                Pose(point=Point(x=round(x, 1), y=round(y, 1), z=z)))
            continue

        elif cat == 'wall':
            """ usually auto generated by read_xlm when auto_walls is set to True """
            if 'north' in name or 'south' in name:
                w, l = l, w
            box = create_box(w=w, l=l, h=WALL_HEIGHT, color=GREY, collision=True)
            pose = Pose(point=Point(x=x, y=y, z=WALL_HEIGHT/2))
            obj = Object(box, name=name, category='wall')
            world.add_box(obj, pose)
            obj.add_text(name)
            continue

        elif cat == 'room':
            ## find the door to decide wall lengths
            # load_room(world, name, x, y, w, l, o['doors'], SCALING, asset_path=asset_path)
            continue

        ## add the object itself
        yaw = {0: 0, 90: PI / 2, 180: PI, 270: -PI / 2}[o['yaw']]
        ri = random_instance and not ('kitchen' in plan_name and cat == 'oven')
        obj_kwargs = dict(x=round(x, 1), y=round(y, 1), yaw=yaw, floor=floor, w=round(w, 1), l=round(l, 1))
        obj = world.add_object(Object(load_asset(cat, random_instance=ri, **obj_kwargs), category=cat))
        body = obj.body

        if cat == 'doorframe':
            obj.name = name
            door_joint = get_movable_joints(obj)[0]
            lower = get_joint_limits(body, door_joint)[0]
            set_joint_position(body, door_joint, lower)
            office_number = name.split('_')[-1]
            door = world.add_object(Door(body, joint=door_joint, name=f'door_{office_number}'))

            room_region = world.name_to_body(f'office_{office_number}')
            if room_region is not None:
                walls = create_room_given_room_door(room_region, body)
                world.add_object(Object(walls, name=f'wall_{office_number}', category='wall'))
                set_pose(walls, get_pose(room_region))

        elif cat == 'door':
            world.add_box(
                Floor(create_box(w=round(w, 1), l=round(l, 1), h=FLOOR_HEIGHT, color=TAN, collision=True), name=f'doorway_{name}'),
                Pose(point=Point(x=round(x, 1), y=round(y, 1), z=-2 * FLOOR_HEIGHT)))

        #######################################################
        ## add movable objects on designated places
        if not load_movables:
            continue

        if cat in regions:

            if debug:
                world.open_doors_drawers(body, ADD_JOINT=False)
                set_camera_target_body(body, dx=0.05, dy=0.05, dz=0.5)

            for link in get_links(body):
                # dump_link(body, link)
                # set_color(body, YELLOW, link)
                # draw_link_name(body, link)
                link_name = get_link_name(body, link)

                if surfaces is not None and cat in surfaces and link_name in surfaces[cat]:
                    surface = Surface(body, link=link)
                    world.add_object(surface)
                    for o in surfaces[cat][link_name]:
                        obj = surface.place_new_obj(o, random_instance=random_instance, verbose=verbose)
                        if verbose:
                            print(f'\tadding object {obj.name} to surface {surface.lisdf_name}')

                if spaces is not None and cat in spaces and link_name in spaces[cat]:
                    space = Space(body, link=link)
                    world.add_object(space)
                    for o in spaces[cat][link_name]:
                        obj = space.place_new_obj(o, random_instance=random_instance)
                        if verbose:
                            print(f'\tadding object {obj.name} to space {space.lisdf_name}')
            if debug:
                world.close_doors_drawers(body)

    world.close_all_doors_drawers(verbose=verbose)
    for surface in ['faucet_platform', 'shelf_top']:
        obj = world.name_to_object(surface)
        if obj is not None:
            obj.categories.remove('surface')
        # world.remove_body_from_planning(world.name_to_body(surface))
    set_renderer(True)
    return floor
