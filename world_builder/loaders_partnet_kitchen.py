from __future__ import print_function
import random
import time
import sys
from pprint import pprint
import copy

from pybullet_tools.utils import euler_from_quat, set_camera_pose, draw_aabb, WHITE, get_link_pose, \
    get_joint_limits, unit_pose, draw_pose, stable_z, wait_unlocked, get_pose, set_pose, get_bodies
from pybullet_tools.bullet_utils import aabb_larger, open_joint, collided, in_list, equal
from pybullet_tools.pose_utils import get_learned_yaw, get_learned_poses

from world_builder.loaders import *
from world_builder.world_utils import get_partnet_doors, \
    get_partnet_spaces, FURNITURE_WHITE, adjust_for_reachability, get_partnet_links_by_type, \
    FURNITURE_YELLOW, FURNITURE_GREY, HEX_to_RGB


######################################################################################


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
        fridgestorage.place_obj(cabbage)
    else:
        return pose


def load_random_mini_kitchen_counter(world, movable_category='food', w=6, l=6, h=0.9, wb=.07, hb=.1,
                                     table_only=False, SAMPLING=False):
    """ each kitchen counter has one minifridge and one microwave
    """
    floor = world.add_object(
        Floor(create_box(w=w, l=l, h=FLOOR_HEIGHT, color=TAN, collision=True)),
        Pose(point=Point(x=w/2, y=l/2, z=-2 * FLOOR_HEIGHT)))

    h = random.uniform(0.3, 0.9)
    # h = random.uniform(0.4, 1.1)
    counter = world.add_object(Object(
        load_asset('KitchenCounter', x=w/2, y=l/2, yaw=math.pi, floor=floor, h=h,
                   random_instance=True, verbose=False), category='supporter', name='counter'))

    ## --- add cabage on an external table
    x, y = 1, 3
    # table = world.add_object(
    #     Object(create_box(0.5, 0.5, h, color=(.75, .75, .75, 1)), category='supporter', name='table'),
    #     Pose(point=Point(x=x, y=y, z=h / 2)))
    cat = movable_category.capitalize()
    cabbage = world.add_object(Movable(
        load_asset(cat, x=x, y=y, yaw=random.uniform(-math.pi, math.pi),
                   floor=floor, random_instance=True, sampling=SAMPLING),
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


def load_storage_mechanism(world, obj, epsilon=0.3, **kwargs):
    space = None
    ## --- ADD EACH DOOR JOINT
    doors = get_partnet_doors(obj.path, obj.body)
    for b, j in doors:
        world.add_joint_object(b, j, 'door')
        obj.doors.append((b, j))
        if random.random() < epsilon:
            world.open_joint(b, j, extent=0.9*random.random(), **kwargs)

    ## --- ADD ONE SPACE TO BE PUT INTO
    spaces = get_partnet_spaces(obj.path, obj.body)
    for b, _, l in spaces:
        space = world.add_object(Space(b, l, name=f'storage'))  ## f'{obj.category}::storage'
        break
    return doors, space


def load_fridge_with_food_on_surface(world, counter, name='minifridge',
                                     cabbage=None, SAMPLING=False):
    (x, y, _), _ = get_pose(counter)
    SAMPLING = cabbage if SAMPLING else False
    minifridge = world.add_object(Object(
        load_asset('MiniFridge', x=x, y=y, yaw=math.pi, floor=counter, sampling=SAMPLING,
                   random_instance=True), name=name))

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
        place_in_cabinet(fridgestorage, cabbage, world=world)

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
    while collided(world.robot, obstacles, verbose=verbose, world=world, tag='ensure robot cfree'):
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
                   random_instance=True, verbose=False),
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
    new_food = world.add_object(Movable(
        load_asset(cat, x=0, y=0, yaw=random.uniform(-math.pi, math.pi),
                   floor=floor, random_instance=True, sampling=SAMPLING),
        category=cat
    ))

    s = random_space()
    place_in_cabinet(s, new_food, world=world)
    max_trial = 20
    # print(f'\nfood ({max_trial})\t', new_food.name, nice(get_pose(new_food.body)))
    # print(f'first food\t', world.get_name_from_body(food), nice(get_pose(food)))
    while collided(new_food, [food], verbose=verbose, world=world, tag='load food'):
        s = random_space()
        max_trial -= 1
        place_in_cabinet(s, new_food)
        # print(f'\nfood ({max_trial})\t', new_food.name, nice(get_pose(new_food.body)))
        # print(f'first food\t', world.get_name_from_body(food), nice(get_pose(food)))
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
                                        w=w, l=l, h=center_z, random_instance=True), name=cat)
            else:
                obj = Object(load_asset(cat, x=center_x, y=center_y, yaw=yaw, floor=floor,
                                        w=w, l=l, h=2.5 * MIN_COUNTER_Z, random_instance=True), name=cat)
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
    microwave = counter.place_new_obj('microwave', scale=0.4 + 0.1 * random.random())
    microwave.set_pose(Pose(point=microwave.get_pose()[0], euler=Euler(yaw=math.pi)))

    ## add pot
    pot = counter.place_new_obj('kitchenpot', scale=0.2)
    pot.set_pose(Pose(point=pot.get_pose()[0], euler=Euler(yaw=yaw)))

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
            Object(load_asset('CabinetTop', x=0, y=min_shelf_y, z=SHELF_HEIGHT, yaw=math.pi, random_instance=ins),
                   category='cabinet', name='cabinet')
            )
    else:  ## if max_shelf_y < 0.5:
        ins = random.choice(['00003'])
        cabinet = world.add_object(
            Object(load_asset('CabinetTop', x=0, y=max_shelf_y, z=SHELF_HEIGHT, yaw=math.pi, random_instance=ins),
                   category='cabinet', name='cabinet')
            )

    food_ids, bottle_ids, medicine_ids = load_counter_movables(world, [counter, shelf], obstacles=[])

    # add camera
    camera_pose = Pose(point=Point(x=4.2, y=0, z=2.5), euler=Euler(roll=PI / 2 + PI / 8, pitch=0, yaw=-PI / 2))
    world.add_camera(camera_pose)
    # rgb, depth, segmented, view_pose, camera_matrix = world.camera.get_image()
    # wait_unlocked()

    # world.set_learned_pose_list_gen(learned_pigi_pose_list_gen)

    return food_ids, bottle_ids, medicine_ids


#################################################################


def load_counter_movables(world, counters, d_x_min=None, obstacles=[],
                          verbose=False, reachability_check=True):
    categories = ['food', 'bottle', 'medicine']
    start = time.time()
    robot = world.robot
    state = State(world)
    size_matter = len(obstacles) > 0 and obstacles[-1].name == 'braiser_bottom'
    satisfied = []
    if isinstance(counters, list):
        counters = {k: counters for k in ['food', 'bottle', 'medicine']}  ## , 'bowl', 'mug', 'pan'
    if d_x_min is None:
        d_x_min = - 0.3
    instances = {k: None for k in counters}
    n_objects = {k: 2 for k in categories}
    n_objects['bottle'] = 1

    if world.note in [31]:
        braiser_bottom = world.name_to_object('braiser_bottom')
        obstacles = [o for o in obstacles if o.pybullet_name != braiser_bottom.pybullet_name]
        move_lid_away(world, [world.name_to_object('floor')], epsilon=1.0)
    elif world.note in [11]:
        from_storage = random.choice(['minifridge', 'cabinettop'])
        counters['food'] = [world.name_to_object(f"{from_storage}::storage")]
    elif world.note in [551]:
        counters['food'] = [world.name_to_object(f"minifridge::storage")]
        counters['bottle'] = [world.name_to_object(f"sink_bottom")]
        counters['medicine'] = [world.name_to_object(f"cabinettop_storage")]
        from world_builder.asset_constants import DONT_LOAD
        DONT_LOAD.append('VeggieZucchini')
    elif world.note in [552]:
        counters['food'] = [world.name_to_object(f"sink_bottom")]
        counters['bottle'] = [world.name_to_object(f"sink_counter_left"),
                              world.name_to_object(f"sink_counter_right"), ]
        counters['medicine'] = [world.name_to_object(f"cabinettop_storage")]
        from world_builder.asset_constants import DONT_LOAD
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

    if verbose:
        print('-' * 20 + ' surfaces to sample movables ' + '-' * 20)
        pprint(counters)
        print('-' * 60)
        print('\nload_counter_movables(obstacles={})\n'.format([o.name for o in obstacles]))

    def check_size_matter(obj):
        if size_matter and aabb_larger(obstacles[-1], obj):
            satisfied.append(obj)

    def place_on_counter(obj_name, category=None, counter_choices=None, ins=True):
        if counter_choices is None:
            counter_choices = counters[obj_name]
        counter = random.choice(counter_choices)
        obj = counter.place_new_obj(obj_name, category=category, random_instance=ins)
        if verbose:
            print(f'          placed {obj} on {counter.name}')
        if 'bottom' not in counter.name:
            adjust_for_reachability(obj, counter, d_x_min)
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

            world.remove_object(obj, verbose=verbose)
            obj = place_on_counter(obj_name, category, **kwargs)
            check_size_matter(obj)

            trials -= 1
            if trials == 0:
                print('cant ensure_cfree')
                sys.exit()
                # return None
            again = check_conditions(obj)
        return obj

    ## add food items
    food_ids = []
    in_braiser = False
    for i in range(n_objects['food']):
        kwargs = dict()
        if world.note in [31] and not in_braiser:
            kwargs['counter_choices'] = [braiser_bottom]
        obj_cat = 'food'
        obj_category = 'edible'
        if instances['food'] is not None:
            kwargs['ins'] = instances['food'][i]
        obj = place_on_counter(obj_cat, category=obj_category, **kwargs)
        check_size_matter(obj)
        obj = ensure_cfree(obj, obstacles, obj_name=obj_cat, category=obj_category, **kwargs)
        in_braiser = in_braiser or 'braiser_bottom' in obj.supporting_surface.name
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

    # ## add bowl
    # bowl_ids = []
    # for i in range(1):
    #     obj = place_on_counter('bowl')
    #     obj = ensure_cfree(obj, obstacles, obj_name='bowl')
    #     # state = State(copy.deepcopy(world), gripper=state.gripper)
    #     bowl_ids.append(obj)
    #     obstacles.append(obj.body)
    #
    # ## add mug
    # mug_ids = []
    # for i in range(1):
    #     obj = place_on_counter('mug')
    #     obj = ensure_cfree(obj, obstacles, obj_name='mug')
    #     # state = State(copy.deepcopy(world), gripper=state.gripper)
    #     mug_ids.append(obj)
    #     obstacles.append(obj.body)
    #
    # ## add pan
    # pan_ids = []
    # for i in range(1):
    #     obj = place_on_counter('pan')
    #     obj = ensure_cfree(obj, obstacles, obj_name='pan')
    #     # state = State(copy.deepcopy(world), gripper=state.gripper)
    #     pan_ids.append(obj)
    #     obstacles.append(obj.body)

    if world.note in [3, 31]:
        put_lid_on_braiser(world)
    print('... finished loading movables in {}s'.format(round(time.time() - start, 2)))
    # world.summarize_all_objects()
    # wait_unlocked()
    return food_ids, bottle_ids, medicine_ids ## , bowl_ids, mug_ids, pan_ids


def move_lid_away(world, counters, epsilon=1.0):
    lid = world.name_to_body('braiserlid')  ## , obstacles=movables+obstacles
    counters_tmp = world.find_surfaces_for_placement(lid, counters)
    if len(counters_tmp) == 0:
        raise Exception('No counters found, try another seed')
    # world.add_highlighter(counters[0])

    if random.random() < epsilon:
        counters_tmp[0].place_obj(world.BODY_TO_OBJECT[lid])
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
                   random_instance=True, verbose=False), category='supporter', name='counter'))
    # set_camera_target_body(counter, dx=3, dy=0, dz=2)
    aabb_ext = get_aabb_extent(get_aabb(counter.body))
    y = l/2 - aabb_ext[1] / 2 + aabb_ext[0] / 2
    tray = world.add_object(Object(
        load_asset('Tray', x=w/2, y=y, yaw=math.pi, floor=counter,
                   random_instance=True, verbose=False), name='tray'))
    tray_bottom = world.add_surface_by_keyword(tray, 'tray_bottom')

    ## --- add cabage on an external table
    items = []
    for cat in categories:
        RI = '103104' if cat == 'Stapler' else True
        item = world.add_object(Movable(
            load_asset(cat, x=0, y=0, yaw=random.uniform(-math.pi, math.pi),
                       random_instance=RI), category=cat
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
    faucet_obj.adjust_pose(dx=x_new - x)


WALL_HEIGHT = 2.5
WALL_WIDTH = 0.1
COUNTER_THICKNESS = 0.05
diswasher = 'DishwasherBox'


def sample_kitchen_sink(world, floor=None, x=0.0, y=1.0, verbose=False, random_scale=1.0):

    if floor is None:
        floor = create_floor_covering_base_limits(world, x_min=-0.5)
        x = 0

    ins = True
    if world.note in [551, 552]:
        ins = '45305'
    base = world.add_object(Object(
        load_asset('SinkBase', x=x, y=y, yaw=math.pi, floor=floor,
                   random_instance=ins, verbose=verbose, random_scale=random_scale), name='sinkbase'))
    dx = base.lx / 2
    base.adjust_pose(dx=dx)
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
                   random_instance=ins, verbose=verbose, random_scale=random_scale), name='sink'))
    # TODO: instead of 0.05, make it random
    # align the front of the sink with the front of the base in the x direction
    dx = (base.aabb().upper[0] - sink.aabb().upper[0]) - 0.05 * random_scale
    # align the center of the sink with the center of the base in the y direction
    dy = sink.ly / 2 - (sink.aabb().upper[1] - y)
    sink.adjust_pose(dx=dx, dy=dy, dz=0)
    sink.adjust_pose(dx=0, dy=0, dz=-sink.height + COUNTER_THICKNESS)
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
                   random_instance=ins, verbose=verbose, random_scale=random_scale), name='faucet'))
    # adjust placement of faucet to be behind the sink
    place_faucet_by_sink(faucet, sink, world=world, gap=0.01 * random_scale)
    faucet.adjust_pose(dz=COUNTER_THICKNESS)

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
    # print(ordering)
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
        cabinet.adjust_pose(dx=0.3 * random_scale)
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
            kwargs['random_instance'] = cabinet.mobility_id
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
                                                    cabi_type=cabi_type, random_instance=ins)
    ## then load shelves
    add_shelves(counters+others, color, bled=bled, obstacles=obstacles)
    set_camera_target_body(cabinets[0])

    return cabinets, shelves


#####################################################


def load_braiser(world, supporter, x_min=None, verbose=False):
    ins = True
    if world.note in [551, 552]:
        ins = random.choice(['100038', '100023'])  ## larger braisers
    elif world.note in [553]:
        ins = random.choice(['100015'])  ## shallower braisers big enough for zucchini, ,'100693'
    braiser = supporter.place_new_obj('BraiserBody', random_instance=ins, verbose=verbose)
    braiser.adjust_pose(yaw=PI)
    if supporter.mobility_id == '102044':
        aabb = supporter.aabb()
        y = aabb.lower[1] + 2/5 * supporter.ly
        braiser.adjust_pose(y=y)
    set_camera_target_body(braiser)

    if x_min is None:
        x_min = supporter.aabb().upper[0] - 0.3
    adjust_for_reachability(braiser, supporter, x_min)
    set_camera_target_body(braiser)

    lid = braiser.place_new_obj('BraiserLid', category='movable', name='BraiserLid', max_trial=1,
                                random_instance=braiser.mobility_id, verbose=verbose)
    world.make_transparent(lid)
    put_lid_on_braiser(world, lid, braiser)

    braiser_bottom = world.add_surface_by_keyword(braiser, 'braiser_bottom')
    return braiser, braiser_bottom


def get_lid_pose_on_braiser(braiser):
    point, quat = get_pose(braiser)
    r, p, y = euler_from_quat(quat)
    return (point, quat_from_euler((r, p, y + PI / 4)))


def put_lid_on_braiser(world, lid=None, braiser=None):
    if lid is None:
        lid = world.name_to_object('BraiserLid')
    if braiser is None:
        braiser = world.name_to_object('BraiserBody')
    lid.set_pose(get_lid_pose_on_braiser(braiser))
    braiser.attach_obj(lid)


#####################################################


def load_all_furniture(world, ordering, floor, base, start, color, wall_height,
                       under_counter, on_base, full_body, tall_body):
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
            load_asset(category, yaw=math.pi, floor=floor, random_instance=ins, verbose=False),
            name=category, category=category))

    def load_furniture_base(furniture):
        return world.add_object(Object(
            load_asset('MiniFridgeBase', l=furniture.ly, yaw=math.pi, floor=floor,
                       random_instance=True, verbose=False),
            name=f'{furniture.category}Base', category=f'{furniture.category}Base'))

    counter_regions = []
    tall_obstacles = []
    right_counter_lower = right_counter_upper = base.aabb().upper[1]
    left_counter_lower = left_counter_upper = base.aabb().lower[1]
    x_lower = base.aabb().lower[0]

    adjust_y = {}

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
    world.name_to_object('minifridge').adjust_pose(dx=0.2)

    ## make wall
    l = right_counter_upper - left_counter_lower
    y = (right_counter_upper + left_counter_lower) / 2
    x = x_lower - WALL_WIDTH / 2
    wall = world.add_object(
        Supporter(create_box(w=WALL_WIDTH, l=l, h=wall_height, color=color), name='wall'),
        Pose(point=Point(x=x, y=y, z=wall_height/2)))
    # floor.adjust_pose(dx=x_lower - WALL_WIDTH)

    return counter_regions, tall_obstacles, adjust_y, x_lower, left_counter_lower, right_counter_upper


def create_counter_top(world, counters, counter_regions, base, color,
                       counter_x, counter_z, counter_w, adjust_y, x_lower):
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
            Supporter(create_box(w=counter_w, l=upper - lower,
                                 h=COUNTER_THICKNESS, color=color), name=name),
            Pose(point=Point(x=counter_x, y=(upper + lower) / 2, z=counter_z))))
        # print('lower, upper', (round(lower, 2), round(upper, 2)))

    ## to cover up the wide objects at the back
    if x_lower < base.aabb().lower[0]:
        x_upper = base.aabb().lower[0]
        x = (x_upper + x_lower) / 2
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


def load_cooking_appliances(world, ordering, counters, x_food_min, oven):
    obstacles = []
    microwave = None
    if 'MicrowaveHanging' not in ordering:
        wide_counters = [c for c in counters if c.ly > 0.66]
        if len(wide_counters) > 0:
            counter = wide_counters[0]
            microwave = counter.place_new_obj('microwave', scale=0.4 + 0.1 * random.random(),
                                              random_instance=True, verbose=False)
            microwave.set_pose(Pose(point=microwave.get_pose()[0], euler=Euler(yaw=math.pi)))
            obstacles.append(microwave)
    else:
        microwave = world.name_to_object('MicrowaveHanging')
    # if microwave is not None:
    #     counters.append(microwave)
    #     world.add_to_cat(microwave.body, 'supporter')

    braiser, braiser_bottom = load_braiser(world, oven, x_min=x_food_min)
    obstacles.extend([braiser, braiser_bottom])
    return microwave, obstacles


def load_storage_spaces(world, epsilon=0.0, make_doors_transparent=True, **kwargs):
    """ epsilon: probability of each door being open """
    if make_doors_transparent:
        world.make_doors_transparent()
    load_storage_mechanism(world, world.name_to_object('minifridge'), epsilon=epsilon, **kwargs)
    for cabi_type in ['cabinettop', 'cabinetupper']:
        cabi = world.cat_to_objects(cabi_type)
        if len(cabi) > 0:
            cabi = world.name_to_object(cabi_type)
            load_storage_mechanism(world, cabi, epsilon=epsilon, **kwargs)


def load_movables(world, counters, shelves, obstacles, x_food_min, reachability_check):
    all_counters = {
        'food': counters,
        'bottle': counters,  ##  + [sink_bottom],
        'medicine': shelves,
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

    ## load objects into reachable places
    food_ids, bottle_ids, medicine_ids = \
        load_counter_movables(world, all_counters, d_x_min=0.3, obstacles=obstacles,
                              reachability_check=reachability_check)
    movables = food_ids + bottle_ids + medicine_ids
    return movables


#####################################################


def sample_full_kitchen(world, verbose=True, pause=True, reachability_check=True,
                        open_door_epsilon=0.5, make_doors_transparent=False):
    h_lower_cabinets = 1
    dh_cabinets = 1.2 ## 0.8
    h_upper_cabinets = 0.768
    l_max_kitchen = 8

    under_counter = ['SinkBase', 'CabinetLower', 'DishwasherBox']
    on_base = ['MicrowaveHanging', 'MiniFridge']
    full_body = ['CabinetTall', 'Fridge', 'OvenCounter']
    tall_body = ['CabinetTall', 'Fridge', 'MiniFridge']

    ###############################################################

    """ step 0: sample an ordering of lower furniture """
    floor = create_floor_covering_base_limits(world)
    ordering = sample_kitchen_furniture_ordering()
    while 'SinkBase' not in ordering:
        ordering = sample_kitchen_furniture_ordering()

    """ step 1: sample a sink """
    start = ordering.index('SinkBase')
    sink_y = l_max_kitchen * start / len(ordering) + np.random.normal(0, 0.5)
    floor, base, counter_x, counter_w, counter_z, color, counters = \
        sample_kitchen_sink(world, floor=floor, y=sink_y)

    """ step 2: arrange furniture on the left and right of sink base, 
                along with the extended counter """
    wall_height = h_lower_cabinets + dh_cabinets + h_upper_cabinets + COUNTER_THICKNESS
    counter_regions, tall_obstacles, adjust_y, x_lower, left_counter_lower, right_counter_upper = \
        load_all_furniture(world, ordering, floor, base, start, color, wall_height,
                           under_counter, on_base, full_body, tall_body)

    """ step 3: make all the counters """
    create_counter_top(world, counters, counter_regions, base, color,
                       counter_x, counter_z, counter_w, adjust_y, x_lower)

    """ step 4: put upper cabinets and shelves """
    oven = world.name_to_object('OvenCounter')
    cabinets, shelves = load_full_kitchen_upper_cabinets(world, counters, x_lower, left_counter_lower,
                                                         right_counter_upper, others=[oven],
                                                         dz=dh_cabinets, obstacles=tall_obstacles)

    """ step 5: add additional surfaces in furniture """
    sink = world.name_to_object('sink')
    sink_bottom = world.add_surface_by_keyword(sink, 'sink_bottom')

    """ step 6: place electronics and cooking appliances on counters """
    x_food_min = base.aabb().upper[0] - 0.3
    microwave, obstacles = load_cooking_appliances(world, ordering, counters, x_food_min, oven)
    shelves += [microwave]

    """ step 7: place electronics and cooking appliances on counters """
    load_storage_spaces(world, epsilon=open_door_epsilon, make_doors_transparent=make_doors_transparent, verbose=verbose)

    """ step 8: place movables on counters """
    movables = load_movables(world, counters, shelves, obstacles, x_food_min, reachability_check)

    set_camera_pose((4, 4, 3), (0, 4, 0))
    if pause:
        set_renderer(True)
        wait_unlocked()
    return movables, counters


def make_sure_obstacles(world, case, movables, counters, objects, food=None):
    assert case in [
        2, ## sink
        3, ## braiser
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
    # set_camera_target_body(obj, dx=d[0], dy=d[1], dz=d[2])
    world.planning_config['camera_zoomins'].append(
        {'name': world.BODY_TO_OBJECT[obj].name, 'd': d}
    )

    """ add obstacles to the from or to region """
    bottom_obj = world.BODY_TO_OBJECT[obj_bottom]
    obstacles = [a.pybullet_name for a in bottom_obj.supported_objects]
    start = time.time()
    time_allowed = 4
    while (case in [2, 21] and len(obstacles) <= 2 and random.random() < 1) \
            or (case in [3] and len(obstacles) == 0 and random.random() < 0.2):
        all_to_move = []

        """ add one goal unrelated obstacles """
        existing = [o.body for o in obstacles]
        something = None
        if case in [2]:
            something = [m for m in movables if m.body not in existing]
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
            for cat in ['edible', 'bottle']:
                objs = world.cat_to_objects(cat)
                if len(objs) > 0:
                    all_to_move += random.sample(objs, 2 if random.random() < 0.2 else 1)

        """ move objects to the clustered region """
        for something in all_to_move:
            pkwargs = dict(max_trial=5, obstacles=obstacles)
            result = bottom_obj.place_obj(something, **pkwargs)
            if result is not None:
                possible = [something.pybullet_name] + [a.pybullet_name for a in bottom_obj.supported_objects]
                obstacles += [o for o in possible if o not in obstacles]
            ## move objects away if there is no space
            else:
                counters_tmp = world.find_surfaces_for_placement(something, counters)
                counters_tmp[0].place_obj(something, **pkwargs)

        if time.time() - start > time_allowed:
            print('cant make_sure_obstacles')
            sys.exit()

    """ make sure obstacles can be moved away """
    for o in obstacles:
        world.add_to_cat(o, 'movable')
        # skeleton.extend([(k, arm, o) for k in pick_place_actions])
        # goals.append(('On', o.body, random.choice(sink_counters)))
        # goals = [('Holding', arm, o.body)]
        if o not in objects:
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
        world.add_to_cat(lid, 'movable')
        counters_tmp = move_lid_away(world, counters, epsilon=epsilon)
        if counters_tmp is not None:
            objects += [counters_tmp.pybullet_name]
            world.add_to_cat(counters_tmp, 'surface')
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


def sample_two_tables_plates(world, verbose=False):
    """ two tables side by side facing the kitchen counters, with four plates on each """
    x = random.uniform(3, 3.5)
    y1 = random.uniform(1, 3)

    table1 = sample_table(world, x=x, y=y1, verbose=verbose)
    plates = load_plates_on_table(world, table1, verbose)

    table2 = sample_table(world, x=x, y=y1, verbose=verbose)
    table2.adjust_next_to(table1, direction='+y', align='+x')
    plates += load_plates_on_table(world, table2, verbose)

    return [table1, table2], plates


def sample_table(world, random_instance=True, **kwargs):
    """ a table facing the kitchen counters, x < 4 """
    floor = world.name_to_body('floor')
    table = world.add_object(Supporter(
        load_asset('DiningTable', yaw=0, floor=floor, random_instance=random_instance, **kwargs)))
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


###################################################################################


def is_cabinet_top(world, space):
    return 'space' in world.get_name(space) or 'storage' in world.get_name(space) \
        or 'space' in world.get_type(space) or 'storage' in world.get_type(space)


def check_kitchen_placement(world, body, surface, **kwargs):
    if isinstance(surface, list):
        surface = surface[0]

    body_id = world.get_mobility_identifier(body)
    if isinstance(body_id, int): ## reachable space, feg
        return []
    if isinstance(surface, tuple):
        surface_body = surface[0]
        surface_point = get_pose(surface[0])[0]
        surface_aabb = get_aabb(surface[0], link=surface[1])
    else:
        surface_body = surface
        surface_point = get_pose(surface)[0]
        surface_aabb = get_aabb(surface)
    surface_id = world.get_mobility_identifier(surface)
    poses = get_learned_poses(body_id, surface_id, body, surface_body,
                              surface_point=surface_point, **kwargs)
    if surface_id == 'box':
        original_pose = get_pose(body)
        y_lower = surface_aabb.lower[1]
        y_upper = surface_aabb.upper[1]
        def random_y(pose):
            set_pose(body, pose)
            aabb = get_aabb(body)
            (x, y, z), quat = pose
            y = np.random.uniform(y_lower+(y-aabb.lower[1]), y_upper-(aabb.upper[1]-y))
            z += get_aabb_extent(surface_aabb)[2]/2  ## add counter thickness
            return (x, y, z), quat
        poses = [random_y(pose) for pose in poses]
        set_pose(body, original_pose)

    if len(poses) == 0 and is_cabinet_top(world, surface):
        pose = place_in_cabinet(surface, body, place=False)
        poses = [pose]
    return poses


# def learned_pigi_pose_list_gen(world, body, surfaces, num_samples=30, obstacles=[], verbose=True):
#


######################################################################################


if __name__ == '__main__':
    ordering = sample_kitchen_furniture_ordering()
