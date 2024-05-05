from pybullet_tools.utils import get_link_pose, remove_body

from world_builder.loaders import *


def load_rooms(world, DOOR_GAP=1.9):

    kitchen = world.add_object(
        Location(create_room(width=3, length=3, height=WALL_HEIGHT,
                             thickness=0.05, gap=DOOR_GAP, yaw=0.), name='kitchen'),
        Pose(point=Point(x=-1, y=0, z=0)))

    storage = world.add_object(
        Location(create_room(width=3, length=2, height=WALL_HEIGHT,
                             thickness=0.05, gap=DOOR_GAP, yaw=0.), name='storage'),
        Pose(point=Point(x=-1, y=-2.5, z=0)))

    laundry_room = world.add_object(
        Location(create_room(width=3, length=3, height=WALL_HEIGHT,
                             thickness=0.05, gap=DOOR_GAP, yaw=PI), name='laundry_room'),
        Pose(point=Point(x=4, y=0, z=0)))

    hallway = world.add_object(
        Location(create_box(w=2, l=5, h=FLOOR_HEIGHT, color=YELLOW, collision=True),
                 name='hallway'),
        Pose(point=Point(x=1.5, y=-1, z=-2 * FLOOR_HEIGHT)))

    living_room = world.add_object(
        Location(create_room(width=3, length=8, height=WALL_HEIGHT,
                             thickness=0.05, gap=2, yaw=-PI / 2), name='living_room'),
        Pose(point=Point(x=1.5, y=3, z=0)))

    aabb = aabb_union([get_aabb(body) for body in world.cat_to_bodies('location')])
    x, y, _ = get_aabb_center(aabb)
    w, l, _ = get_aabb_extent(aabb)
    floor = world.add_object(
        Floor(create_box(w=w, l=l, h=FLOOR_HEIGHT, color=TAN, collision=True)),
        Pose(point=Point(x=x, y=y, z=-2 * FLOOR_HEIGHT - 0.02)))

    return floor


def load_cart(world, cart_x=-0.25, cart_y=0, marker_name='marker', marker_size=.04, instance='100496'):
    world.set_skip_joints()

    floor = world.name_to_body('floor')
    load_kwargs = dict(floor=floor, random_instance=instance)

    cart = world.add_object(Steerable(
        load_asset('Cart', x=cart_x, y=cart_y, yaw=PI, **load_kwargs), category='cart'))
    x, y, z = find_point_for_single_push(cart.body)

    ## --------- mark the cart handle to grasp --------
    marker = world.add_object(
        Movable(create_box(marker_size, marker_size, marker_size, color=LIGHT_GREY),  ## (0, 1, 0, 1)
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
        Movable(create_box(.07, .07, .1, mass=mass, color=(0, 1, 0, 1)), name='cabbage'),
        Pose(point=Point(x=-1.3, y=0, z=h + .1 / 2)))

    kitchen = world.add_object(
        Location(create_room(width=3, length=3, height=WALL_HEIGHT,
                             thickness=0.05, gap=1.9, yaw=0.), name='kitchen'),
        Pose(point=Point(x=-1, y=0, z=0)))

    laundry_room = world.add_object(
        Location(create_box(w=2, l=5, h=FLOOR_HEIGHT, color=YELLOW, collision=True),
                 name='laundry_room'),
        Pose(point=Point(x=5, y=0, z=-2 * FLOOR_HEIGHT)))

    # doorway = world.add_object(
    #     Location(create_box(w=2, l=5, h=FLOOR_HEIGHT, color=YELLOW, collision=True),
    #                 name='doorway'),
    #     Pose(point=Point(x=8, y=0, z=-2 * FLOOR_HEIGHT)))

    storage = world.add_object(
        Location(create_box(w=3, l=2, h=FLOOR_HEIGHT, color=YELLOW, collision=True),
                 name='storage'),
        Pose(point=Point(x=-1, y=-2.5, z=-2 * FLOOR_HEIGHT)))

    cart, marker = load_cart(world)

    return cabbage, kitchen, laundry_room, storage, cart, marker


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


def load_blocked_kitchen(world, w=.5, h=.9, mass=1, DOOR_GAP=1.9):
    table = world.add_object(
        Object(create_box(w, w, h, color=(.75, .75, .75, 1)), category='supporter', name='table'),
        Pose(point=Point(x=-1.3, y=0, z=h / 2)))

    cabbage = world.add_object(
        Movable(create_box(.07, .07, .1, mass=mass, color=(0, 1, 0, 1)), name='cabbage'),
        Pose(point=Point(x=-1.3, y=0, z=h + .1 / 2)))

    kitchen = world.add_object(
        Location(create_room(width=3, length=3, height=WALL_HEIGHT,
                             thickness=0.05, gap=DOOR_GAP, yaw=0.), name='kitchen'),
        Pose(point=Point(x=-1, y=0, z=0)))

    cart, marker = load_cart(world)

    return cabbage, cart, marker


def load_blocked_sink(world, w=.5, h=.9, mass=1, DOOR_GAP=1.9):
    sink = world.add_object(
        Supporter(create_box(w, w, h, color=(.25, .25, .75, 1)), category='sink'),
        Pose(point=Point(x=-1.3, y=-3, z=h / 2)))

    storage = world.add_object(
        Location(create_room(width=3, length=3, height=WALL_HEIGHT,
                             thickness=0.05, gap=DOOR_GAP, yaw=0.), name='storage'),
        Pose(point=Point(x=-1, y=-3, z=0)))

    cart2, marker2 = load_cart(world, cart_x=-0.25, cart_y=-3, marker_name='marker2')

    return sink, cart2, marker2


def load_blocked_stove(world, w=.5, h=.9, mass=1, DOOR_GAP=1.9):
    stove = world.add_object(
        Supporter(create_box(w, w, h, color=(.75, .25, .25, 1)), category='stove'),
        Pose(point=Point(x=-1.3, y=3, z=h / 2)))

    storage2 = world.add_object(
        Location(create_room(width=3, length=3, height=WALL_HEIGHT,
                             thickness=0.05, gap=DOOR_GAP, yaw=0.), name='storage2'),
        Pose(point=Point(x=-1, y=3, z=0)))

    cart3, marker3 = load_cart(world, cart_x=-0.25, cart_y=3, marker_name='marker3')

    return stove, cart3, marker3


##############################################################################


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
    for (x, y, z) in pts:
        dist = np.linalg.norm(np.asarray([x,y])-wheel_c)
        if dist > max_dist:
            max_dist = dist
            max_pt = (x,y,z)

    return max_pt