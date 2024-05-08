import random

from pybullet_tools.pr2_primitives import Conf, get_group_joints
from pybullet_tools.utils import invert, get_name, pairwise_collision, sample_placement_on_aabb, \
    get_link_pose, get_pose, set_pose, sample_placement, aabb_from_extent_center
from pybullet_tools.pose_utils import sample_pose, xyzyaw_to_pose, sample_center_top_surface
from pybullet_tools.bullet_utils import nice, collided, equal

from world_builder.world_utils import sort_body_indices
from world_builder.loaders import *

part_names = {
    'sektion': 'side cabinet',
    'chewie_door_left_joint': 'side cabinet left door',
    'chewie_door_right_joint': 'side cabinet right door',
    'indigo_drawer_top': 'top drawer space',
    'indigo_drawer_top_joint': 'top drawer',
    'indigo_tmp': 'counter top on the right',
    'hitman_tmp': 'counter top on the left',
    'braiserlid': 'pot lid',
    'braiserbody': 'pot body',
    'braiser_bottom': 'pot bottom',
    'front_right_stove': 'stove',
    'knob_joint_1': 'stove knob'
}

###############################################################################

default_supports = [
    ['appliance', 'microwave', True, 'microwave', 'hitman_tmp'],
    ['food', 'MeatTurkeyLeg', True, 'chicken-leg', 'shelf_top'],
    ['food', 'VeggieCabbage', True, 'cabbage', 'upper_shelf'],  ## ['shelf_bottom', 'indigo_drawer_top]
    ['food', 'Salter', '3934', 'salt-shaker', 'sektion'],
    ['food', 'Salter', '5861', 'pepper-shaker', 'sektion'],
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

saved_relposes = {
    ('fork', 'indigo_drawer_top'): ((0.141, -0.012, -0.033), (0.0, 0.0, 0.94, 0.34)),
    ('fork', 'upper_shelf'): ((1.051, 6.288, 0.42), (0.0, 0.0, 0.94, 0.338)),
    ('cabbage', 'upper_shelf'): ((-0.062, 0.182, -0.256), (-0.64, 0.301, 0.301, 0.64)),
    ('cabbage', 'indigo_drawer_top'): ((0.115, -0.172, 0.004), (0.0, 0.0, 0.173, 0.985)),
}

saved_poses = {
    ('pot', 'indigo_tmp'): ((0.63, 8.88, 0.11), (0.0, 0.0, -0.68, 0.73)),
    ('microwave', 'hitman_tmp'): ((0.43, 6.38, 1.0), (0.0, 0.0, 1.0, 0)),
    ('vinegar-bottle', 'sektion'): ((0.75, 7.41, 1.24), (0.0, 0.0, 0.0, 1.0)), ## ((0.75, 7.3, 1.24), (0, 0, 0, 1)),
    ('vinegar-bottle', 'dagger'): ((0.45, 8.83, 1.54), (0.0, 0.0, 0.0, 1.0)),
    ('vinegar-bottle', 'indigo_tmp'): ((0.59, 8.88, 0.16), (0.0, 0.0, 0.0, 1.0)),
    ('vinegar-bottle', 'shelf_bottom'): ((0.64, 4.88, 0.89), (0.0, 0.0, 0.0, 1.0)),
    # ('chicken-leg', 'indigo_tmp'): ((0.717, 8.714, 0.849), (0.0, -0.0, -0.99987, 0.0163)), ## grasp the meat end
    ('chicken-leg', 'indigo_tmp'): ((0.787, 8.841, 0.849), (0.0, 0.0, 0.239, 0.971)), ## grasp the bone end
    ('chicken-leg', 'shelf_bottom'): ((0.654, 5.062, 0.797), (0.0, 0.0, 0.97, 0.25)),
    ('chicken-leg', 'shelf_top'): ((0.654, 4.846, 1.384), (-0.0, 0.0, -0.182, 0.983)),
    ('cabbage', 'shelf_bottom'): ((0.668, 4.862, 0.83), (0, 0, 0.747, 0.665)),
    # ('cabbage', 'upper_shelf'): ((1.006, 6.295, 0.461), (0.0, 0.0, 0.941, 0.338)),
    # ('cabbage', 'indigo_drawer_top'): ((1.12, 8.671, 0.726), (0.0, 0.0, 0.173, 0.985)),
    ('salt-shaker', 'sektion'): ((0.771, 7.071, 1.146), (0.0, 0.0, 1.0, 0)),
    ('pepper-shaker', 'sektion'): ((0.764, 7.303, 1.16), (0.0, 0.0, 1.0, 0)),
    ('fork', 'indigo_tmp'): ((0.767, 8.565, 0.842), (0.0, 0.0, 0.543415, 0.8395)),
}

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
    ]
}

FRONT_CAMERA_POINT = (3.9, 7, 1.3)
DOWNWARD_CAMERA_POINT = (2.9, 7, 3.3)


#######################################################################################################


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
        counter = world.name_to_object('hitman_tmp')
        counter.place_obj(cabbage)
        (_, y, z), _ = cabbage.get_pose()
        cabbage.set_pose(Pose(point=Point(x=0.85, y=y, z=z)))
        return cabbage
    return None


def load_braiser_bottom(world):
    braiser = world.name_to_body('braiserbody')
    world.add_object(Surface(braiser, link_from_name(braiser, 'braiser_bottom')))
    world.add_to_cat(world.name_to_body('braiserlid'), 'movable')


def load_cooking_mechanism(world):
    load_braiser_bottom(world)
    stove_knob = world.add_joints_by_keyword('oven', 'knob_joint_1')[0]
    # dishwasher_door = world.add_joints_by_keyword('dishwasher', 'dishwasher_door')[0]


def get_objects_for_open_kitchen(world):
    object_names = ['chicken-leg', 'fridge', 'fridge_door', 'fork',
                    'braiserbody', 'braiserlid', 'braiser_bottom',
                    'indigo_drawer_top', 'indigo_drawer_top_joint', 'indigo_tmp',
                    'sektion', 'chewie_door_left_joint', 'chewie_door_right_joint',
                    'salt-shaker', 'pepper-shaker',
                    'front_right_stove', 'knob_joint_1']
    objects = [world.name_to_body(name) for name in object_names]
    objects = sort_body_indices(objects)
    world.set_english_names(part_names)
    world.remove_bodies_from_planning([], exceptions=objects)

    print('reduce_objects_for_open_kitchen')
    print(f"\t{len(objects)} objects provided:\t {objects}")
    print(f"\t{len(world.BODY_TO_OBJECT.keys())} objects in world:\t {sort_body_indices(list(world.BODY_TO_OBJECT.keys()))}")
    return objects


def load_open_problem_kitchen(world, reduce_objects=False, open_doors_for=[]):
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
            'front_right_stove': ['BraiserBody'],
            'hitman_tmp': [],
            'indigo_tmp': ['BraiserLid'],
        },
    }
    custom_supports = {
        'cabbage': 'shelf_bottom',
        'fork': 'indigo_drawer_top'
    }
    load_full_kitchen(world, surfaces=surfaces, spaces=spaces, load_cabbage=False)
    movables, movable_to_doors = load_nvidia_kitchen_movables(world, open_doors_for=open_doors_for,
                                                              custom_supports=custom_supports)
    load_cooking_mechanism(world)

    objects = None
    if reduce_objects:
        objects = get_objects_for_open_kitchen(world)
    return objects, movables, movable_to_doors


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
            return pose, True
    for kk, pose in saved_poses.items():
        if kk[1] in key[1]:
            link_pose = get_link_pose(supporter.body, supporter.link)
            return multiply(link_pose, pose), True

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
             world=world, verbose=True, tag='place_in_nvidia_kitchen_space')


def load_nvidia_kitchen_movables(world: World, open_doors_for: list = [], custom_supports: dict = {}):

    for elems in default_supports:
        if elems[-2] in custom_supports:
            elems[-1] = custom_supports[elems[-2]]

    """ load joints """
    supporter_to_doors = load_nvidia_kitchen_joints(world)

    """ add surfaces """
    for body_name, surface_name in [
        ('fridge', 'shelf_bottom'),
        ('fridge', 'shelf_top'),
    ]:
        body = world.name_to_body(body_name)
        shelf = world.add_object(Surface(
            body, link=link_from_name(body, surface_name), name=surface_name, category='supporter'
        ))

    # ## left half of the kitchen
    # set_camera_pose(camera_point=[3.5, 9.5, 2], target_point=[1, 7, 1])

    """ load movables """
    movables = {}
    movable_to_doors = {}
    for category, asset_name, rand_ins, name, supporter_name in default_supports:
        movable = world.add_object(Movable(
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


def learned_nvidia_bconf_list_gen(world, inputs, verbose=True):
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
    return to_return
