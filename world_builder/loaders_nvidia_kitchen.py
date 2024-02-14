import random

from pybullet_tools.pr2_primitives import Conf, get_group_joints
from pybullet_tools.utils import invert, get_name, pairwise_collision
from pybullet_tools.bullet_utils import sample_pose, xyzyaw_to_pose

from world_builder.loaders import *

###############################################################################

default_supports = [
    ['appliance', 'microwave', True, 'microwave', 'hitman_tmp'],
    ['food', 'MeatTurkeyLeg', True, 'chicken-leg', 'shelf_bottom'],
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
    ('chicken-leg', 'indigo_tmp'): ((0.717, 8.714, 0.849), (0.0, -0.0, -0.99987, 0.0163)),
    ('chicken-leg', 'shelf_bottom'): ((0.654, 5.062, 0.797), (0.0, 0.0, 0.97, 0.25)),
    ('cabbage', 'shelf_bottom'): ((0.668, 4.862, 0.83), (0, 0, 0.747, 0.665)),
    # ('cabbage', 'upper_shelf'): ((1.006, 6.295, 0.461), (0.0, 0.0, 0.941, 0.338)),
    # ('cabbage', 'indigo_drawer_top'): ((1.12, 8.671, 0.726), (0.0, 0.0, 0.173, 0.985)),
    ('salt-shaker', 'sektion'): ((0.771, 7.071, 1.146), (0.0, 0.0, 0.175, 0.98)),
    ('pepper-shaker', 'sektion'): ((0.764, 7.303, 1.16), (0.0, 0.0, 0.95, 0.34)),
    ('fork', 'indigo_tmp'): ((0.767, 8.565, 0.842), (0.0, 0.0, 0.543415, 0.8395)),
}

saved_base_confs = {
    ('cabbage', 'shelf_bottom'): [
        (1.54, 4.693, 0.49, 2.081),
        ((1.353, 4.55, 0.651, 1.732), {0: 1.353, 1: 4.55, 2: 1.732, 17: 0.651, 61: 0.574, 62: 0.996, 63: -0.63, 65: -0.822, 66: -2.804, 68: -1.318, 69: -2.06}),
        ((1.374, 4.857, 0.367, -3.29), {0: 1.374, 1: 4.857, 2: -3.29, 17: 0.367, 61: 0.18, 62: 0.478, 63: 0.819, 65: -0.862, 66: 2.424, 68: -1.732, 69: 4.031}),
        ((1.43, 5.035, 0.442, -3.284), {0: 1.43, 1: 5.035, 2: -3.284, 17: 0.442, 61: 0.195, 62: 0.147, 63: 2.962, 65: -0.34, 66: 0.2, 68: -1.089, 69: 3.011}),
        ((1.369, 5.083, 0.366, 2.76), {0: 1.369, 1: 5.083, 2: 2.76, 17: 0.366, 61: 0.547, 62: -0.309, 63: 3.004, 65: -1.228, 66: 0.213, 68: -0.666, 69: -3.304}),
    ],
    ('cabbage', 'upper_shelf'): [
        ((1.2067, 5.65, 0.0253, 1.35), {0: 1.2067001768469119, 1: 5.649899504044555, 2: 1.34932216824778, 17: 0.025273033185228437, 61: 0.5622022164634521, 62: -0.11050738689251488, 63: 2.3065452971538147, 65: -1.382707387923262, 66: 1.2637544829338638, 68: -0.8836125960523644, 69: 2.9453111543097945}),
        ((1.166, 5.488, 0.222, 0.475), {0: 1.166, 1: 5.488, 2: 0.475, 17: 0.222, 61: 1.106, 62: 0.172, 63: 3.107, 65: -0.943, 66: 0.077, 68: -0.457, 69: -2.503}),
        ((1.684, 6.371, 0.328, 1.767), {0: 1.684, 1: 6.371, 2: 1.767, 17: 0.328, 61: 1.422, 62: 0.361, 63: 3.127, 65: -1.221, 66: 2.251, 68: -0.018, 69: 1.634}),
        ((1.392, 5.57, 0.141, 0.726), {0: 1.392, 1: 5.57, 2: 0.726, 17: 0.141, 61: 1.261, 62: -0.043, 63: 2.932, 65: -1.239, 66: 0.526, 68: -0.427, 69: 0.612}),
        ((1.737, 6.033, 0.279, -5.162), {0: 1.737, 1: 6.033, 2: -5.162, 17: 0.279, 61: 1.966, 62: 0.37, 63: 2.515, 65: -1.125, 66: -4.711, 68: -0.579, 69: 2.463}),
        ((1.666, 6.347, 0.404, 1.66), {0: 1.666, 1: 6.347, 2: 1.66, 17: 0.404, 61: 1.22, 62: 0.656, 63: 3.653, 65: -0.948, 66: -1.806, 68: -0.41, 69: 0.355}),

    ],
    ('fork', 'indigo_drawer_top'): [
        ((1.718, 8.893, 0.0, -2.825), {0: 1.718, 1: 8.893, 2: -2.825, 17: 0.0, 61: -0.669, 62: 0.358, 63: -0.757, 65: -0.769, 66: 3.234, 68: -0.274, 69: -0.961}),
        ((1.718, 8.324, 0.439, 1.462), {0: 1.718, 1: 8.324, 2: 1.462, 17: 0.439, 61: 1.081, 62: 0.608, 63: 1.493, 65: -0.693, 66: 2.039, 68: -1.16, 69: -2.243}),
        ((1.702, 9.162, 0.002, -2.329), {0: 1.702, 1: 9.162, 2: -2.329, 17: 0.002, 61: -0.59, 62: 0.137, 63: 0.951, 65: -0.15, 66: 0.185, 68: -0.112, 69: 0.449}),
    ],
    ('fork', 'indigo_tmp'): [
        ((1.273, 8.334, 0.15, 0.881), {0: 1.273, 1: 8.334, 2: 0.881, 17: 0.15, 61: 1.009, 62: 1.396, 63: 0.382, 65: -1.986, 66: -1.594, 68: -1.654, 69: -2.563}),
        ((1.493, 8.18, 0.039, 1.83), {0: 1.493, 1: 8.18, 2: 1.83, 17: 0.039, 61: 0.19, 62: 0.083, 63: 2.022, 65: -0.357, 66: 2.77, 68: -1.454, 69: 2.908}),
        ((1.837, 8.537, 0.0, 1.649), {0: 1.837, 1: 8.537, 2: 1.649, 17: 0.0, 61: 0.586, 62: 0.407, 63: -0.782, 65: -0.921, 66: -1.597, 68: -0.398, 69: -2.335}),
        ((1.307, 8.365, 0.293, 1.613), {0: 1.307, 1: 8.365, 2: 1.613, 17: 0.293, 61: 1.063, 62: 0.265, 63: 0.077, 65: -0.412, 66: -3.217, 68: -1.716, 69: -2.743}),
    ],
    ('chicken-leg', 'indigo_tmp'):  [
        ((1.785, 8.656, 0.467, 0.816), {0: 1.785, 1: 8.656, 2: 0.816, 17: 0.467, 61: 2.277, 62: 0.716, 63: -0.8, 65: -0.399, 66: 1.156, 68: -0.468, 69: -0.476}),
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
                                 custom_limits=custom_limits, USE_TORSO=True)

    floor = load_floor_plan(world, plan_name='kitchen_v2.svg', **kwargs)
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
        supporter.place_obj(obj, world=world)

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
        movable = world.add_object(Moveable(
            load_asset(asset_name, x=0, y=0, yaw=random.uniform(-math.pi, math.pi), RANDOM_INSTANCE=rand_ins),
            category=category, name=name
        ))
        movable.supporting_surface = world.name_to_object(supporter_name)

        interactive = name in []  ## 'cabbage'
        doors = supporter_to_doors[supporter_name] if supporter_name in supporter_to_doors else []
        place_in_nvidia_kitchen_space(movable, supporter_name, interactive=interactive, doors=doors)

        movables[name] = movable.body
        movable_to_doors[name] = doors

        if name in open_doors_for:
            for door, extent in doors:
                world.open_joint(door, extent=extent)

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
    ## --------- Special case for plates -------------
    results = check_plate_placement(world, body, surfaces, num_samples=num_samples, obstacles=obstacles)

    name = world.BODY_TO_OBJECT[body].shorter_name
    for surface in surfaces:
        body, _, link = surface
        key = (name, get_link_name(body, link))
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


#####################################################################################################


def learned_nvidia_bconf_list_gen(world, inputs, verbose=True):
    a, o, p = inputs[:3]
    robot = world.robot
    joints = robot.get_group_joints('base-torso')
    movable = world.BODY_TO_OBJECT[o].shorter_name

    to_return = []
    for key, bqs in saved_base_confs.items():
        if key[0] == movable:
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