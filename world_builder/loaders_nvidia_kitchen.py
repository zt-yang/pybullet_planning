from pybullet_tools.utils import invert
from world_builder.loaders import *

###############################################################################


def load_full_kitchen(world, load_cabbage=True, **kwargs):
    world.set_skip_joints()

    if world.robot is None:
        custom_limits = ((0, 4), (4, 13))
        robot = create_pr2_robot(world, base_q=(1.79, 6, PI / 2 + PI / 2),
                                 custom_limits=custom_limits, USE_TORSO=True)

    floor = load_floor_plan(world, plan_name='kitchen_v2.svg', **kwargs)
    world.remove_object(floor)

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


def place_in_nvidia_kitchen_space(obj, supporter_name, interactive=False, doors=[]):
    """ place an object on a supporter in the nvidia kitchen using saved poses for debugging """
    world = obj.world
    supporter = world.name_to_object(supporter_name)
    pose = get_nvidia_kitchen_hacky_pose(obj, supporter_name)
    if pose is not None:
        obj.set_pose(pose)
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
        print(f'\n{title} starting at', nice(get_pose(obj), one_tuple=False))
        obj.change_pose_interactive()
        link_pose = get_link_pose(supporter.body, supporter.link)
        print('relative_pose\t', link_pose)  ## ((1.0046393871307373, 8.843374252319336, 0.7221270799636841), (0.0, 0.0, 0.0, 1.0))
        pose_relative = multiply(invert(link_pose), get_pose(obj))
        print(f'{title} ending at\t{nice(pose_relative, one_tuple=False)}\n')

        ## close all doors that have been opened
        for door, extent in doors:
            world.close_joint(door)

    ## check if the object is in collision with the surface
    collided(obj.body, [world.name_to_object(supporter_name).body],
             world=world, verbose=True, tag='place_in_nvidia_kitchen_space')


def get_nvidia_kitchen_hacky_pose(obj, supporter_name):
    if obj is None:
        return None
    world = obj.world
    if isinstance(supporter_name, tuple):
        supporter_name = world.get_name(supporter_name)
    relative_poses = {
        ('fork', 'indigo_drawer_top'): ((0.141, -0.012, -0.033), (0.0, 0.0, 0.94, 0.34))
    }
    poses = {
        ('pot', 'indigo_tmp'): ((0.63, 8.88, 0.11), (0.0, 0.0, -0.68, 0.73)),
        ('microwave', 'hitman_tmp'): ((0.43, 6.38, 0.98), (0.0, 0.0, -1, 0)),
        ('vinegar-bottle', 'sektion'): ((0.75, 7.41, 1.24), (0.0, 0.0, 0.0, 1.0)), ## ((0.75, 7.3, 1.24), (0, 0, 0, 1)),
        ('vinegar-bottle', 'dagger'): ((0.45, 8.83, 1.54), (0.0, 0.0, 0.0, 1.0)),
        ('vinegar-bottle', 'indigo_tmp'): ((0.59, 8.88, 0.16), (0.0, 0.0, 0.0, 1.0)),
        ('vinegar-bottle', 'shelf_bottom'): ((0.64, 4.88, 0.89), (0.0, 0.0, 0.0, 1.0)),
        ('chicken-leg', 'shelf_bottom'): ((0.654, 5.062, 0.797), (0.0, 0.0, 0.97, 0.25)),
        ('cabbage', 'shelf_bottom'): ((0.668, 4.832, 0.83), (0.0, 0.0, 0.6, 0.8)),
        ('salt-shaker', 'sektion'): ((0.771, 7.071, 1.146), (0.0, 0.0, 0.175, 0.98)),
        ('pepper-shaker', 'sektion'): ((0.764, 7.303, 1.16), (0.0, 0.0, 0.95, 0.34)),
        # ('fork', 'indigo_drawer_top'): ((1.146, 8.831, 0.689), (0.0, 0.0, 0.94, 0.34))
    }
    key = (obj.shorter_name, supporter_name)
    if key in relative_poses:
        supporter = world.name_to_object(supporter_name)
        link_pose = get_link_pose(supporter.body, supporter.link)  ## ((0.593946099281311, 8.843374252319336, 0.7221270799636841), (0.0, 0.0, 0.0, 1.0))
        print('relative_pose\t', link_pose)
        return multiply(link_pose, relative_poses[key])
    if key in poses:
        return poses[key]
    for kk, pose in poses.items():
        if kk[0].lower == key[0].lower and kk[1] in key[1]:
            return pose
    return None


def load_nvidia_kitchen_movables(world: World, open_doors_for: list = []):

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
    for category, asset_name, rand_ins, name, supporter_name in [
        ('appliance', 'microwave', True, 'microwave', 'hitman_tmp'),
        ('food', 'MeatTurkeyLeg', True, 'chicken-leg', 'shelf_bottom'),
        ('food', 'VeggieCabbage', True, 'cabbage', 'shelf_bottom'),
        ('food', 'Salter', '3934', 'salt-shaker', 'sektion'),
        ('food', 'Salter', '5861', 'pepper-shaker', 'sektion'),
        ('utensil', 'PotBody', True, 'pot', 'indigo_tmp'),
        ('utensil', 'KitchenFork', True, 'fork', 'indigo_drawer_top'),
        # ('utensil', 'KitchenKnife', True, 'knife', 'indigo_drawer_top'),
    ]:
        movable = world.add_object(Moveable(
            load_asset(asset_name, x=0, y=0, yaw=random.uniform(-math.pi, math.pi), RANDOM_INSTANCE=rand_ins),
            category=category, name=name
        ))
        movable.supporting_surface = world.name_to_object(supporter_name)

        doors = supporter_to_doors[supporter_name] if supporter_name in supporter_to_doors else []
        place_in_nvidia_kitchen_space(movable, supporter_name, interactive=False, doors=doors)

        movables[name] = movable.body
        movable_to_doors[name] = doors

        if name in open_doors_for:
            for door, extent in doors:
                world.open_joint(door, extent=extent)

    return movables, movable_to_doors


def load_nvidia_kitchen_joints(world: World, open_doors: bool = False):

    """ load joints """
    supporter_to_doors = {}
    for body_name, door_names, pstn, supporter_name in [
            ('counter', ['chewie_door_left_joint', 'chewie_door_right_joint'], 1.4, 'sektion'),
            # ('counter', ['dagger_door_left_joint', 'dagger_door_right_joint'], 1, 'dagger'),
            # ('counter', ['indigo_door_left_joint', 'indigo_door_right_joint'], 1, 'indigo_tmp'),
            ('counter', ['indigo_drawer_top'], 1, 'indigo_drawer_top'),
            ('fridge', ['fridge_door'], 0.5, 'shelf_bottom'),
        ]:
        doors = []
        for door_name in door_names:
            door = world.add_joints_by_keyword(body_name, door_name)[0]
            doors.append((door, pstn))
        supporter_to_doors[supporter_name] = doors

    return supporter_to_doors
