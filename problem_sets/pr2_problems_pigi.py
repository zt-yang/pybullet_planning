from world_builder.loaders_partnet_kitchen import *
from world_builder.actions import pull_actions, pick_place_actions

from robot_builder.robot_builders import build_robot_from_args
from robot_builder.robots import PR2Robot

from problem_sets.problem_utils import problem_template


def test_full_kitchen_domain(args, world_loader_fn, x_max=3, **kwargs):
    kwargs['robot_builder_args'].update({
        'custom_limits': ((0.7, -2, 0), (x_max, 10, 3)),
        'initial_xy': (2, 4),
        'draw_base_limits': True
    })
    return problem_template(args, robot_builder_fn=build_robot_from_args, world_loader_fn=world_loader_fn, **kwargs)


def test_full_kitchen(args, **kwargs):
    """
    case 0: picking up or opening one obj
    case 1: storing objects in minifridge / cabinet
    case 2: put object in sink
    case 3: put object in braiser
    case 4: take object out of minifridge to oven counter
    """
    def loader_fn(world, **world_builder_args):
        world.set_skip_joints()
        world.set_learned_pose_list_gen(check_kitchen_placement)

        case = random.choice(world_builder_args['goal_variations'])
        world.note = case

        movables, counters = sample_full_kitchen(world, pause=False, reachability_check=False,
                                                 open_door_epsilon=0.5, make_doors_transparent=True)

        objects = []
        skeleton = []

        if case == 0:
            goals = sample_full_kitchen_goal_single_pick_or_pull(world, movables)

        elif case in [1, 31, 21, 11, 321, 991]:
            goals, objects = sample_full_kitchen_goal_rearrange_to_storage(world, movables, counters)

        elif case in [2, 3, 992, 993]:
            goals, skeleton, objects = sample_full_kitchen_goal_rearrange_to_sink_or_braiser(world, movables, counters)

        elif case in [4, 41]:
            goals, skeleton, objects = sample_full_kitchen_goal_pot_to_counter(world)

        elif case in [551, 552, 553, 554, 555]:
            goals, skeleton, objects = sample_full_kitchen_goal_demo(world, counters)

        else:
            raise Exception('Invalid case')

        world.remove_bodies_from_planning(goals=goals, exceptions=objects)
        return {'goals': goals, 'skeleton': skeleton}

    return test_full_kitchen_domain(args, loader_fn, **kwargs)


def sample_full_kitchen_goal_rearrange_to_sink_or_braiser(world, movables, counters):
    objects = []
    skeleton = []
    case = world.note
    arm = world.robot.arms[0]

    # sink_counters = [world.name_to_body(n) for n in ['sink_counter_left', 'sink_counter_right']]

    food, obj_bottom, objects = make_sure_obstacles(world, case, movables, counters, objects)
    world.add_to_cat(food, 'movable')
    lid = world.name_to_body('braiserlid')

    goals = ("test_object_grasps", food)
    goals = [('Holding', arm, lid)]
    goals = [('On', food, obj_bottom)]

    # if case in [3]:
    # goals = ("test_object_grasps", lid)
    # goals = [('Holding', arm, lid)]
    # goals = [('On', lid, counters[0])]

    # world.remove_object(lid)
    # objects.remove(lid)
    # goals = [('On', food, obj_bottom)]
    # goals = [('On', lid, counters[0].pybullet_name)]
    # goals = [('On', food, obj_bottom), ('On', lid, counters[0].pybullet_name)]

    # skeleton += [(k, arm, lid) for k in pick_place_actions]
    # skeleton += [(k, arm, food) for k in pick_place_actions]
    return goals, skeleton, objects


def sample_full_kitchen_goal_rearrange_to_storage(world, movables, counters):
    objects = []
    skeleton = []
    case = world.note
    arm = world.robot.arms[0]

    movable_type = random.choice(['edible', 'bottle']) if case == 1 else 'edible'

    container_choices = ['minifridge']  ## ['minifridge', 'cabinettop']

    ## two containers
    if case == 11:
        storage_body = world.cat_to_objects('edible')[0].supporting_surface.body
        objects.append(storage_body)
        from_storage = world.BODY_TO_OBJECT[storage_body]
        world.planning_config['camera_zoomins'].append({'name': from_storage.name, 'd': [1.2, 0.0, 2]})
        for door in from_storage.doors:
            if random.random() < 0.5:
                world.open_joint(door[0], door[1], extent=random.uniform(0.3, 0.9))  ## , random_gen=True
            objects.append(door)
        container_choices = [s for s in container_choices if s != from_storage.mobility_category.lower()]
    container_name = random.choice(container_choices)
    foods = world.cat_to_bodies(movable_type)
    counter = world.name_to_body('shelf')

    container = world.name_to_object(container_name)
    container_space = world.name_to_body(f'{container_name}::storage')
    world.planning_config['camera_zoomins'].append({'name': container.name, 'd': [1.2, 0.0, 2]})
    objects += [container_space] + container.doors

    goals = []

    """ open some doors """
    for door in container.doors:
        if (case == 1 and random.random() < 0.3) or \
                (case in [21, 31] and random.random() < 0.5) or \
                (case in [11, 321] and random.random() < 0.5):
            world.open_joint(door[0], door[1], random_gen=True)
        skeleton.extend([(k, arm, door) for k in pull_actions])
    if case == 991:
        world.open_doors_drawers(container, random_gen=True)
        foods = foods[:1]

    """ add movable objects """
    for f in foods[:1]:
        world.add_to_cat(f, 'movable')
        # world.add_highlighter(f)
        objects.append(f)
        skeleton.extend([(k, arm, f) for k in pick_place_actions])
        # goals.append(('In', f, container_space))
    # skeleton.append(('declare_store_in_space', '@edible'))

    # set_camera_target_body(foods[0])
    # goals += [('In', food, container_space) for food in foods[1:]]
    # goals += [('GraspedHandle', container.doors[0])]  ## right arm, (9, 1)
    # goals += [('OpenedJoint', container.doors[1])]  ## right arm, (9, 3)
    # goals += [('OpenedJoint', door) for door in container.doors]
    # goals = [('OpenedJoint', container.doors[0]), ('OpenedJoint', container.doors[1]),
    #          ('StoredInSpace', f'@{movable_type}', container_space)]
    # goals += [('On', food, counter) for food in foods]
    goals += [('In', food, container_space) for food in foods[:1]]
    # goals = [('StoredInSpace', f'@{movable_type}', container_space)]

    # ## debugging
    # goals = ('test_pose_inside_gen', (foods[0], container_space))
    goals = [('Holding', arm, food) for food in foods[:1]]

    if case in [21, 31]:  ##
        """ rearrange from braiser / sink to cabinet / fridge """
        _, obj_bottom, objects = make_sure_obstacles(world, case, movables, counters, objects)
        objects.extend(random.sample(counters[:2], 2))
        objects.append(obj_bottom)

    return goals, objects


def sample_full_kitchen_goal_single_pick_or_pull(world, movables):
    arm = world.robot.arms[0]

    doors = world.name_to_object('minifridge').doors  ## 'cabinettop'
    random.shuffle(doors)

    movable = movables[0]
    world.add_to_cat(movable, 'movable')
    world.add_highlighter(movable)

    goals = [('OpenedJoint', d) for d in doors]

    goals = [('HandleGrasped', doors[0], 'left')]  ## holding the handle
    goals = [('OpenedJoint', doors[0])]  ## having opened the joint
    goals = [('GraspedHandle', doors[0])]  ## having released the handle
    # goals = [('Holding', arm, movable)]

    return goals


def sample_full_kitchen_goal_demo(world, counters):
    objects = []
    skeleton = []
    case = world.note
    arm = world.robot.arms[0]

    ## demo of long horizon planning for RSS demo
    foods = world.cat_to_bodies('edible')
    food_cook = foods[0]
    food_block = foods[1]
    bottles = world.cat_to_bodies('bottle')
    bottle_block = bottles[1]
    oven_counter = world.name_to_body('oven_counter')
    counter = world.name_to_body('counter#1')
    sink = world.name_to_body('sink_bottom')
    sink_counter = world.name_to_body('sink_counter_right')
    braiser = world.name_to_body('braiser_bottom')
    lid = world.name_to_body('braiserlid')

    fridge = world.name_to_object('minifridge')
    fridge_door = fridge.doors[0]
    medicine = world.name_to_body('medicine#1')

    if case == 551:
        """ open door, remove bottle from sink, place food inside """
        cabinet = world.name_to_object('cabinettop')
        cabinet_door_1 = cabinet.doors[0]
        cabinet_door_2 = cabinet.doors[1]

        skeleton += [(k, arm, fridge_door) for k in pull_actions]
        skeleton += [(k, arm, bottle_block) for k in pick_place_actions]
        skeleton += [(k, arm, food_cook) for k in pick_place_actions]
        skeleton += [('wait-clean', food_cook)]

        goals = [
            ('OpenedJoint', fridge_door),
            ('On', bottle_block, sink_counter),
            ('On', food_cook, sink),
            ('Cleaned', food_cook),
        ]

        for c in [food_cook, bottle_block]:
            world.add_to_cat(c, 'movable')
        world.add_to_cat(sink, 'cleaningsurface')
        world.add_to_cat(food_cook, 'edible')

        objects += [sink]

        # world.open_doors_drawers(fridge.pybullet_name)
        # world.open_doors_drawers(cabinet.pybullet_name)
        # wait_unlocked()
        world.close_doors_drawers(fridge.pybullet_name)
        world.close_doors_drawers(cabinet.pybullet_name)

    elif case == 552:
        """ remove lid and food, then place in food """

        world.name_to_object('braiser_bottom').place_obj(world.BODY_TO_OBJECT[food_block], world=world)

        skeleton += [(k, arm, lid) for k in pick_place_actions]
        skeleton += [(k, arm, food_block) for k in pick_place_actions]
        skeleton += [(k, arm, food_cook) for k in pick_place_actions]
        # skeleton += [(k, arm, cabinet_door_1) for k in pull_actions]
        # skeleton += [(k, arm, cabinet_door_2) for k in pull_actions]
        # skeleton += [(k, arm, medicine) for k in pick_place_actions]

        goals = [
            ('On', lid, counter),
            ('On', food_block, counter),
            ('On', food_cook, braiser),
            # ('OpenedJoint', cabinet_door_1), ('OpenedJoint', cabinet_door_2),
            # ('On', medicine, braiser)
        ]

        for c in [food_cook, food_block, lid, medicine]:
            world.add_to_cat(c, 'movable')

        objects += [braiser]

    elif case == 553:
        """ place zucchini inside a pot with some food in it , seed=607348 """
        potato = world.name_to_object('potato')
        zucchini = world.name_to_object('zucchini')
        # world.name_to_object('braiser_bottom').place_obj(potato, world=world)
        counters[0].place_obj(zucchini, world=world)
        world.remove_object(lid)
        zucchini = zucchini.body
        world.add_to_cat(zucchini, 'movable')

        skeleton += [(k, arm, zucchini) for k in pick_place_actions]

        goals = [
            ('On', zucchini, braiser),
        ]

    elif case == 554:
        """ pick up medicine from sink with bottles in it , seed=892751 """
        artichoke = world.name_to_object('artichoke')
        sink_obj = world.name_to_object('sink_bottom')
        sink_obj.place_obj(artichoke, world=world)
        y = get_aabb_center(sink_obj.aabb())[1] - 0.1
        x = sink_obj.aabb().lower[0] + 0.13
        artichoke.adjust_pose(x=x, y=y, yaw=-1.7, world=world)
        artichoke = artichoke.body
        world.add_to_cat(artichoke, 'movable')

        ## for generating good solutions
        bottle1 = world.name_to_object('bottle#1')
        bottle1.adjust_pose(x=0.534, y=3.42, z=1.072, yaw=2.512, world=world)
        bottle2 = world.name_to_object('bottle#2')
        bottle2.adjust_pose(x=0.595, y=3.652, z=1.0, yaw=1.324, world=world)
        world.add_to_cat(bottle1.body, 'movable')
        skeleton += [(k, arm, bottle1.body) for k in pick_place_actions]

        skeleton += [(k, arm, artichoke) for k in pick_place_actions[:1]]

        goals = [
            ('On', bottle1.body, world.name_to_body('sink_counter_left')),
            ('Holding', 'left', artichoke),
        ]

    elif case == 555:
        """ place object inside a cabinet with three doors , seed=892751 """
        world.open_doors_drawers(fridge.pybullet_name, hide_door=True)
        fridge_storage = world.name_to_body('minifridge::storage')
        world.add_to_cat(medicine, 'movable')
        objects += fridge.doors

        skeleton += [(k, arm, medicine) for k in pick_place_actions]

        goals = [
            ('In', medicine, fridge_storage),
        ]

    else:
        assert False

    return goals, skeleton, objects


def sample_full_kitchen_goal_pot_to_counter(world):
    objects = []
    skeleton = []
    case = world.note
    arm = world.robot.arms[0]

    if case == 4:
        lid = world.name_to_object('braiserlid')
        world.remove_object(lid)

        ovencounter = world.name_to_body('ovencounter')
        world.add_to_cat(ovencounter, 'supporter')

        cabinettop = world.name_to_object('cabinettop')
        cabinettop_doors = cabinettop.doors
        cabinettop_space = world.name_to_object('cabinettop::storage')
        world.open_doors_drawers(cabinettop, hide_door=True)  ## , extent=1.5)

        ## put braiser in space
        braiserbody = world.name_to_object('braiserbody')
        world.add_to_cat(braiserbody.body, 'movable')
        cabinettop_space.place_obj(braiserbody)
        braiserbody.adjust_pose(yaw=PI / 2)
        if isinstance(world.robot, PR2Robot):
            dx = cabinettop_space.aabb().upper[0] - braiserbody.aabb().upper[0] - 0.05
            braiserbody.adjust_pose(dx=dx, dz=0.05)
            world.add_highlighter(braiserbody)
            ## not colliding with cabinettop
            if collided(braiserbody.body, [cabinettop], articulated=False, world=world, verbose=True, min_num_pts=0):
                print(f'braiserbody !!! object collided with cabinettop')

        objects = [braiserbody, ovencounter, cabinettop_space, cabinettop] + cabinettop_doors
        for door in cabinettop_doors:
            skeleton += [(k, arm, door) for k in pull_actions]
        skeleton += [(k, arm, braiserbody) for k in pick_place_actions]
        skeleton = [(k, arm, braiserbody) for k in pick_place_actions]
        skeleton = []

        goals = [('OpenedJoint', cabinettop_doors[0])]
        goals = [('OpenedJoint', cabinettop_doors[0]), ('OpenedJoint', cabinettop_doors[1])]
        goals = [('OpenedJoint', cabinettop_doors[0]), ('OpenedJoint', cabinettop_doors[1]),
                 ('On', braiserbody.body, ovencounter)]
        goals = [('On', braiserbody.body, ovencounter)]
        # goals = [('Holding', arm, braiserbody.body)]

        set_camera_target_body(cabinettop, dx=1, dy=0, dz=1.4)

    elif case == 41:
        lid = world.name_to_object('braiserlid')
        world.add_to_cat(lid, 'movable')

        shelf = world.name_to_object('shelf')
        ovencounter = world.name_to_body('ovencounter')
        world.add_to_cat(ovencounter, 'supporter')

        braiserbody = world.name_to_object('braiserbody')
        world.add_to_cat(braiserbody.body, 'movable')
        world.add_to_cat(braiserbody.body, 'supporter')
        shelf.place_obj(braiserbody)
        braiserbody.adjust_pose(yaw=PI / 2)

        skeleton += [(k, arm, braiserbody) for k in pick_place_actions]
        skeleton = [(k, arm, braiserbody) for k in pick_place_actions]

        goals = [('On', braiserbody.body, ovencounter)]  ## , ('On', lid.body, braiserbody.body)

    else:
        assert False, 'Unknown'

    return goals, skeleton, objects


########################################################################################


def test_kitchen_dinner(args, **kwargs):
    def loader_fn(world, **world_builder_args):
        arm = world.robot.arms[0]
        world.set_skip_joints()
        if 'note' in world_builder_args:
            world.note = world_builder_args['note']

        objects = []
        skeleton = []
        goals = []

        movables, cabinets, counters, obstacles, x_food_min = \
            sample_full_kitchen(world, w=4, verbose=False, pause=False)
        # table, plates = sample_table_plates(world)
        tables, plates = sample_two_tables_plates(world)
        plates.reverse()

        ## for cooking
        braiser = world.name_to_object('braiser_bottom')
        world.add_to_cat(braiser, 'heatingsurface')
        world.add_to_cat(braiser, 'surface')
        # objects += [braiser]
        world.remove_object(world.name_to_object('braiserlid'))

        ## for cleaning
        sink = world.name_to_object('sink_bottom')
        world.add_to_cat(sink, 'cleaningsurface')
        world.add_to_cat(sink, 'surface')
        objects += [sink]

        ## for adding obstacles in the sink
        if False:
            bottles = world.cat_to_objects('bottle')
            # bottle = random.choice(bottles)
            i = 0
            for bottle in bottles:
                sink.place_obj(bottle, world=world)
                world.add_to_cat(bottle, 'movable')
                objects += [bottle.body]
                # goals += [('On', bottle, plates[i])]
                i += 1

        num_goal_objects = min([len(plates), len(movables), 1])
        num_plates = num_goal_objects
        num_movables = num_goal_objects

        if world.note == 'more_plates':
            num_plates = len(plates)
        if world.note == 'more_movables':
            num_movables = len(movables)

        for i in range(num_plates):
            world.add_to_cat(plates[i], 'surface')
            world.add_to_cat(plates[i], 'plate')
            objects += [plates[i]]

        for i in range(num_movables):
            world.add_to_cat(movables[i], 'movable')
            objects += [movables[i]]

        for i in range(num_goal_objects):
            goals += [('Served', movables[i], plates[i])]
            # goals += [('Cleaned', movables[i])]
            # goals += [('Cooked', movables[i])]
            # goals += [('On', movables[i], braiser.pybullet_name)]
            # goals += [('On', movables[i], sink.pybullet_name)]
            # break

        world.planning_config.update({
            'supporting_surfaces': world.summarize_supporting_surfaces(),
            'supported_movables': world.summarize_supported_movables()
        })
        world.remove_bodies_from_planning(goals=goals, exceptions=objects)

        return {'goals': goals, 'skeleton': skeleton}

    return test_full_kitchen_domain(args, loader_fn, x_max=4, **kwargs)




