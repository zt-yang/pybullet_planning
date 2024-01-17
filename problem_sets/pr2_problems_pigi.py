from world_builder.loaders_partnet_kitchen import *
from world_builder.robot_builders import build_robot_from_args

from problem_utils import test_template, pull_actions, pick_place_actions


####################################################


def test_mini_kitchen_domain(args, world_loader_fn, **kwargs):
    # config.update(
    #     {'robot_builder': create_pr2_robot,
    #      'robot_builder_args': robot_builder_args,
    #      'world_loader_fn': world_loader_fn,}
    # )

    if 'pr2' in args.domain_pddl:
        robot_builder_args = {'robot_name': 'pr2', 'custom_limits': ((0, -3), (3, 3)),
                              'base_q': (2, 0, 0), 'draw_base_limits': True}
    else:
        robot_builder_args = {'robot_name': 'feg', 'custom_limits': {0: (0, 3), 1: (-3, 3), 2: (0, 3)},
                              'initial_q': (2, 0, 1, 0, 0, 0), 'draw_base_limits': True}
        # args.domain_pddl = 'feg_kitchen.pddl'
        # args.stream_pddl = 'feg_stream_kitchen.pddl'
    return test_template(args, robot_builder_fn=build_robot_from_args, robot_builder_args=robot_builder_args,
                         world_loader_fn=world_loader_fn, **kwargs)


def test_mini_kitchen(args, **kwargs):
    def loader_fn(world, **world_builder_args):
        arm = world.robot.arms[0]
        food_ids, bottle_ids = load_kitchen_mini_scene(world)[:2]

        cabbage = food_ids[0].body
        distractor = food_ids[1].body
        counter = world.name_to_body('counter')
        shelf = world.name_to_body('shelf')
        objects = [cabbage, distractor, counter, shelf, world.robot]
        exclude_from_planning = [o for o in get_bodies() if o not in objects]
        for o in exclude_from_planning:
            world.remove_body_from_planning(o)
        world.add_to_cat(cabbage, 'moveable')
        world.add_to_cat(distractor, 'moveable')
        world.add_highlighter(cabbage)

        skeleton = [
            ('pick_hand', arm, distractor),
            ('place_hand', arm, distractor),
            ('pick_hand', arm, cabbage),
            ('place_hand', arm, cabbage),
        ]

        goals = [('Holding', arm, cabbage)]
        goals = [('On', cabbage, shelf)]
        return goals, skeleton

    return test_mini_kitchen_domain(args, loader_fn, **kwargs)


def test_mini_kitchen_data(args, **kwargs):
    def loader_fn(world, **world_builder_args):
        arm = world.robot.arms[0]
        food_ids, bottle_ids, medicine_ids = load_kitchen_mini_scene(world)

        skeleton = []
        goals = []
        objects = [world.robot.body]

        def get_all_bottles():
            """ all bottles go to the cabinet """
            cabinet = world.name_to_object('cabinet')
            objects.append(cabinet.body)
            cabinet_spaces = get_partnet_spaces(cabinet.path, cabinet.body)
            count = 0
            for door in cabinet.doors:
                body_joint = (cabinet.body, door)
                skeleton.append(('grasp_pull_handle', arm, body_joint))
                objects.append(body_joint)
                count += 1
                if count > 2:
                    break

            for b, _, l in cabinet_spaces:
                cabinetstorage = world.add_object(Space(b, l, name=f'fridge_storage'))
                objects.append(cabinetstorage.pybullet_name)
            cabinet = cabinetstorage.pybullet_name

            for item in bottle_ids:
                world.add_to_cat(item, 'moveable')
                item = item.body
                skeleton.append(('pick_hand', arm, item))
                skeleton.append(('place_hand', arm, item))
                goals.append(('In', item, cabinet))
                objects.append(item)

        def get_all_food():
            """ all food go into the fridge """
            minifridge = world.name_to_object('minifridge')
            objects.append(minifridge.body)
            count = 0
            for door in minifridge.doors:
                body_joint = (minifridge.body, door)
                skeleton.append(('grasp_pull_handle', arm, body_joint))
                objects.append(body_joint)
                count += 1
                if count > 2:
                    break

            minifridge_spaces = get_partnet_spaces(minifridge.path, minifridge.body)
            for b, _, l in minifridge_spaces:
                fridgestorage = world.add_object(Space(b, l, name=f'fridge_storage'))
                objects.append(fridgestorage.pybullet_name)
            minifridge = fridgestorage.pybullet_name

            for item in food_ids:
                world.add_to_cat(item, 'moveable')
                item = item.body
                skeleton.append(('pick_hand', arm, item))
                skeleton.append(('place_hand', arm, item))
                goals.append(('In', item, minifridge))
                objects.append(item)

        def get_all_medicine():
            """ all medicine go onto the shelf """
            shelf = world.name_to_body('shelf')
            objects.append(shelf)
            for item in medicine_ids:
                world.add_to_cat(item, 'moveable')
                item = item.body
                skeleton.append(('pick_hand', arm, item))
                skeleton.append(('place_hand', arm, item))
                goals.append(('On', item, shelf))
                objects.append(item)

        get_all_food()
        # get_all_bottles()
        # get_all_medicine()
        exclude_from_planning = [o for o in get_bodies() if o not in objects]
        for o in exclude_from_planning:
            world.remove_body_from_planning(o)

        return goals, skeleton

    return test_mini_kitchen_domain(args, loader_fn, **kwargs)


#################################################################


def test_full_kitchen_domain(args, world_loader_fn, x_max=3, **kwargs):
    kwargs['robot_builder_args'].update({
        'custom_limits': ((-0.5, -2, 0), (x_max, 10, 3)),
        'initial_xy': (2, 4),
        'draw_base_limits': True
    })
    return test_template(args, robot_builder_fn=build_robot_from_args, world_loader_fn=world_loader_fn, **kwargs)


def test_full_kitchen(args, **kwargs):
    """
    case 0: picking up or opening one obj
    case 1: storing objects in minifridge / cabinet
    case 2: put object in sink
    case 3: put object in braiser
    case 4: take object out of minifridge to oven counter
    """
    def loader_fn(world, **world_builder_args):
        arm = world.robot.arms[0]
        world.set_skip_joints()
        world.set_learned_pose_list_gen(check_kitchen_placement)

        case = random.choice([1])
        if 'note' in world_builder_args:
            case = world_builder_args['note']
        world.note = case

        moveables, cabinets, counters, obstacles, x_food_min = \
            sample_full_kitchen(world, verbose=False, pause=False, reachability_check=False)

        objects = []
        skeleton = []

        ## making demos for RSS
        if case in [551, 552, 553, 554, 555]:

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
                    world.add_to_cat(c, 'moveable')
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
                    world.add_to_cat(c, 'moveable')

                objects += [braiser]

            elif case == 553:
                """ place zucchini inside a pot with some food in it , seed=607348 """
                potato = world.name_to_object('potato')
                zucchini = world.name_to_object('zucchini')
                # world.name_to_object('braiser_bottom').place_obj(potato, world=world)
                counters[0].place_obj(zucchini, world=world)
                world.remove_object(lid)
                zucchini = zucchini.body
                world.add_to_cat(zucchini, 'moveable')

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
                artichoke.adjust_pose(x=x, y=y, theta=-1.7, world=world)
                artichoke = artichoke.body
                world.add_to_cat(artichoke, 'moveable')

                ## for generating good solutions
                bottle1 = world.name_to_object('bottle#1')
                bottle1.adjust_pose(x=0.534, y=3.42, z=1.072, theta=2.512, world=world)
                bottle2 = world.name_to_object('bottle#2')
                bottle2.adjust_pose(x=0.595, y=3.652, z=1.0, theta=1.324, world=world)
                world.add_to_cat(bottle1.body, 'moveable')
                skeleton += [(k, arm, bottle1.body) for k in pick_place_actions]

                skeleton += [(k, arm, artichoke) for k in pick_place_actions[:1]]

                goals = [
                    ('On', bottle1.body, world.name_to_body('sink_counter_left')),
                    ('Holding', 'left', artichoke),
                ]

            elif case == 555:
                """ place object inside a cabinet with three doors , seed=892751 """
                world.open_doors_drawers(fridge.pybullet_name, hide_door=True)
                fridge_storage = world.name_to_body('minifridge_storage')
                world.add_to_cat(medicine, 'moveable')
                objects += fridge.doors

                skeleton += [(k, arm, medicine) for k in pick_place_actions]

                goals = [
                    ('In', medicine, fridge_storage),
                ]

        elif case == 0:
            """ single pick or pull """
            doors = world.name_to_object('minifridge').doors  ## 'cabinettop'
            random.shuffle(doors)

            moveable = moveables[0]
            world.add_to_cat(moveable, 'moveable')
            world.add_highlighter(moveable)

            goals = [('OpenedJoint', d) for d in doors]

            goals = [('HandleGrasped', doors[0], 'left')]  ## holding the handle
            goals = [('OpenedJoint', doors[0])]  ## having opened the joint
            goals = [('GraspedHandle', doors[0])]  ## having released the handle
            # goals = [('Holding', arm, moveable)]

        elif case in [1, 31, 21, 11, 321, 991]:
            """ rearrange to cabinet / fridge """

            movable_type = random.choice(['edible', 'bottle']) if case == 1 else 'edible'

            container_choices = ['minifridge'] ## ['minifridge', 'cabinettop']

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
            container_space = world.name_to_body(f'{container_name}_storage')
            world.planning_config['camera_zoomins'].append({'name': container.name, 'd': [1.2, 0.0, 2]})
            objects += [container_space] + container.doors

            goals = []

            """ open some doors """
            for door in container.doors:
                if (case == 1 and random.random() < 0.3) or \
                        (case in [21, 31] and random.random() < 0.3) or \
                        (case in [11, 321] and random.random() < 0.5):
                    world.open_joint(door[0], door[1], random_gen=True)
                skeleton.extend([(k, arm, door) for k in pull_actions])
            if case == 991:
                world.open_doors_drawers(container, random_gen=True)
                foods = foods[:1]

            """ add movable objects """
            for f in foods[:1]:
                world.add_to_cat(f, 'moveable')
                # world.add_highlighter(f)
                objects.append(f)
                skeleton.extend([(k, arm, f) for k in pick_place_actions])
                # goals.append(('In', f, container_space))
            # skeleton.append(('declare_store_in_space', '@edible'))

            # set_camera_target_body(foods[0])
            # goals += [('Holding', arm, food) for food in foods[1:]]
            # goals += [('In', food, container_space) for food in foods[1:]]
            # goals += [('GraspedHandle', container.doors[0])]  ## right arm, (9, 1)
            # goals += [('OpenedJoint', container.doors[1])]  ## right arm, (9, 3)
            # goals += [('OpenedJoint', door) for door in container.doors]
            # goals = [('OpenedJoint', container.doors[0]), ('OpenedJoint', container.doors[1]),
            #          ('StoredInSpace', f'@{movable_type}', container_space)]
            # goals += [('On', food, counter) for food in foods]
            goals += [('In', food, container_space) for food in foods[:1]]
            # goals = [('StoredInSpace', f'@{movable_type}', container_space)]

            if case in [21, 31]:  ##
                """ rearrange from and to cabinet / fridge, from braiser or sink """
                _, obj_bottom, objects = make_sure_obstacles(world, case, moveables, counters, objects)
                objects.extend(random.sample(counters[:2], 2))
                objects.append(obj_bottom)

        elif case in [2, 3, 992, 993]:
            """ rearrange to and from sink bottom / pot bottom """
            # sink_counters = [world.name_to_body(n) for n in ['sink_counter_left', 'sink_counter_right']]

            food, obj_bottom, objects = make_sure_obstacles(world, case, moveables, counters, objects)
            world.add_to_cat(food, 'moveable')
            lid = world.name_to_body('braiserlid')

            goals = ('test_grasps', food)
            goals = [('Holding', arm, lid)]
            goals = [('On', food, obj_bottom)]

            # if case in [3]:
                # goals = ('test_grasps', lid)
                # goals = [('Holding', arm, lid)]
                # goals = [('On', lid, counters[0])]

                # world.remove_object(lid)
                # objects.remove(lid)
                # goals = [('On', food, obj_bottom)]
                # goals = [('On', lid, counters[0].pybullet_name)]
                # goals = [('On', food, obj_bottom), ('On', lid, counters[0].pybullet_name)]
                # skeleton += [(k, arm, lid) for k in pick_place_actions]
                # skeleton += [(k, arm, food) for k in pick_place_actions]

        elif case == 4:
            lid = world.name_to_object('braiserlid')
            world.remove_object(lid)

            ovencounter = world.name_to_body('ovencounter')
            world.add_to_cat(ovencounter, 'supporter')

            cabinettop = world.name_to_object('cabinettop')
            cabinettop_doors = cabinettop.doors
            cabinettop_space = world.name_to_object('cabinettop_storage')
            world.open_doors_drawers(cabinettop, hide_door=True)  ## , extent=1.5)

            ## put braiser in space
            braiserbody = world.name_to_object('braiserbody')
            world.add_to_cat(braiserbody.body, 'moveable')
            cabinettop_space.place_obj(braiserbody, world=world)
            braiserbody.adjust_pose(theta=PI/2, world=world)
            if isinstance(world.robot, PR2Robot):
                dx = cabinettop_space.aabb().upper[0] - braiserbody.aabb().upper[0] - 0.05
                braiserbody.adjust_pose(dx=dx, dz=0.05, world=world)
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
            world.add_to_cat(lid, 'moveable')

            shelf = world.name_to_object('shelf')
            ovencounter = world.name_to_body('ovencounter')
            world.add_to_cat(ovencounter, 'supporter')

            braiserbody = world.name_to_object('braiserbody')
            world.add_to_cat(braiserbody.body, 'moveable')
            world.add_to_cat(braiserbody.body, 'supporter')
            shelf.place_obj(braiserbody)
            braiserbody.adjust_pose(theta=PI/2)

            skeleton += [(k, arm, braiserbody) for k in pick_place_actions]
            skeleton = [(k, arm, braiserbody) for k in pick_place_actions]

            goals = [('On', braiserbody.body, ovencounter)] ## , ('On', lid.body, braiserbody.body)

        else:
            raise Exception('Invalid case')

        world.planning_config.update({
            'supporting_surfaces': world.summarize_supporting_surfaces(),
            'supported_movables': world.summarize_supported_movables()
        })
        world.remove_bodies_from_planning(goals=goals, exceptions=objects)
        # wait_if_gui()
        return goals, skeleton

    return test_full_kitchen_domain(args, loader_fn, **kwargs)


def test_kitchen_dinner(args, **kwargs):
    def loader_fn(world, **world_builder_args):
        arm = world.robot.arms[0]
        world.set_skip_joints()
        if 'note' in world_builder_args:
            world.note = world_builder_args['note']

        objects = []
        skeleton = []
        goals = []

        moveables, cabinets, counters, obstacles, x_food_min = \
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
                world.add_to_cat(bottle, 'moveable')
                objects += [bottle.body]
                # goals += [('On', bottle, plates[i])]
                i += 1

        num_goal_objects = min([len(plates), len(moveables), 1])
        num_plates = num_goal_objects
        num_movables = num_goal_objects

        if world.note == 'more_plates':
            num_plates = len(plates)
        if world.note == 'more_movables':
            num_movables = len(moveables)

        for i in range(num_plates):
            world.add_to_cat(plates[i], 'surface')
            world.add_to_cat(plates[i], 'plate')
            objects += [plates[i]]

        for i in range(num_movables):
            world.add_to_cat(moveables[i], 'moveable')
            objects += [moveables[i]]

        for i in range(num_goal_objects):
            goals += [('Served', moveables[i], plates[i])]
            # goals += [('Cleaned', moveables[i])]
            # goals += [('Cooked', moveables[i])]
            # goals += [('On', moveables[i], braiser.pybullet_name)]
            # goals += [('On', moveables[i], sink.pybullet_name)]
            # break

        world.planning_config.update({
            'supporting_surfaces': world.summarize_supporting_surfaces(),
            'supported_movables': world.summarize_supported_movables()
        })
        world.remove_bodies_from_planning(goals=goals, exceptions=objects)

        return goals, skeleton

    return test_full_kitchen_domain(args, loader_fn, x_max=4, **kwargs)




