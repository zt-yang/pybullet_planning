from world_builder.loaders_partnet_kitchen import *

from robot_builder.robot_builders import build_robot_from_args

from problem_sets.problem_utils import problem_template


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
    return problem_template(args, robot_builder_fn=build_robot_from_args, robot_builder_args=robot_builder_args,
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
        world.add_to_cat(cabbage, 'movable')
        world.add_to_cat(distractor, 'movable')
        world.add_highlighter(cabbage)

        skeleton = [
            ('pick_hand', arm, distractor),
            ('place_hand', arm, distractor),
            ('pick_hand', arm, cabbage),
            ('place_hand', arm, cabbage),
        ]

        goals = [('Holding', arm, cabbage)]
        goals = [('On', cabbage, shelf)]
        return {'goals': goals, 'skeleton': skeleton}

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
                world.add_to_cat(item, 'movable')
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
                world.add_to_cat(item, 'movable')
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
                world.add_to_cat(item, 'movable')
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

        return {'goals': goals, 'skeleton': skeleton}

    return test_mini_kitchen_domain(args, loader_fn, **kwargs)