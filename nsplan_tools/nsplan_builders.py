import random
from nsplan_tools.nsplan_loaders import sample_clean_dish_v0
from pybullet_tools.utils import dump_world

def test_clean_dish_feg(world, **kwargs):
    ## hyperparameters for world sampling
    world.set_skip_joints()
    world.note = 1

    # use kwargs to initialize obj_dict
    # sink_counter_left = world.name_to_body('sink_counter_left')
    # sink_counter_right = world.name_to_body('sink_counter_right')
    # shelve = world.name_to_object('shelf_lower')
    # sink = world.name_to_body('sink_bottom')
    # cabinet_space = world.name_to_body('cabinettop_storage')
    obj_dict = {0: {"class": "mug", "instance": None, "location": "shelf_lower"}}
    goal_dict = {0: {"goal_location": "cabinettop_storage", "goal_state": "clean"}}
    # goal_dict = {0: {"goal_location": "sink_bottom", "goal_state": "dirty"}}

    obj_dict, cabinets, counters, obstacles, x_food_min = \
        sample_clean_dish_v0(world, obj_dict, verbose=False, pause=False)

    dump_world()

    goal = sample_clean_dish_goal(world, obj_dict, goal_dict, **kwargs)
    return goal


# def sample_clean_dish_goal(world):
#     objects = []
#
#     print("\nobjects by category")
#     print(world.OBJECTS_BY_CATEGORY)
#
#     print("\nsupporting surfaces")
#     world.summarize_supporting_surfaces()
#     print("\n")
#
#     print("\nBODY_TO_OBJECT")
#     print(world.BODY_TO_OBJECT)
#     print("\n")
#
#     # input("here")
#
#     food = random.choice(world.cat_to_bodies('edible'))
#     bottle = random.choice(world.cat_to_bodies('bottle'))
#     bottles = world.cat_to_bodies('bottle')
#     bowl = random.choice(world.cat_to_bodies('bowl'))
#     bowls = world.cat_to_bodies('bowl')
#     mug = random.choice(world.cat_to_objects('mug'))
#     mugs = world.cat_to_bodies('mug')
#     pan = random.choice(world.cat_to_objects('pan'))
#
#     sink_counter_left = world.name_to_body('sink_counter_left')
#     sink_counter_right = world.name_to_body('sink_counter_right')
#     shelve = world.name_to_object('shelf_lower')
#
#     # fridge = world.name_to_object('minifridge_storage')
#     # fridge_door = fridge.doors[0]
#     dishwasher = world.name_to_object('dishwasherbox')
#     sink = world.name_to_object('sink#1')
#
#     ## ----------------------------------------------------
#     ## YANG: should be on the bottom of the sink instead of on the bounding box of the sink (floating)
#     sink = world.name_to_body('sink_bottom')
#     cabinet = world.name_to_object('cabinettop')
#     cabinet_space = world.name_to_body('cabinettop_storage')
#     cabinet_door = cabinet.doors[0]
#
#     world.add_to_cat(sink, 'CleaningSurface')
#     objects += [sink, cabinet, cabinet_space] + bottles
#     ## ----------------------------------------------------
#
#     # objects += [fridge_door]
#
#     world.add_to_cat(food, 'moveable')
#     [world.add_to_cat(b, 'moveable') for b in bottles]
#     world.add_to_cat(bowl, 'moveable')
#     world.add_to_cat(mug, 'moveable')
#     world.add_to_cat(pan, 'moveable')
#     print("\nobjects by category after adding to moveable")
#     print(world.OBJECTS_BY_CATEGORY)
#
#     hand = world.robot.arms[0]
#     # goal_candidates = [
#     #     [('Holding', hand, bottle)],
#     #     [('On', food, counter)],
#     #     [('In', food, fridge)],
#     #     [('OpenedJoint', fridge_door)],
#     # ]
#     # goals = random.choice(goal_candidates)
#
#     # goals = [('On', bottle, counter)]
#     # goals = [('In', bottle, dishwasher)]
#     # goals = [('On', bottle, sink)]
#
#     ## ---------------- testing --------------------
#     # goals = [('On', food, sink)]
#     # goals = [('On', bottle, sink)]
#     # goals = [('Cleaned', bottle)]
#     # goals = [('Cleaned', bowl)]
#     # goals = [('In', bottle, cabinet_space), ('Cleaned', bottle)]
#     # goals = [('In', bottle, cabinet_space), ('NoDirtyPlateInCabinet', cabinet_space)]
#     # goals = [('On', bowl, sink)]
#     # goals = [('Holding', hand, bowl)]
#     # goals = [('On', bowl, sink), ('On', bottle, sink)]
#     # goals = [('In', bowl, cabinet_space), ('NoDirtyPlateInCabinet', cabinet_space)]
#     # goals = [('On', bowl, shelve)]
#     # goals = [('In', bowl, cabinet_space)]
#     # goals = [('In', bowl, cabinet_space), ('NoDirtyPlateInCabinet', cabinet_space)]
#     ## ------------------------------------------------
#
#     # goals = [('Holding', hand, bottle)]
#     # goals = [('Holding', hand, bottle)]
#     # goals = [('On', bowl, sink)]
#     # goals = [('On', bowl, shelve)]
#     # goals = [('Cleaned', bottle)]
#     # goals = [('On', bottles[0], shelve)]
#     # goals = [('In', bowl, cabinet_space), ('NoDirtyPlateInCabinet', cabinet_space)]
#     # objects += bottles
#     # goals = [('In', bowls[1], cabinet_space), ('NoDirtyPlateInCabinet', cabinet_space)]
#     # goals = [('In', bowls[0], cabinet_space), ('In', bowls[1], cabinet_space), ('NoDirtyPlateInCabinet', cabinet_space)]
#
#     # 0410
#     # goals = [('Holding', hand, bowl)]
#     # goals = [('On', pan, shelve)]
#     goals = [('In', mug, cabinet_space), ('NoDirtyMugInCabinet', cabinet_space)]
#
#     # goals = [('In', bowl, cabinet_space), ('NoDirtyBowlInCabinet', cabinet_space), ('In', mug, cabinet_space), ('NoDirtyMugInCabinet', cabinet_space)]
#     # objects += bottles  ## TODO: removed temporarily to speed up testing
#
#     # goals = [('OpenedJoint', cabinet_door)]
#     objects += cabinet.doors
#
#     ## set initial state
#     world.add_clean_object(bowl)
#
#     # removing all fridge doors to save planning time for testing domain
#     for door in cabinet.doors:
#         # world.open_joint(door[0], door[1], hide_door=True)
#         world.close_joint(door[0], door[1])
#
#     world.remove_bodies_from_planning(goals=goals, exceptions=objects)
#
#     # ################################################
#     # # debug
#     #
#     # from pybullet_tools.flying_gripper_utils import quick_demo_debug
#     #
#     # quick_demo_debug(world)
#     #
#     # input("after quick demo")
#
#
#
#     return goals


def sample_clean_dish_goal(world, obj_dict, goal_dict, use_doors, **kwargs):

    objects = []

    ## print
    print("\nobjects by category")
    print(world.OBJECTS_BY_CATEGORY)
    print("\nsupporting surfaces")
    world.summarize_supporting_surfaces()
    print("\nBODY_TO_OBJECT")
    print(world.BODY_TO_OBJECT)

    ## get env fixture
    # sink_counter_left = world.name_to_body('sink_counter_left')
    # sink_counter_right = world.name_to_body('sink_counter_right')
    # shelve = world.name_to_body('shelf_lower')
    # sink = world.name_to_body('sink_bottom')
    # cabinet_space = world.name_to_body('cabinettop_storage')
    # dishwasher = world.name_to_object('dishwasherbox')
    # sink = world.name_to_object('sink#1')
    # fridge = world.name_to_object('minifridge_storage')
    # fridge_door = fridge.doors[0]
    # TODO: is this necessary?
    # objects += [sink, cabinet, cabinet_space, shelve, sink_counter_left, sink_counter_right]

    ## get objects
    # food = random.choice(world.cat_to_bodies('edible'))
    # bottle = random.choice(world.cat_to_bodies('bottle'))
    # bottles = world.cat_to_bodies('bottle')
    # bowl = random.choice(world.cat_to_bodies('bowl'))
    # bowls = world.cat_to_bodies('bowl')
    # mug = random.choice(world.cat_to_objects('mug'))
    # mugs = world.cat_to_bodies('mug')
    # pan = random.choice(world.cat_to_objects('pan'))

    ## make objects moveable
    # world.add_to_cat(food, 'moveable')
    # [world.add_to_cat(b, 'moveable') for b in bottles]
    # world.add_to_cat(bowl, 'moveable')
    # world.add_to_cat(mug, 'moveable')
    # world.add_to_cat(pan, 'moveable')
    # print("\nobjects by category after adding to moveable")
    # print(world.OBJECTS_BY_CATEGORY)
    for oi in obj_dict:
        world.add_to_cat(obj_dict[oi]["name"], 'moveable')

    ## get robot gripper
    hand = world.robot.arms[0]

    ## set goal condition
    # 1. hold something, goals = [('Holding', hand, bowl)]
    # 2. open door, goals = [('OpenedJoint', cabinet_doors[0])]
    # 3. clean, goals = [('Cleaned', bottle)]
    # 4. on some place, goals = [('On', bottle, shelve)]
    # 5. in cabinet, goals = [('In', mug, cabinet_space), ('NoDirtyMugInCabinet', cabinet_space)]
    # goals = [('On', obj_dict[0]['name'], world.name_to_body(obj_dict[0]['goal_location']))]
    goals = []
    for obj_id in goal_dict:
        obj_goal_spec = goal_dict[obj_id]
        if "goal_location" in obj_goal_spec:
            if obj_goal_spec["goal_location"] == "cabinettop_storage":
                # Note, "In" for cabinet
                goals.append(('In', obj_dict[obj_id]['name'], world.name_to_body(obj_goal_spec["goal_location"])))
                predicate = "NoDirty{}InCabinet".format(obj_dict[obj_id]["class"].capitalize())
                goals.append((predicate, obj_dict[obj_id]['name']))
            else:
                goals.append(('On', obj_dict[obj_id]['name'], world.name_to_body(obj_goal_spec["goal_location"])))
        if "goal_state" in obj_goal_spec:
            if obj_goal_spec["goal_state"] == "clean":
                goals.append(('Cleaned', obj_dict[obj_id]['name']))

    ## add additional objects that need to be moved
    # objects += bowls

    ## configure doors
    cabinet = world.name_to_object('cabinettop')
    if use_doors:
        objects += cabinet.doors
    # removing all fridge doors to save planning time for testing domain
    for door in cabinet.doors:
        if use_doors:
            world.close_joint(door[0], door[1])
        else:
            world.open_joint(door[0], door[1], hide_door=True)

    ## set initial state
    # world.add_clean_object(bowl)

    ## finally, remove irrelevant bodies to speed up planning
    world.remove_bodies_from_planning(goals=goals, exceptions=objects)

    return goals


