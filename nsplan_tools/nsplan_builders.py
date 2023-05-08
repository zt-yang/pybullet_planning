import random
from nsplan_tools.nsplan_loaders import sample_clean_dish_v0
from nsplan_tools.generate_semantic_specification import load_dict_from_json
from pybullet_tools.utils import dump_world


def test_clean_dish_feg(world, semantic_spec_file, **kwargs):
    ## hyperparameters for world sampling
    world.set_skip_joints()
    world.note = 1

    # use kwargs to initialize obj_dict
    # sink_counter_left = world.name_to_body('sink_counter_left')
    # sink_counter_right = world.name_to_body('sink_counter_right')
    # shelve = world.name_to_object('shelf_lower')
    # sink = world.name_to_body('sink_bottom')
    # cabinet_space = world.name_to_body('cabinettop_storage')
    # obj_dict = {0: {"class": "pan", "instance": None, "location": "shelf_lower"},
    #             1: {"class": "bottle", "instance": None, "location": "sink_bottom"}}
    # goal_dict = {0: {"location": "cabinettop_storage", "state": None}}
    # goal_dict = {0: {"goal_location": "sink_bottom", "goal_state": "clean"}}
    semantic_spec_dict = load_dict_from_json(semantic_spec_file)
    obj_dict = semantic_spec_dict["objects"]
    goal_dict =semantic_spec_dict["goals"]

    obj_dict, cabinets, counters, obstacles, x_food_min = \
        sample_clean_dish_v0(world, obj_dict, verbose=False, pause=False)

    dump_world()

    goal = sample_clean_dish_goal_v1(world, obj_dict, goal_dict, **kwargs)

    print("semantic specs")
    print("obj dict:", obj_dict)
    print("goal dict:", goal_dict)

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
#     # food = random.choice(world.cat_to_bodies('edible'))
#     # bottle = random.choice(world.cat_to_bodies('bottle'))
#     # bottles = world.cat_to_bodies('bottle')
#     # bowl = random.choice(world.cat_to_bodies('bowl'))
#     # bowls = world.cat_to_bodies('bowl')
#     # mug = random.choice(world.cat_to_objects('mug'))
#     # mugs = world.cat_to_bodies('mug')
#     # pan = random.choice(world.cat_to_objects('pan'))
#
#     mug = random.choice(world.cat_to_objects('mug'))
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
#     objects += [sink, cabinet, cabinet_space] # + bottles
#     ## ----------------------------------------------------
#
#     # objects += [fridge_door]
#
#     world.add_to_cat(mug, 'moveable')
#     # world.add_to_cat(food, 'moveable')
#     # [world.add_to_cat(b, 'moveable') for b in bottles]
#     # world.add_to_cat(bowl, 'moveable')
#     # world.add_to_cat(pan, 'moveable')
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
#     # objects += cabinet.doors
#
#     ## set initial state
#     # world.add_clean_object(bowl)
#
#     # removing all fridge doors to save planning time for testing domain
#     for door in cabinet.doors:
#         world.open_joint(door[0], door[1], hide_door=True)
#         # world.close_joint(door[0], door[1])
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


def sample_clean_dish_goal_v1(world, obj_dict, goal_dict, use_doors, **kwargs):

    objects = []

    ## print
    print("\nobjects by category")
    print(world.OBJECTS_BY_CATEGORY)
    print("\nsupporting surfaces")
    world.summarize_supporting_surfaces()
    print("\nBODY_TO_OBJECT")
    print(world.BODY_TO_OBJECT)

    ## get env fixture
    sink_counter_left = world.name_to_body('sink_counter_left')
    sink_counter_right = world.name_to_body('sink_counter_right')
    shelve = world.name_to_body('shelf_lower')
    # sink = world.name_to_body('sink_bottom')
    # cabinet_space = world.name_to_body('cabinettop_storage')
    # dishwasher = world.name_to_object('dishwasherbox')
    # sink = world.name_to_object('sink#1')
    # fridge = world.name_to_object('minifridge_storage')
    # fridge_door = fridge.doors[0]

    # important: the part below is necessary
    sink = world.name_to_body('sink_bottom')
    cabinet = world.name_to_object('cabinettop')
    cabinet_space = world.name_to_body('cabinettop_storage')
    world.add_to_cat(sink, 'CleaningSurface')
    # TODO: using all environments will make planning slow, but we don't know where to move distractor objects a priori
    input("DEBUG: We are adding all locations. Planning will be slow. Press key to confirm")
    objects += [sink, cabinet, cabinet_space, sink_counter_left, sink_counter_right, shelve]
    # objects += [sink, cabinet, cabinet_space, shelve]
    # objects += [sink, shelve]

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
    for oi in sorted(obj_dict.keys()):
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
    for obj_id in sorted(goal_dict.keys()):
        obj_goal_spec = goal_dict[obj_id]
        if obj_goal_spec["location"] is not None:
            if obj_goal_spec["location"] == "cabinettop_storage":
                # Note, "In" for cabinet
                goals.append(('In', obj_dict[obj_id]['name'], world.name_to_body(obj_goal_spec["location"])))
                predicate = "NoDirty{}InCabinet".format(obj_dict[obj_id]["class"].capitalize())
                goals.append((predicate, world.name_to_body(obj_goal_spec["location"])))
            else:
                goals.append(('On', obj_dict[obj_id]['name'], world.name_to_body(obj_goal_spec["location"])))
        if obj_goal_spec["state"] is not None:
            if obj_goal_spec["state"] == "clean":
                goals.append(('Cleaned', obj_dict[obj_id]['name']))

    ## add additional objects that need to be moved
    # objects += bowls
    for obj_id in sorted(obj_dict.keys()):
        if obj_id not in goal_dict:
            objects.append(obj_dict[obj_id]['name'])

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
    for oi in sorted(obj_dict.keys()):
        if obj_dict[oi]["state"] == "clean":
            add_clean_object(world, obj_dict[oi]["name"])

    ## finally, remove irrelevant bodies to speed up planning
    world.remove_bodies_from_planning(goals=goals, exceptions=objects)

    return goals

def add_clean_object(world, object):
    world.add_to_init(("Cleaned", object))

# def sample_clean_dish_goal_v2(world, obj_dict, goal_dict, use_doors, **kwargs):
#
#     objects = []
#
#     ## print
#     print("\nobjects by category")
#     print(world.OBJECTS_BY_CATEGORY)
#     print("\nsupporting surfaces")
#     world.summarize_supporting_surfaces()
#     print("\nBODY_TO_OBJECT")
#     print(world.BODY_TO_OBJECT)
#
#     ## get env fixture
#     loc_name_to_variable = {}
#     loc_name_to_variable['sink_counter_left'] = sink_counter_left = world.name_to_body('sink_counter_left')
#     loc_name_to_variable['sink_counter_right'] = sink_counter_right = world.name_to_body('sink_counter_right')
#     loc_name_to_variable['shelf_lower'] = shelve = world.name_to_object('shelf_lower')
#     loc_name_to_variable['sink_bottom'] = sink = world.name_to_body('sink_bottom')
#     loc_name_to_variable['cabinettop_storage'] = cabinet_space = world.name_to_body('cabinettop_storage')
#     cabinet = world.name_to_object('cabinettop')
#
#     objects += [sink, cabinet, cabinet_space]
#     world.add_to_cat(sink, 'CleaningSurface')
#
#     ## get objects
#     # for oi in obj_dict:
#     #     world.cat_to_objects(obj_dict[oi]["class"])
#     # food = random.choice(world.cat_to_bodies('edible'))
#     # bottle = random.choice(world.cat_to_bodies('bottle'))
#     # bottles = world.cat_to_bodies('bottle')
#     # bowl = random.choice(world.cat_to_bodies('bowl'))
#     # bowls = world.cat_to_bodies('bowl')
#     # mug = random.choice(world.cat_to_objects('mug'))
#     # mugs = world.cat_to_bodies('mug')
#     # pan = random.choice(world.cat_to_objects('pan'))
#
#     ## make objects moveable
#     # world.add_to_cat(food, 'moveable')
#     # [world.add_to_cat(b, 'moveable') for b in bottles]
#     # world.add_to_cat(bowl, 'moveable')
#     # world.add_to_cat(mug, 'moveable')
#     # world.add_to_cat(pan, 'moveable')
#     # print("\nobjects by category after adding to moveable")
#     # print(world.OBJECTS_BY_CATEGORY)
#     for oi in obj_dict:
#         world.add_to_cat(obj_dict[oi]["name"], 'moveable')
#
#     ## get robot gripper
#     hand = world.robot.arms[0]
#
#     ## set goal condition
#     # 1. hold something, goals = [('Holding', hand, bowl)]
#     # 2. open door, goals = [('OpenedJoint', cabinet_doors[0])]
#     # 3. clean, goals = [('Cleaned', bottle)]
#     # 4. on some place, goals = [('On', bottle, shelve)]
#     # 5. in cabinet, goals = [('In', mug, cabinet_space), ('NoDirtyMugInCabinet', cabinet_space)]
#     # goals = [('On', obj_dict[0]['name'], world.name_to_body(obj_dict[0]['goal_location']))]
#     goals = []
#     for obj_id in goal_dict:
#         obj_goal_spec = goal_dict[obj_id]
#         if "goal_location" in obj_goal_spec:
#             if obj_goal_spec["goal_location"] == "cabinettop_storage":
#                 # Note, "In" for cabinet
#                 goals.append(('In', obj_dict[obj_id]['name'], loc_name_to_variable[obj_goal_spec["goal_location"]]))
#                 predicate = "NoDirty{}InCabinet".format(obj_dict[obj_id]["class"].capitalize())
#                 goals.append((predicate, loc_name_to_variable[obj_goal_spec["goal_location"]]))
#             else:
#                 goals.append(('On', obj_dict[obj_id]['name'], loc_name_to_variable[obj_goal_spec["goal_location"]]))
#         if "goal_state" in obj_goal_spec:
#             if obj_goal_spec["goal_state"] == "clean":
#                 goals.append(('Cleaned', obj_dict[obj_id]['name']))
#
#     # mug = random.choice(world.cat_to_objects('mug'))
#     # cabinet_space = world.name_to_body('cabinettop_storage')
#     # goals = [('In', mug, cabinet_space), ('NoDirtyMugInCabinet', cabinet_space)]
#     # assert mug == obj_dict[0]['name']
#     # obj_goal_spec = goal_dict[0]
#     # assert cabinet_space == loc_name_to_variable[obj_goal_spec["goal_location"]]
#     random.choice([0,1,2])
#     random.choice([0, 1, 2])
#     random.choice([0, 1, 2])
#
#     print(goals)
#     input("goals, next?")
#
#     ## add additional objects that need to be moved
#     # objects += bowls
#
#     ## configure doors
#     cabinet = world.name_to_object('cabinettop')
#     if use_doors:
#         objects += cabinet.doors
#     # removing all fridge doors to save planning time for testing domain
#     for door in cabinet.doors:
#         if use_doors:
#             world.close_joint(door[0], door[1])
#         else:
#             world.open_joint(door[0], door[1], hide_door=True)
#
#     ## set initial state
#     # world.add_clean_object(bowl)
#
#     ## finally, remove irrelevant bodies to speed up planning
#     world.remove_bodies_from_planning(goals=goals, exceptions=objects)
#
#     return goals

