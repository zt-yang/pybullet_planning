import random
from nsplan_tools.nsplan_loaders import sample_clean_dish_v0
from pybullet_tools.utils import dump_world

def test_clean_dish_feg(world, **kwargs):
    ## hyperparameters for world sampling
    world.set_skip_joints()
    world.note = 1

    moveables, cabinets, counters, obstacles, x_food_min = \
        sample_clean_dish_v0(world, verbose=False, pause=False)

    dump_world()
    # input("here")

    goal = sample_clean_dish_goal(world)
    return goal


def sample_clean_dish_goal(world):
    objects = []

    print("\nobjects by category")
    print(world.OBJECTS_BY_CATEGORY)

    print("\nsupporting surfaces")
    world.summarize_supporting_surfaces()
    print("\n")

    # input("here")

    food = random.choice(world.cat_to_bodies('edible'))
    bottle = random.choice(world.cat_to_bodies('bottle'))
    bowl = random.choice(world.cat_to_bodies('bowl'))
    sink_counter_left = world.name_to_body('sink_counter_left')
    sink_counter_right = world.name_to_body('sink_counter_right')
    shelve = world.name_to_object('shelf_lower')

    # fridge = world.name_to_object('minifridge_storage')
    # fridge_door = fridge.doors[0]
    dishwasher = world.name_to_object('dishwasherbox')
    sink = world.name_to_object('sink#1')

    ## ----------------------------------------------------
    ## YANG: should be on the bottom of the sink instead of on the bounding box of the sink (floating)
    sink = world.name_to_body('sink_bottom')
    cabinet = world.name_to_object('cabinettop')
    cabinet_space = world.name_to_body('cabinettop_storage')
    world.add_to_cat(sink, 'CleaningSurface')
    objects += [sink, cabinet, cabinet_space]
    ## ----------------------------------------------------

    # objects += [fridge_door]

    world.add_to_cat(food, 'moveable')
    world.add_to_cat(bottle, 'moveable')
    world.add_to_cat(bowl, 'moveable')

    hand = world.robot.arms[0]
    # goal_candidates = [
    #     [('Holding', hand, bottle)],
    #     [('On', food, counter)],
    #     [('In', food, fridge)],
    #     [('OpenedJoint', fridge_door)],
    # ]
    # goals = random.choice(goal_candidates)

    # goals = [('On', bottle, counter)]
    # goals = [('In', bottle, dishwasher)]
    # goals = [('On', bottle, sink)]

    ## ---------------- testing --------------------
    # goals = [('On', food, sink)]
    # goals = [('On', bottle, sink)]
    # goals = [('Cleaned', bottle)]
    # goals = [('Cleaned', bowl)]
    # goals = [('In', bottle, cabinet_space), ('Cleaned', bottle)]
    # goals = [('In', bottle, cabinet_space), ('NoDirtyPlateInCabinet', cabinet_space)]
    # goals = [('On', bowl, sink)]
    # goals = [('Holding', hand, bowl)]
    # goals = [('On', bowl, sink), ('On', bottle, sink)]
    goals = [('In', bowl, cabinet_space), ('NoDirtyPlateInCabinet', cabinet_space)]
    # goals = [('On', bowl, shelve)]
    # goals = [('In', bowl, cabinet_space)]
    ## ------------------------------------------------

    ## removing all fridge doors to save planning time for testing domain
    for door in cabinet.doors:
        world.open_joint(door[0], door[1], hide_door=True)

    world.remove_bodies_from_planning(goals=goals, exceptions=objects)




    # ################################################
    # # debug
    #
    # from pybullet_tools.flying_gripper_utils import quick_demo_debug
    #
    # quick_demo_debug(world)
    #
    # input("after quick demo")



    return goals