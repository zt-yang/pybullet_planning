import copy
import numpy as np
from collections import namedtuple
import json

# WorldObject = namedtuple("Object", ["id", "class", "instance", "location", "state"], defaults=[None] * 3)
# GoalCondition = namedtuple("Goal", ["location", "state"], defaults=[None] * 2)

DISTRACTOR_OBJECTS = ["medicine", "bottle", "edible", "bowl", "mug", "pan"]
GOAL_OBJECTS = ["bowl", "mug", "pan"]
LOCATIONS = ["sink_counter_left", "sink_counter_right", "shelf_lower", "sink_bottom", "cabinettop_storage"]


def generate_semantic_specification(seed=0, max_num_goal_objs=1, max_num_distractor_objs=0):

    np.random.seed(seed)

    # example
    # objs_dict = {0: {"class": "pan", "instance": None, "location": "shelf_lower"},
    #                 1: {"class": "bottle", "instance": None, "location": "sink_bottom"}}
    # goals_dict = {0: {"goal_location": "cabinettop_storage"}}

    objs_dict = {}
    goals_dict = {}

    ########################
    # sample objects
    num_goal_objs = np.random.randint(1, max_num_goal_objs + 1)
    if max_num_distractor_objs:
        num_distractor_objs = np.random.randint(1, max_num_distractor_objs + 1)
    else:
        num_distractor_objs = 0

    goal_objs = []
    for oi in range(num_goal_objs):
        obj_cls = np.random.choice(GOAL_OBJECTS)
        obj_ins = None
        obj_loc = np.random.choice(LOCATIONS)
        obj_id = len(objs_dict)
        objs_dict[obj_id] = {"class": obj_cls, "instance": obj_ins, "location": obj_loc}
        goal_objs.append(obj_id)

    for oi in range(num_distractor_objs):
        obj_cls = np.random.choice(DISTRACTOR_OBJECTS)
        obj_ins = None
        obj_loc = np.random.choice(LOCATIONS)
        objs_dict[len(objs_dict)] = {"class": obj_cls, "instance": obj_ins, "location": obj_loc}

    ########################
    # sample goal conditions
    for obj_id in goal_objs:
        remaining_locations = copy.deepcopy(LOCATIONS)

        # remove initial location
        remaining_locations.remove(objs_dict[obj_id]["location"])

        goal_loc = np.random.choice(remaining_locations)

        if goal_loc == "cabinettop_storage":
            goals_dict[obj_id] = {"location": goal_loc, "state": None}
        else:
            # has the option to add clean as goal state
            goal_state = np.random.choice(["clean", None])
            goals_dict[obj_id] = {"location": goal_loc, "state": goal_state}

    dict = {"objects": objs_dict, "goals": goals_dict}
    return dict


def save_semantic_spec(dict, json_filename):
    with open(json_filename, "w") as fh:
        json.dump(dict, fh, sort_keys=True, indent=4)


def load_semantic_spec(json_filename):
    with open(json_filename, "r") as fh:
        dict = json.load(fh)
    return dict


if __name__ == "__main__":
    dict = generate_semantic_specification()
    save_semantic_spec(dict, "/home/weiyu/Research/nsplan/original/kitchen-worlds/outputs/test.json")





