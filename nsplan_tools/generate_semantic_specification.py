import os
import copy
import numpy as np
from collections import namedtuple
import json
import argparse

# WorldObject = namedtuple("Object", ["id", "class", "instance", "location", "state"], defaults=[None] * 3)
# GoalCondition = namedtuple("Goal", ["location", "state"], defaults=[None] * 2)

DISTRACTOR_OBJECTS = ["medicine", "bottle", "edible", "bowl", "mug", "pan"]
GOAL_OBJECTS = ["bowl", "mug", "pan"]
LOCATIONS = ["sink_counter_left", "sink_counter_right", "shelf_lower", "sink_bottom", "cabinettop_storage"]


def generate_semantic_specification(seed=0, max_num_goal_objs=1, max_num_distractor_objs=0, exclude_initial_loc=False):

    np.random.seed(seed)

    # example
    # objs_dict = {0: {"class": "pan", "instance": None, "location": "shelf_lower"},
    #                 1: {"class": "bottle", "instance": None, "location": "sink_bottom"}}
    # goals_dict = {0: {"goal_location": "cabinettop_storage"}}

    objs_dict = {}
    goals_dict = {}

    ########################
    # sample objects
    assert max_num_goal_objs >= 1
    num_goal_objs = np.random.randint(1, max_num_goal_objs + 1)
    if max_num_distractor_objs > 1:
        num_distractor_objs = np.random.randint(1, max_num_distractor_objs + 1)
    else:
        num_distractor_objs = max_num_distractor_objs

    goal_objs = []
    for oi in range(num_goal_objs):
        obj_cls = np.random.choice(GOAL_OBJECTS)
        obj_ins = None
        obj_loc = np.random.choice(LOCATIONS)
        obj_id = len(objs_dict)
        objs_dict[obj_id] = {"class": obj_cls, "instance": obj_ins, "location": obj_loc}
        goal_objs.append(obj_id)
    # TODO: if object in cabinet, initial state is clean

    for oi in range(num_distractor_objs):
        obj_cls = np.random.choice(DISTRACTOR_OBJECTS)
        obj_ins = None
        obj_loc = np.random.choice(LOCATIONS)
        objs_dict[len(objs_dict)] = {"class": obj_cls, "instance": obj_ins, "location": obj_loc}

    ########################
    # sample goal conditions
    for obj_id in goal_objs:
        if exclude_initial_loc:
            remaining_locations = copy.deepcopy(LOCATIONS)
            # remove initial location
            remaining_locations.remove(objs_dict[obj_id]["location"])
        else:
            remaining_locations = LOCATIONS

        goal_loc = np.random.choice(remaining_locations)

        if goal_loc == "cabinettop_storage":
            goals_dict[obj_id] = {"location": goal_loc, "state": None}
        else:
            # has the option to add clean as goal state
            goal_state = np.random.choice(["clean", None])
            goals_dict[obj_id] = {"location": goal_loc, "state": goal_state}

    dict = {"objects": objs_dict, "goals": goals_dict}
    return dict


def save_dict_to_json(dict, json_filename):
    with open(json_filename, "w") as fh:
        json.dump(dict, fh, sort_keys=True, indent=4)


def load_dict_from_json(json_filename):
    with open(json_filename, "r") as fh:
        dict = json.load(fh)
    return dict


def to_json_str(dict):
    return json.dumps(dict, sort_keys=True)


def generate_semantic_specs(save_dir, max_seed, max_num_goal_objs, max_num_distractor_objs, exclude_initial_loc):

    params = locals()
    save_dict_to_json(params, os.path.join(save_dir, "params.json"))

    unique_dict_str_to_seed = {}
    for si in range(max_seed):
        dict = generate_semantic_specification(seed=si,
                                               max_num_goal_objs=max_num_goal_objs,
                                               max_num_distractor_objs=max_num_distractor_objs,
                                               exclude_initial_loc=exclude_initial_loc)
        json_str = to_json_str(dict)
        if json_str not in unique_dict_str_to_seed:
            unique_dict_str_to_seed[json_str] = si

    print(f"generate {len(unique_dict_str_to_seed)} unique out of {max_seed} seeds")
    # for json_str in unique_dict_str_to_seed:
    #     print(unique_dict_str_to_seed[json_str], json_str)



if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--max_seed', type=int, default=10)
    # parser.add_argument('--max_num_goal_objs', type=int, default=1)
    # parser.add_argument('--max_num_distractor_objs', type=int, default=3)
    # parser.add_argument('--exclude_initial_loc', type=int, default=0)
    # args = parser.parse_args()

    save_dir = "/home/weiyu/Research/nsplan/original/kitchen-worlds/outputs/0418"
    generate_semantic_specs(save_dir=save_dir, max_seed=10000, max_num_goal_objs=1, max_num_distractor_objs=3, exclude_initial_loc=False)





