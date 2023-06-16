import os
import copy
import numpy as np
from collections import namedtuple
import json
import argparse

# WorldObject = namedtuple("Object", ["id", "class", "instance", "location", "state"], defaults=[None] * 3)
# GoalCondition = namedtuple("Goal", ["location", "state"], defaults=[None] * 2)

DISTRACTOR_OBJECTS = ["medicine", "bottle", "food", "bowl", "mug", "pan"]
GOAL_OBJECTS = ["bowl", "mug", "pan"]
LOCATIONS = ["sink_counter_left", "sink_counter_right", "shelf_lower", "sink_bottom", "cabinettop_storage"]
COLORS = ["red", "green", "blue", "yellow", "grey", "brown"]

# defined in pybullet_tools/utils.py
# RED = RGBA(1, 0, 0, 1)
# GREEN = RGBA(0, 1, 0, 1)
# BLUE = RGBA(0, 0, 1, 1)
# BLACK = RGBA(0, 0, 0, 1)
# WHITE = RGBA(1, 1, 1, 1)
# BROWN = RGBA(0.396, 0.263, 0.129, 1)
# TAN = RGBA(0.824, 0.706, 0.549, 1)
# GREY = RGBA(0.5, 0.5, 0.5, 1)
# YELLOW = RGBA(1, 1, 0, 1)

def generate_semantic_specification(seed=0, max_num_goal_objs=1, max_num_distractor_objs=0, exclude_initial_loc=False):

    np.random.seed(seed)

    # example
    # objs_dict = {0: {"class": "pan", "instance": None, "location": "shelf_lower", "state": "clean"},
    #                 1: {"class": "bottle", "instance": None, "location": "sink_bottom", "state": None}}
    # goals_dict = {0: {"location": "cabinettop_storage", "state": "clean"}}

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

        # if object in cabinet, initial state is clean
        if obj_loc == "cabinettop_storage":
            obj_state = "clean"
        else:
            obj_state = np.random.choice(["clean", None])

        objs_dict[obj_id] = {"class": obj_cls, "instance": obj_ins, "location": obj_loc, "state": obj_state}
        goal_objs.append(obj_id)

    for oi in range(num_distractor_objs):
        obj_cls = np.random.choice(DISTRACTOR_OBJECTS)
        obj_ins = None
        obj_loc = np.random.choice(LOCATIONS)

        # if object in cabinet, initial state is clean
        if obj_loc == "cabinettop_storage":
            obj_state = "clean"
        else:
            obj_state = np.random.choice(["clean", None])

        objs_dict[len(objs_dict)] = {"class": obj_cls, "instance": obj_ins, "location": obj_loc, "state": obj_state}

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


def generate_semantic_specification_with_color(seed=0, max_num_goal_objs=1, max_num_distractor_objs=0, exclude_initial_loc=False):

    np.random.seed(seed)

    # example
    # objs_dict = {0: {"class": "pan", "instance": None, "location": "shelf_lower", "state": "clean"},
    #                 1: {"class": "bottle", "instance": None, "location": "sink_bottom", "state": None}}
    # goals_dict = {0: {"location": "cabinettop_storage", "state": "clean"}}

    objs_dict = {}
    goals_dict = {}

    ########################
    # sample objects
    assert max_num_goal_objs >= 1
    num_goal_objs = np.random.randint(1, max_num_goal_objs + 1)

    obj_class_to_colors = {}
    for obj_cls in set(DISTRACTOR_OBJECTS).union(set(GOAL_OBJECTS)):
        obj_class_to_colors[obj_cls] = copy.deepcopy(COLORS)

    if max_num_distractor_objs > 1:
        num_distractor_objs = np.random.randint(1, max_num_distractor_objs + 1)
    else:
        num_distractor_objs = max_num_distractor_objs

    goal_objs = []
    for oi in range(num_goal_objs):

        # if len(GOAL_OBJECTS) == 0:
        #     print(f"Fail to sample for seed {seed}")
        #     return None

        obj_cls = np.random.choice(GOAL_OBJECTS)
        obj_ins = None
        obj_loc = np.random.choice(LOCATIONS)
        obj_id = len(objs_dict)

        if len(obj_class_to_colors[obj_cls]) == 0:
            return None
        else:
            obj_color = np.random.choice(obj_class_to_colors[obj_cls])
            obj_class_to_colors[obj_cls].remove(obj_color)

        # if object in cabinet, initial state is clean
        if obj_loc == "cabinettop_storage":
            obj_state = "clean"
        else:
            obj_state = np.random.choice(["clean", None])

        objs_dict[obj_id] = {"class": obj_cls, "instance": obj_ins, "location": obj_loc, "state": obj_state, "color": obj_color}
        goal_objs.append(obj_id)


    for oi in range(num_distractor_objs):

        if len(DISTRACTOR_OBJECTS) == 0:
            print(f"Fail to sample for seed {seed}")
            return None

        obj_cls = np.random.choice(DISTRACTOR_OBJECTS)
        obj_ins = None
        obj_loc = np.random.choice(LOCATIONS)

        if len(obj_class_to_colors[obj_cls]) == 0:
            return None
        else:
            obj_color = np.random.choice(obj_class_to_colors[obj_cls])
            obj_class_to_colors[obj_cls].remove(obj_color)

        # if object in cabinet, initial state is clean
        if obj_loc == "cabinettop_storage":
            obj_state = "clean"
        else:
            obj_state = np.random.choice(["clean", None])

        objs_dict[len(objs_dict)] = {"class": obj_cls, "instance": obj_ins, "location": obj_loc, "state": obj_state, "color": obj_color}

    # ########################
    # # sample colors for objects, objects in the same class cannot have the same color
    # for obj_id in objs_dict:
    #

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

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    params = locals()
    save_dict_to_json(params, os.path.join(save_dir, "params.json"))

    unique_dict_str_to_seed = {}
    for si in range(max_seed):
        dict = generate_semantic_specification_with_color(seed=si,
                                               max_num_goal_objs=max_num_goal_objs,
                                               max_num_distractor_objs=max_num_distractor_objs,
                                               exclude_initial_loc=exclude_initial_loc)

        if dict is None:
            continue

        json_str = to_json_str(dict)
        if json_str not in unique_dict_str_to_seed:
            unique_dict_str_to_seed[json_str] = si

    print(f"generate {len(unique_dict_str_to_seed)} unique out of {max_seed} seeds")
    for json_str in unique_dict_str_to_seed:
        # print(unique_dict_str_to_seed[json_str], json_str)
        save_dict_to_json(json.loads(json_str), os.path.join(save_dir, f"{unique_dict_str_to_seed[json_str]}.json"))

def get_semantic_specs(save_dir):
    return [f for f in sorted(os.listdir(save_dir)) if f != "params.json"]


if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--max_seed', type=int, default=10)
    # parser.add_argument('--max_num_goal_objs', type=int, default=1)
    # parser.add_argument('--max_num_distractor_objs', type=int, default=3)
    # parser.add_argument('--exclude_initial_loc', type=int, default=0)
    # args = parser.parse_args()

    save_dir = "/home/weiyu/Research/nsplan/original/kitchen-worlds/outputs/0423/semantic_specs"
    generate_semantic_specs(save_dir=save_dir, max_seed=100, max_num_goal_objs=1, max_num_distractor_objs=3, exclude_initial_loc=False)





