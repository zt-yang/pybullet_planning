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


def generate_semantic_specification_with_color(seed=0, max_num_goal_objs=1, max_num_distractor_objs=0, exclude_initial_loc=False, force_distractor_objs_in_sink_num=None, force_goal_loc=None):

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

    if force_distractor_objs_in_sink_num is not None:
        assert force_distractor_objs_in_sink_num > 0, "force_distractor_objs_in_sink_num must be positive"
        num_objs_in_sink = np.random.randint(1, force_distractor_objs_in_sink_num + 1)
        if num_objs_in_sink > 0:
            sink_obj_idxs = np.random.permutation(num_distractor_objs)[:num_objs_in_sink]
        else:
            sink_obj_idxs = []
    else:
        sink_obj_idxs = []

    for oi in range(num_distractor_objs):

        if len(DISTRACTOR_OBJECTS) == 0:
            print(f"Fail to sample for seed {seed}")
            return None

        obj_cls = np.random.choice(DISTRACTOR_OBJECTS)
        obj_ins = None

        if len(sink_obj_idxs) > 0:
            if oi in sink_obj_idxs:
                obj_loc = "sink_bottom"
            else:
                location_copy = copy.deepcopy(LOCATIONS)
                location_copy.remove("sink_bottom")
                obj_loc = np.random.choice(location_copy)
        else:
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

        if force_goal_loc is not None:
            if force_goal_loc not in remaining_locations:
                print(f"Fail to sample for seed {seed} because force_goal_loc {force_goal_loc} is not in remaining_locations {remaining_locations}")
                return None
            goal_loc = force_goal_loc
        else:
            goal_loc = np.random.choice(remaining_locations)

        if goal_loc == "cabinettop_storage":
            goals_dict[obj_id] = {"location": goal_loc, "state": None}
        else:
            # has the option to add clean as goal state
            goal_state = np.random.choice(["clean", None])
            goals_dict[obj_id] = {"location": goal_loc, "state": goal_state}

    dict = {"objects": objs_dict, "goals": goals_dict}
    return dict


def generate_semantic_specification_with_color_withheld_red_bowl(seed=0, max_num_goal_objs=1, max_num_distractor_objs=0,
                                               exclude_initial_loc=False, force_distractor_objs_in_sink_num=None,
                                               force_goal_loc=None):
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

        obj_cls = "bowl"
        obj_ins = None
        obj_loc = np.random.choice(LOCATIONS)
        obj_id = len(objs_dict)

        if len(obj_class_to_colors[obj_cls]) == 0:
            return None
        else:
            obj_color = "red"
            obj_class_to_colors[obj_cls].remove(obj_color)

        # if object in cabinet, initial state is clean
        if obj_loc == "cabinettop_storage":
            obj_state = "clean"
        else:
            obj_state = np.random.choice(["clean", None])

        objs_dict[obj_id] = {"class": obj_cls, "instance": obj_ins, "location": obj_loc, "state": obj_state,
                             "color": obj_color}
        goal_objs.append(obj_id)

    if force_distractor_objs_in_sink_num is not None:
        assert force_distractor_objs_in_sink_num > 0, "force_distractor_objs_in_sink_num must be positive"
        num_objs_in_sink = np.random.randint(1, force_distractor_objs_in_sink_num + 1)
        if num_objs_in_sink > 0:
            sink_obj_idxs = np.random.permutation(num_distractor_objs)[:num_objs_in_sink]
        else:
            sink_obj_idxs = []
    else:
        sink_obj_idxs = []

    for oi in range(num_distractor_objs):

        if len(DISTRACTOR_OBJECTS) == 0:
            print(f"Fail to sample for seed {seed}")
            return None

        obj_cls = np.random.choice(DISTRACTOR_OBJECTS)
        obj_ins = None

        if len(sink_obj_idxs) > 0:
            if oi in sink_obj_idxs:
                obj_loc = "sink_bottom"
            else:
                location_copy = copy.deepcopy(LOCATIONS)
                location_copy.remove("sink_bottom")
                obj_loc = np.random.choice(location_copy)
        else:
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

        objs_dict[len(objs_dict)] = {"class": obj_cls, "instance": obj_ins, "location": obj_loc, "state": obj_state,
                                     "color": obj_color}

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

        if force_goal_loc is not None:
            if force_goal_loc not in remaining_locations:
                print(
                    f"Fail to sample for seed {seed} because force_goal_loc {force_goal_loc} is not in remaining_locations {remaining_locations}")
                return None
            goal_loc = force_goal_loc
        else:
            goal_loc = np.random.choice(remaining_locations)

        if goal_loc == "cabinettop_storage":
            goals_dict[obj_id] = {"location": goal_loc, "state": None}
        else:
            # has the option to add clean as goal state
            goal_state = np.random.choice(["clean", None])
            goals_dict[obj_id] = {"location": goal_loc, "state": goal_state}

    dict = {"objects": objs_dict, "goals": goals_dict}
    return dict


def generate_semantic_specification_with_color_hard_sink(seed=0, max_num_goal_objs=1, max_num_distractor_objs=0, exclude_initial_loc=False, force_distractor_objs_in_sink_num=None, force_goal_loc=None, force_distractor_obj_cls=None):

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

    assert max_num_distractor_objs >= force_distractor_objs_in_sink_num, "max_num_distractor_objs must be greater than or equal to force_distractor_objs_in_sink_num"

    if max_num_distractor_objs > 1:
        num_distractor_objs = max(np.random.randint(1, max_num_distractor_objs + 1), force_distractor_objs_in_sink_num)
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

    if force_distractor_objs_in_sink_num is not None:
        assert force_distractor_objs_in_sink_num > 0, "force_distractor_objs_in_sink_num must be positive"
        sink_obj_idxs = np.random.permutation(num_distractor_objs)[:force_distractor_objs_in_sink_num]
    else:
        sink_obj_idxs = []

    for oi in range(num_distractor_objs):

        if len(DISTRACTOR_OBJECTS) == 0:
            print(f"Fail to sample for seed {seed}")
            return None

        if force_distractor_obj_cls is not None:
            assert force_distractor_obj_cls in DISTRACTOR_OBJECTS, "force_distractor_obj_cls must be in DISTRACTOR_OBJECTS"
            obj_cls = force_distractor_obj_cls
        else:
            obj_cls = np.random.choice(DISTRACTOR_OBJECTS)
        obj_ins = None

        if len(sink_obj_idxs) > 0:
            if oi in sink_obj_idxs:
                obj_loc = "sink_bottom"
            else:
                location_copy = copy.deepcopy(LOCATIONS)
                location_copy.remove("sink_bottom")
                obj_loc = np.random.choice(location_copy)
        else:
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

        if force_goal_loc is not None:
            if force_goal_loc not in remaining_locations:
                print(f"Fail to sample for seed {seed} because force_goal_loc {force_goal_loc} is not in remaining_locations {remaining_locations}")
                return None
            goal_loc = force_goal_loc
        else:
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


def generate_semantic_specs(save_dir, max_seed, max_num_goal_objs, max_num_distractor_objs, exclude_initial_loc, force_distractor_objs_in_sink_num, force_goal_loc, force_distractor_obj_cls, hard_sink=False, red_bowl=False):
    """

    :param save_dir:
    :param max_seed:
    :param max_num_goal_objs:
    :param max_num_distractor_objs:
    :param exclude_initial_loc:
    :param force_distractor_objs_in_sink_num: if none, at least one and at most max_num_distractor_objs objects will be in the sink
    :return:
    """

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    params = locals()
    save_dict_to_json(params, os.path.join(save_dir, "params.json"))

    unique_dict_str_to_seed = {}
    for si in range(max_seed):
        if hard_sink:
            dict = generate_semantic_specification_with_color_hard_sink(seed=si,
                                                              max_num_goal_objs=max_num_goal_objs,
                                                              max_num_distractor_objs=max_num_distractor_objs,
                                                              exclude_initial_loc=exclude_initial_loc,
                                                              force_distractor_objs_in_sink_num=force_distractor_objs_in_sink_num,
                                                              force_goal_loc=force_goal_loc, force_distractor_obj_cls=force_distractor_obj_cls)
        elif red_bowl:
            dict = generate_semantic_specification_with_color_withheld_red_bowl(seed=si,
                                                              max_num_goal_objs=max_num_goal_objs,
                                                              max_num_distractor_objs=max_num_distractor_objs,
                                                              exclude_initial_loc=exclude_initial_loc,
                                                              force_distractor_objs_in_sink_num=force_distractor_objs_in_sink_num,
                                                              force_goal_loc=force_goal_loc)
        else:
            dict = generate_semantic_specification_with_color(seed=si,
                                                              max_num_goal_objs=max_num_goal_objs,
                                                              max_num_distractor_objs=max_num_distractor_objs,
                                                              exclude_initial_loc=exclude_initial_loc,
                                                              force_distractor_objs_in_sink_num=force_distractor_objs_in_sink_num,
                                                              force_goal_loc=force_goal_loc)

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


def get_semantic_spec(save_dir, id):
    return os.path.join(save_dir, "{}.json".format(id))


if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--max_seed', type=int, default=10)
    # parser.add_argument('--max_num_goal_objs', type=int, default=1)
    # parser.add_argument('--max_num_distractor_objs', type=int, default=3)
    # parser.add_argument('--exclude_initial_loc', type=int, default=0)
    # args = parser.parse_args()

    # save_dir = "/home/weiyu/data_drive/nsplan/0922/semantic_specs"
    # save_dir = "/svl/u/weiyul/data_drive/nsplan/0919_constrained_placing/semantic_specs"
    # generate_semantic_specs(save_dir=save_dir, max_seed=1000,
    #                         max_num_goal_objs=1, max_num_distractor_objs=3,
    #                         exclude_initial_loc=False, force_distractor_objs_in_sink_num=3, force_goal_loc="sink_bottom")

    # # generate test examples
    # save_dir = "/home/weiyu/data_drive/nsplan/0922_sink_v1/semantic_specs"
    # generate_semantic_specs(save_dir=save_dir, max_seed=1000,
    #                         max_num_goal_objs=1, max_num_distractor_objs=3,
    #                         exclude_initial_loc=False, force_distractor_objs_in_sink_num=1, force_goal_loc="sink_bottom",
    #                         force_distractor_obj_cls="pan",
    #                         hard_sink=True)

    # # generate test examples
    # save_dir = "/home/weiyu/data_drive/nsplan/0922_red_bowl/semantic_specs"
    # generate_semantic_specs(save_dir=save_dir, max_seed=1000,
    #                         max_num_goal_objs=1, max_num_distractor_objs=3,
    #                         exclude_initial_loc=False, force_distractor_objs_in_sink_num=3, force_goal_loc=None,
    #                         force_distractor_obj_cls=None,
    #                         hard_sink=False, red_bowl=True)

    # generate test examples
    save_dir = "/home/weiyu/data_drive/nsplan/0923_constrained_placing/semantic_specs"
    generate_semantic_specs(save_dir=save_dir, max_seed=1000,
                            max_num_goal_objs=1, max_num_distractor_objs=3,
                            exclude_initial_loc=False, force_distractor_objs_in_sink_num=1, force_goal_loc="sink_bottom",
                            force_distractor_obj_cls="pan",
                            hard_sink=True)

    # # generate test examples
    # save_dir = "/svl/u/weiyul/data_drive/nsplan/0923_constrained_placing/semantic_specs"
    # generate_semantic_specs(save_dir=save_dir, max_seed=1000,
    #                         max_num_goal_objs=1, max_num_distractor_objs=3,
    #                         exclude_initial_loc=False, force_distractor_objs_in_sink_num=1, force_goal_loc="sink_bottom",
    #                         force_distractor_obj_cls="pan",
    #                         hard_sink=True)
