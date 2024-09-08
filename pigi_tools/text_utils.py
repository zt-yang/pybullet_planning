import copy

ACTION_NAMES = {
    "move_base": "move the robot",
    "move_base_wconf": "move the robot",
    "pick": "pick the object",
    "pick_from_supporter": "pick the object",
    "place": "place the object",
    "place_to_supporter": "place the object",
    "arrange": "place the object on somewhere",

    "grasp_handle": "grasp the handle of door joint",
    "pull_handle": "pull the handle of door joint",
    "pull_handle_with_link": "pull the handle of drawer joint",
    "pull_door_handle": "pull the handle of door joint",
    "ungrasp_handle": "release the grasp of the handle of door joint",
    "grasp_pull_ungrasp_handle": "grasp and pull the handle of door joint",
    "grasp_pull_ungrasp_handle_with_link": "grasp and pull the handle of drawer joint",

    "declare_store_in_space": "declare that all objects of this type have been stored into that space",
    "nudge_door": "push the door open more",
    "sprinkle": "sprinkle some condiments into a region",
}

ACTION_ABV = {
    "move_base": "",
    "move_base_wconf": "",
    "pick": "k",
    "place": "c",
    "arrange": "c",

    "grasp_handle": "",
    "pull_handle": "l",
    "pull_handle_with_link": "l",
    "pull_door_handle": "l",
    "ungrasp_handle": "",
    "grasp_pull_ungrasp_handle": "l",
    "grasp_pull_ungrasp_handle_with_link": "l",

    "declare_store_in_space": "",
    "nudge_door": "n",
    "sprinkle": "s",
}

TRANSLATIONS = {
    "aconf": "this is an arm configuration",
    "arm": "this is an arm",
    "ataconf": "robot puts its arm at that configuration",
    "atpose": "this object locates at that pose",
    "atposition": "this joint is set at that positions",
    "containable": "this object can contain that object",
    "contained": "this object is containing that object at that pose",
    "controllable": "this robot arm can be controlled",
    "door": "this is a door",
    "floor": "this is the floor",
    "food": "this is a kind of food",
    "graspable": "this object can be grasped",
    "handempty": "this robot is not holding anything",
    "holding": "this robot is holding that object",
    "in": "this object is inside that object",
    "isjointto": "this is a joint of that object",
    "islinkto": "this is a link of that object",
    "joint": "this is a joint",
    "object": "this is an object",
    "on": "this object is on that object",
    "pose": "this is a pose",
    "position": "this is a position",
    "space": "this is a space",
    "stackable": "this object can allow that object to stack on it",
    "supporter": "this object can support other objects",
    "supported": "this object is supporting that object at that pose",
    "wconf": "this object is a world configuration",
    "inwconf": "this object is in that world configuration",

    "meatturkeyleg": "this is a turkey leg",
    "veggieartichoke": "this is an artichoke",
    "veggiecabbage": "this is a cabbage",
    "veggiecauliflower": "this is a cauliflower",
    "veggiegreenpepper": "this is a green pepper",
    "veggiepotato": "this is a potato",
    "veggiesweetpotato": "this is a sweetpotato",
    "veggietomato": "this is a tomato",
    "veggiezucchini": "this is a zucchini",
    "stapler": "this is a stapler",

    "reachable": "this object is reachable",
    "unreachable": "this object is occluded",
    "aabby": "the object has a width of that value",
    "distancey": "the objects have a distance of that value",

    "storedinspace": "all objects of this type are stored in that space",
    "oftype": "the object is of that type",
    "supporter": "this is a supporter",
    "@edible": "food",
    "@bottle": "bottle",
    "@medicine": "medicine",

}
ALL_RELATIONS = copy.deepcopy(TRANSLATIONS)
ALL_RELATIONS.update({k: k for k in ACTION_NAMES.values()})

joint_groups = ['base', 'base-torso', 'left', 'right']
TRANSLATIONS.update({k: k for k in joint_groups})

DISCARDED = [
    "defaultconf", "isclosedposition",
    "contained x 1", "contained x 2", "isjointgroupof",
    "floor", "food", "stapler", "object", "door", "surface", "edible", ## "supporter",
    "pose", "position", "aconf", "ataconf", "wconf", "inwconf",
]
OMITTED_OBJS = ['scene', 'floor', 'pr2', 'floor1', 'pr20']
ALL_RELATIONS = {k: v for k, v in ALL_RELATIONS.items() if k not in DISCARDED \
                 and not k.startswith('veggie') and not k.startswith('meat')}
