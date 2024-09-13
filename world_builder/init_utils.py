import random

from pybullet_tools.general_streams import Position
from pybullet_tools.bullet_utils import is_joint_open
from pybullet_tools.logging_utils import print_debug


def add_joint_status_facts(body, position=None, categories=None, verbose=False, return_description=False):
    init = []
    title = f'get_world_fluents | joint {body} is'

    if is_joint_open(body, threshold=1, is_closed=True):
        init += [('IsClosedPosition', body, position)]
        description = 'fully closed'
        if categories is not None and 'knob' in categories:
            description = 'turned off'

    elif not is_joint_open(body, threshold=0.25):
        init += [('IsClosedPosition', body, position)]
        description = 'slightly open'

    elif is_joint_open(body, threshold=1):
        init += [('IsOpenedPosition', body, position)]
        description = 'fully open'
        if categories is not None and 'knob' in categories:
            description = 'turned on'

    else:
        init += [('IsOpenedPosition', body, position)]
        description = 'partially open'

    if verbose:
        print(title, description)

    if return_description:
        return description

    return init


def check_subgoal_achieved(facts, goal, world):
    result = False
    if goal[0] in ['on', 'in', 'stacked'] and len(goal) == 3:
        body, supporter = goal[1], goal[2]
        atrelpose = [f[-1] for f in facts if f[0].lower() in ['atrelpose'] and f[1] == body and f[-1] == supporter]
        if len(atrelpose) > 0:
            result = True

        atpose = [f[-1] for f in facts if f[0].lower() in ['atpose'] and f[1] == body]
        found = [f for f in facts if f[0].lower() in ['supported', 'contained'] and \
                 f[1] == body and f[2] in atpose and f[3] == supporter]
        if len(found) > 0:
            result = True

    if goal[0] in ['openedjoint', 'closedjoint', 'close', 'open'] and len(goal) == 2 and isinstance(goal[1], tuple):
        joint = goal[1]
        min_position = Position(joint, 'min').value
        atposition = [f[-1] for f in facts if f[0].lower() in ['atposition'] and f[1] == joint]
        if len(atposition):
            if goal[0] in ['openedjoint', 'open'] and atposition[0].value != min_position:
                result = True
            if goal[0] in ['closedjoint', 'close'] and atposition[0].value == min_position:
                result = True

    if goal[0] in ['holding']:
        found = [f for f in facts if f[0].startswith('at') and f[0].endswith('grasp') and f[2] == goal[-1]]
        if len(found) > 0:
            result = True

    print('[world_utils.check_goal_achieved]\t', goal, result)
    return result


def get_potential_placements(goals, init):
    def get_body(body):
        from world_builder.entities import Object
        if isinstance(body, Object):
            return body.pybullet_name
        return body

    placements = {}
    for goal in goals:
        pred = goal[0].lower()
        if pred in ['on', 'in']:
            placements[get_body(goal[1])] = get_body(goal[2])
        elif pred in ['storedinspace']:
            for f in init:
                if f[0].lower() == 'oftype' and f[2] == goal[1]:
                    placements[get_body(f[1])] = get_body(goal[2])
    print('\ninit_utils.get_potential_placements: ', placements, '\n')
    return placements


def get_objects_at_grasp(init, goals, world):
    """ too many object and surface makes planning slow """
    title = '\t[init_utils.get_objects_at_grasp]'
    objs_at_grasp = [(f[1], f[2]) for f in init if f[0].lower() == 'atgrasp']
    if len(objs_at_grasp) > 0:
        ## all arms occupied
        if not world.robot.dual_arm or len(objs_at_grasp) == 2:

            if len(objs_at_grasp) == 2:
                ## if goal mentioned any arms or objects at hand, don't randomly sample an arm
                mentioned_arms = [g[1] for g in goals if len(g) > 1 and g[1] in world.robot.arms]
                mentioned_objs = [g[1] for g in goals if len(g) > 1 and g[1] in [ao[1] for ao in objs_at_grasp]]
                mentioned_objs += [g[2] for g in goals if len(g) > 2 and g[2] in [ao[1] for ao in objs_at_grasp]]
                arm_obj_to_drop = None

                ## if one arm is grasping goal mentioned object, keep that arm and object
                if len(mentioned_objs) == 1:
                    arm_obj_to_drop = [ao for ao in objs_at_grasp if ao[1] not in mentioned_objs][0]

                ## if both arms are empty, see if the arm is used
                elif len(mentioned_objs) == 0:
                    if len(mentioned_arms) > 0:
                        arm_obj_to_drop = [ao for ao in objs_at_grasp if ao[0] not in mentioned_arms][0]
                        # objs_at_grasp = [ao for ao in objs_at_grasp if ao[0] in mentioned_arms]
                    else:
                        random.shuffle(objs_at_grasp)
                        arm_obj_to_drop = objs_at_grasp[1]

                if arm_obj_to_drop is not None:
                    obj_to_drop = arm_obj_to_drop[1]
                    objs_at_grasp = objs_at_grasp[:1]
                    print(f'{title} both arm occupied, just randomly choose one {objs_at_grasp}')

                    found_mentioned = [f for f in init if _related_to_grasping_o(f, obj_to_drop)]
                    print(f'{title} since dropping {arm_obj_to_drop}, removing from init {found_mentioned}')
                    [init.remove(f) for f in found_mentioned]

            objs_at_grasp = [ao[1] for ao in objs_at_grasp]
            from leap_tools.heuristic_utils import add_surfaces_given_obstacles
            add_surfaces = add_surfaces_given_obstacles(world, objs_at_grasp, title=f'fix_init(objs_at_grasp={objs_at_grasp})\t')
            objs_at_grasp += add_surfaces
        else:
            objs_at_grasp = []
    return objs_at_grasp


def _related_to_grasping_o(f, o):
    return f[0] in ['graspable', 'stackable', 'containable'] and f[1] == o


def reduce_init_given_skeleton(init, skeleton):
    to_del = []

    objs_at_grasp = [f[2] for f in init if f[0].lower() == 'atgrasp']
    ## remove a hand to reduce planning time
    if len(objs_at_grasp) == 0 and skeleton is not None and \
            len(skeleton) == 3 and skeleton[1][0].startswith('pull_handle'):
        to_del.append(('canpull', 'right'))

    return to_del


def remove_unnecessary_movable_given_goal(init, goals, world):
    """ when the goal involves putting A into B, don't move B """
    goal = goals[0]
    if goal[0].lower() in ['on']:
        surface = goal[-1]
        if isinstance(surface, tuple):
            surface = surface[0]
        if 'movable' in world.body_to_object(surface).categories:
            found = [f for f in init if (f[0] == 'graspable' and f[1] == surface) \
                    or (f[0] == 'stackable' and f[1] == surface) ]
            print_debug(f'[init.remove_unnecessary_movable_given_goal]\tremoving {found} because goal is {goal}')
            init = [f for f in init if f not in found]
    return init
