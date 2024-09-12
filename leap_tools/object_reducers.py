import numpy as np

from pybullet_tools.utils import get_aabb, get_aabb_center, get_aabb_extent, AABB, aabb_overlap
from pybullet_tools.logging_utils import myprint as print, print_debug

from leap_tools.heuristic_utils import *


def initialize_object_reducer(name):
    if name == 'goal-related':
        return reduce_facts_given_goals

    if name == 'object-related':
        return reduce_facts_given_objects

    if name == 'heuristic-joints':
        return reduce_by_object_heuristic_joints

    if name == 'all-joints':
        return reduce_by_object_all_joints

    if name == 'heuristic-movables':
        return reduce_by_objects_heuristic_movables

    def return_as_is(facts, objects=[], goals=[], world=None):
        return facts
    return return_as_is


def reduce_facts_given_goals(facts, objects=[], goals=[], world=None):
    goal_objects = []
    for g in goals[:1]:
        for elem in g:
            if isinstance(elem, int) or isinstance(elem, tuple):
                goal_objects.append(str(elem))

    filtered_facts = []
    filter_predicates = ['graspable', 'surface', 'plate', 'atpose', 'pose', 'stackable']
    reduced_facts = []
    for f in facts:
        found = f[0] not in filter_predicates
        if not found:
            for elem in f:
                if str(elem) in goal_objects:
                    found = True
                    break
        if found:
            filtered_facts.append(f)
        else:
            reduced_facts.append(f)
    print(f'\n\nreduce_facts_given_goals | reduced {len(reduced_facts)} facts \tremaining {len(filtered_facts)} facts')
    print('\t'+'\n\t'.join([str(f) for f in reduced_facts]))
    return filtered_facts


def reduce_facts_given_objects(facts, objects=[], goals=[], world=None, verbose=False):
    """ the base function for all used by VLM-TAMP """
    arms = [f[1] for f in facts if f[0] == 'controllable']
    dynamic_preds = ['atposition', 'atpose', 'atrelpose', 'atgrasp']
    dynamic_preds_potential = [f[2:] for f in dynamic_preds]

    ## identify goal related objects
    objects_in_goals = []
    for goal in goals:
        for elem in goal[1:]:
            if isinstance(elem, int) and elem in objects:
                objects_in_goals.append(elem)

    ## find relevant supporting surfaces of goal objects
    surfaces = [f[-1] for f in facts if f[0].lower() in ['relpose', 'contained'] and f[1] in objects_in_goals]
    objects += [o for o in surfaces if o not in objects]

    ## retain only facts involving the objects
    if verbose:
        print(f'\nfilter_init_by_objects | objects = {objects}')
    new_facts = []
    removed_facts = []
    dynamic_variables = []
    dynamic_facts_potential = []
    for fact in facts:
        removed = False
        if fact[0] in ['ungraspbconf']:
            removed = True
        elif fact[0] in dynamic_preds:
            dynamic_variables.append(fact[-1])
        elif fact[0] in dynamic_preds_potential:
            removed = True
            dynamic_facts_potential.append(fact)
        elif fact[0] not in ['=']:
            found_objects = [elem for elem in fact[1:] if isinstance(elem, int) or isinstance(elem, tuple)]
            goal_mentioned_objects = [elem for elem in found_objects if elem in objects_in_goals]
            goal_supportive_objects = [elem for elem in found_objects if elem in objects and elem not in objects_in_goals]
            irrelevant_objects = [elem for elem in found_objects if elem not in objects]
            if len(goal_mentioned_objects) == 0 and len(irrelevant_objects) > 0:
                if verbose:
                    print(f'\t removing fact {fact} containing'
                          f'\t irrelevant_objects={irrelevant_objects}'
                          f'\t goal_supportive_objects={goal_supportive_objects}')
                removed = True
            # for elem in fact[1:]:
            #     ## involve objects not in planning object
            #     if (isinstance(elem, int) or isinstance(elem, tuple)) and elem not in objects:
            #         removed = True
            #         if verbose:
            #             print(f'\t removing fact {fact}')
            #         break
        if removed:
            removed_facts.append(fact)
        else:
            new_facts.append(fact)

    dynamic_facts = [f for f in dynamic_facts_potential if f not in new_facts and f[-1] in dynamic_variables]
    dynamic_facts = sorted(dynamic_facts, key=lambda f: f[0])
    new_facts += dynamic_facts
    if verbose:
        print('\nreduce_facts_given_objects\tfound dynamic_facts\n\t'+'\n\t'.join([str(f) for f in dynamic_facts])+'\n')

    ## if the goal is place on surface, remove the surface as graspable to prevent pddlstream blowing up
    new_facts = _remove_exploding_init_combo(new_facts, goals)
    return new_facts


def _remove_exploding_init_combo(facts, goals):
    for goal in goals:
        if goal[0] in ['in', 'on']:
            found_exploding = [f for f in facts if f[0] == 'graspable' and f[1] == goal[-1]]
            found_exploding += [f for f in facts if f[0] in ['stackable', 'containable'] and f[1] == goal[-1]]
            if found_exploding:
                print(f'[object_reducers.reduce_facts_given_objects] removed {found_exploding} because {goal}')
                for f in found_exploding:
                    facts.remove(f)
    return facts


def reduce_by_object_heuristic_joints(facts, objects=[], goals=[], world=None):
    """ add joints that are related to the surface / space mentioned in the goal """

    if goals[0][0] in ['on', 'in']:

        all_joints = []
        for f in facts:
            for elem in f[1:]:
                if isinstance(elem, tuple) and len(elem) == 2:
                    all_joints.append(elem)

        region = goals[0][2]
        for o in all_joints:
            if o[0] == region[0] and o not in objects:
                print(f'add objects ny heuristic-joints\t{o}')
                objects.append(o)

    return reduce_facts_given_objects(facts, objects=objects, goals=goals)


def reduce_by_object_all_joints(facts, objects=[], goals=[], world=None, verbose=False):
    """ add all joints in the problem that may affect a space """
    title = '[object_reducers.reduce_by_object_all_joints]\t'
    objects_with_regions = set([j[1][0] for j in facts if j[0] in ['surface', 'space'] and isinstance(j[1], tuple)])
    joints_to_add = []
    joints_to_skip = []
    for f in facts:
        for elem in f[1:]:
            if isinstance(elem, tuple) and len(elem) == 2 and elem not in objects + joints_to_add + joints_to_skip:
                if elem[0] not in objects_with_regions:
                    joints_to_skip.append(elem)
                    if verbose:
                        print_debug(f'{title} skipping {elem} because it is isolated')
                    continue
                joints_to_add.append(elem)
    if verbose:
        print_debug(f'{title} adding {joints_to_add}')

    return reduce_facts_given_objects(facts, objects=objects+joints_to_add, goals=goals)


def reduce_by_objects_heuristic_movables(facts, objects=[], goals=[], world=None, aabb_expansion=0.5):
    """ add movable objects that are close by in aabb, and also one other surface that's large enough """

    title = 'object_reducers.reduce_by_objects_heuristic_movables\t'

    region_aabb = None
    if goals[0][0] in ['on', 'in', 'sprinkledto']:
        region = goals[0][2]
        region_aabb = get_surface_aabb(region)

    elif goals[0][0] in ['openedjoint', 'closedjoint', 'nudgeddoor']:
        region = joint = goals[0][1]
        region_aabb = get_aabb(joint[0], link=joint[1])

    else:
        print(f'{title} goals {goals} not recognized for region, see object_reducers.py')

    if region_aabb is not None:
        other_surfaces = [f[1] for f in facts if f[0].lower() == 'surface' and f[1] != region]

        ## add big surfaces in the world as temporary surfaces
        big_surfaces = find_big_surfaces(other_surfaces, top_k=1, world=world, title=title)
        objects.extend(big_surfaces)

        ## find obstacles
        all_movables = [f[1] for f in facts if f[0].lower() == 'graspable' and f[1] not in objects]
        obstacles = find_movables_close_to_region(all_movables, region_aabb, title=title, aabb_expansion=aabb_expansion)
        objects.extend(obstacles)
        other_surfaces = [f for f in other_surfaces if f not in big_surfaces]

        ## add another surfaces closest to obstacles
        for o in obstacles:
            add_surfaces = find_surfaces_close_to_region(other_surfaces, region_aabb, top_k=2, title=title+f'surface close to {o}\t')
            objects.extend(add_surfaces)

    return reduce_facts_given_objects(facts, objects=objects, goals=goals)




