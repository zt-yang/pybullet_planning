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

    def return_as_is(facts, objects=[], goals=[]):
        return facts
    return return_as_is


def reduce_facts_given_goals(facts, objects=[], goals=[]):
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
    print('\n\nreduce_facts_from_planning | Reduced %d facts' % len(reduced_facts),
          'Remaining %d facts\n ' % len(filtered_facts))
    print('\n '.join([str(f) for f in reduced_facts]))
    return filtered_facts


def reduce_facts_given_objects(facts, objects=[], goals=[]):
    from pybullet_tools.logging_utils import myprint
    myprint(f'\nfilter_init_by_objects | objects = {objects}')

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
    new_facts = []
    removed_facts = []
    for fact in facts:
        removed = False
        if fact[0] not in ['=']:
            for elem in fact[1:]:
                if (isinstance(elem, int) or isinstance(elem, tuple)) and elem not in objects:
                    removed = True
                    myprint(f'\t removing fact {fact}')
                    break
        if removed:
            removed_facts.append(fact)
        else:
            new_facts.append(fact)
    return new_facts


def reduce_by_object_heuristic_joints(facts, objects=[], goals=[]):
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
                print('add objects ny heuristic-joints', o)
                objects.append(o)

    return reduce_facts_given_objects(facts, objects=objects, goals=goals)


def reduce_by_object_all_joints(facts, objects=[], goals=[]):
    """ add all joints in the problem """
    for f in facts:
        for elem in f[1:]:
            if isinstance(elem, tuple) and len(elem) == 2 and elem not in objects:
                objects.append(elem)

    return reduce_facts_given_objects(facts, objects=objects, goals=goals)


def reduce_by_objects_heuristic_movables(facts, objects=[], goals=[], aabb_expansion=0.5):
    """ add movable objects that are close by in aabb, and also one other surface that's large enough """

    import numpy as np
    from pybullet_tools.utils import get_aabb, AABB, aabb_overlap

    def get_surface_aabb(region):
        return get_aabb(region[0], region[-1]) if isinstance(region, tuple) else get_aabb(region)

    if goals[0][0] in ['on', 'in']:
        region = goals[0][2]
        region_aabb = get_surface_aabb(region)

        all_movables = [f[1] for f in facts if f[0].lower() == 'graspable' and f[1] not in objects]
        other_surfaces = [f[1] for f in facts if f[0].lower() == 'surface' and f[1] != region]

        ## sort other_surfaces by aabb
        big_surface = sorted(other_surfaces, key=get_surface_aabb)[0]
        objects.append(big_surface)
        print('add objects by heuristic-movables | surface:\t', big_surface)

        for o in all_movables:
            mov_aabb = get_aabb(o)
            mov_aabb_expanded = AABB(lower=np.asarray(mov_aabb.lower) - aabb_expansion,
                                     upper=np.asarray(mov_aabb.upper) + aabb_expansion)
            if aabb_overlap(region_aabb, mov_aabb_expanded):
                print('add objects by heuristic-movables | movable:\t', o)
                objects.append(o)

    return reduce_facts_given_objects(facts, objects=objects, goals=goals)
