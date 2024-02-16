def initialize_object_reducer(name):
    if name == 'goal-related':
        return reduce_facts_given_goals

    if name == 'object-related':
        return reduce_facts_given_objects

    if name == 'heuristic-joints':
        return reduce_by_object_heuristic_joints

    if name == 'all-joints':
        return reduce_by_object_all_joints

    def return_as_is(facts, objects, goals):
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
    from pybullet_tools.logging import myprint
    myprint(f'\nfilter_init_by_objects | objects = {objects}')

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
    all_joints = []
    for f in facts:
        for elem in f[1:]:
            if isinstance(elem, tuple) and len(elem) == 2:
                all_joints.append(elem)

    if goals[0][0] in ['on', 'in']:
        region = goals[0][2]
        for o in all_joints:
            if o[0] == region[0] and o not in objects:
                objects.append(o)

    return reduce_facts_given_objects(facts, objects=objects, goals=goals)


def reduce_by_object_all_joints(facts, objects=[], goals=[]):
    """ add all joints in the problem """
    for f in facts:
        for elem in f[1:]:
            if isinstance(elem, tuple) and len(elem) == 2 and elem not in objects:
                objects.append(elem)

    return reduce_facts_given_objects(facts, objects=objects, goals=goals)
