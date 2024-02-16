def initialize_object_reducer(name):
    if name == 'goal-related':
        return reduce_facts_given_goals

    if name == 'object-related':
        return reduce_facts_given_objects

    if name == 'object-heuristic':
        return reduce_facts_given_objects_by_heuristic

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


def reduce_facts_given_objects(facts, objects=[], goals=[], use_heuristic=False):
    from pybullet_tools.logging import myprint
    myprint(f'\nfilter_init_by_objects(use_heuristic={use_heuristic}) -> objects = {objects}')

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


def reduce_facts_given_objects_by_heuristic(facts, objects=[], goals=[]):
    """ add objects that are related to the goal """
    return reduce_facts_given_objects(facts, objects=objects, goals=goals, use_heuristic=True)
