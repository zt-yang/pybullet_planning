def initialize_object_reducer(name):
    if name == 'goal-related':
        return reduce_facts_given_goals

    if name == 'object-related':
        return reduce_facts_given_objects

    def return_as_is(facts, inputs):
        return facts
    return return_as_is


def reduce_facts_given_goals(facts, goals):
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


def reduce_facts_given_objects(facts, objects):
    from pybullet_tools.logging import myprint
    myprint(f'\nfilter_init_by_objects({objects})')

    new_facts = []
    removed_facts = []
    for fact in facts:
        removed = False
        if fact[0] not in ['=']:
            for elem in fact[1:]:
                if (isinstance(elem, int) or isinstance(elem, tuple)) and elem not in objects:
                    removed = True
                    if 'AtRelPose' in fact[0] and ', 48),(0.141, -0.012, -0.033, 0.0, -0.0, 2.447))' in str(fact[2]):
                        print(fact)
                    myprint(f'\t removing fact {fact}')
                    break
        if removed:
            removed_facts.append(fact)
        else:
            new_facts.append(fact)
    return new_facts
