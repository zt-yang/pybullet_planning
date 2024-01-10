from pybullet_tools.utils import wait_unlocked


def initialize_object_reducer(name):
    if name == 'goal-related':
        return reduce_facts_from_planning

    def return_as_is(facts, goals):
        return facts
    return return_as_is


def reduce_facts_from_planning(facts, goals):
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