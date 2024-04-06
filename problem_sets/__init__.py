

def problem_fn_from_name(name):
    import problem_sets.pr2_problems as pr2_problems
    import problem_sets.problems_block_world as pr2_problems_new
    import problem_sets.problems_nvidia as problems_nvidia
    import problem_sets.pr2_problems_pigi as problems_pigi
    import problem_sets.feg_problems as feg_problems
    import problem_sets.problems_kitchen_mini as kitchen_mini
    funk_names = [pr2_problems, feg_problems, pr2_problems_new, problems_pigi, problems_nvidia, kitchen_mini]
    return find_problem_fn_from_name(name, funk_names)


def find_problem_fn_from_name(name, funk_names):
    from inspect import getmembers, isfunction
    for problem_bank in funk_names:
        result = [a[1] for a in getmembers(problem_bank) if isfunction(a[1]) and a[0] == name]
        if len(result) > 0:
            break
    if len(result) == 0:
        raise ValueError('Problem {} not found'.format(name))
    return result[0]
