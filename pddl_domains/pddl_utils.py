from os.path import join, abspath, dirname


borderline = '  ;;' + '-' * 70 + '\n'
comment = '\n' + borderline + """  ;;      extended {key} from {extension}.pddl\n""" + borderline + '\n'
EXTENSIONS_DIR = join(dirname(abspath(__file__)), 'extensions')


def get_full_pddl_path(name):
    return abspath(join(dirname(__file__), f"{name}.pddl"))


def remove_the_last_end_bracket(lines):
    for i in range(len(lines)):
        index = len(lines) - i - 1
        if lines[index].strip() == ')':
            lines = lines[:index]
            break
    return lines


def parse_domain_pddl(domain_name):
    header = []
    predicates = []
    functions = None

    lines = []
    for line in open(get_full_pddl_path(domain_name), 'r').readlines():
        if '(:predicates' in line:
            header += lines
            lines = []

        elif '(:functions' in line:
            predicates += remove_the_last_end_bracket(lines)
            lines = []

        elif '(:action' in line and len(predicates) == 0:
            predicates += remove_the_last_end_bracket(lines)
            lines = [line]
            functions = []

        elif '(:action' in line and functions is None:
            functions = lines
            lines = [line]

        else:
            lines.append(line)

    operators_axioms = remove_the_last_end_bracket(lines)
    return header, predicates, functions, operators_axioms


def parse_stream_pddl(domain_name):
    header = []
    lines = []
    for line in open(get_full_pddl_path(domain_name), 'r').readlines():
        if '(:stream' in line and len(header) == 0:
            header += lines
            lines = [line]
        else:
            lines.append(line)
    if len(header) == 0:
        header += lines
        return header, []
    streams = remove_the_last_end_bracket(lines)
    return header, streams


def save_pddl_file(blocks, domain_name):
    text = ''.join(blocks) + '\n)'
    with open(get_full_pddl_path(domain_name), 'w') as f:
        f.write(text)
    # print(f'Saved {domain_name}')
    # print(text)


def create_domain(base_domain, extensions, output_domain):
    header, new_predicates, functions, new_operators_axioms = parse_domain_pddl(base_domain)
    for extension in extensions:
        _, predicates, _, operators_axioms = parse_domain_pddl(join(EXTENSIONS_DIR, extension))
        new_predicates.extend([comment.format(key='predicates', extension=extension)] + predicates)
        new_operators_axioms.extend([comment.format(key='operators & axioms', extension=extension)] + operators_axioms)
    blocks = header + ['\n  (:predicates\n'] + new_predicates + \
             ['\n  )\n\n  (:functions\n'] + functions + new_operators_axioms
    save_pddl_file(blocks, output_domain)


def create_stream(base_domain, extensions, output_domain):
    header, new_streams = parse_stream_pddl(base_domain)
    for extension in extensions:
        _, streams = parse_stream_pddl(join(EXTENSIONS_DIR, extension))
        new_streams.extend([comment.format(key='streams', extension=extension)] + streams)
    save_pddl_file(header + new_streams, output_domain)


def create_domain_and_stream(base_name, extensions, output_name):
    create_domain(base_domain=f'{base_name}_domain', extensions=[f'_{name}_domain' for name in extensions],
                  output_domain=f'{output_name}_domain')
    create_stream(base_domain=f'{base_name}_stream', extensions=[f'_{name}_stream' for name in extensions],
                  output_domain=f'{output_name}_stream')


def update_namo_pddl():
    create_domain_and_stream('mobile', ['namo'], 'mobile_namo')


def update_kitchen_pddl():
    create_domain_and_stream('mobile', ['cooking'], 'mobile_v2')


def update_kitchen_nudge_pddl():
    """
    nudge_v1 (working): use plan-base-nudge-door to do motion planning
    nudge_v1b (best):   use plan-base-motion to do motion planning considering all world geometries
    nudge_v2 (bad):     use fluents when generating bconf to nudge, based on v1
    nudge_v3 (testing): open then nudge, based on v1b
    """
    create_domain_and_stream('mobile', ['cooking', 'nudge_v1b'], 'mobile_v3')


def update_kitchen_action_pddl():
    create_domain_and_stream('mobile', ['cooking', 'nudge_v1b', 'arrange'], 'mobile_v4')


if __name__ == '__main__':
    # update_namo_pddl()
    # update_kitchen_pddl()
    update_kitchen_action_pddl()
