from os.path import join, abspath, dirname


borderline = '  ;;' + '-' * 70 + '\n'
comment = '\n' + borderline + """  ;;      extended {key} from {extension}.pddl\n""" + borderline + '\n'


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
    functions = []

    lines = []
    for line in open(get_full_pddl_path(domain_name), 'r').readlines():
        if '(:predicates' in line:
            header += lines
            lines = []
        elif '(:functions' in line:
            predicates += remove_the_last_end_bracket(lines) ## [l for l in lines if l.strip() != ')']
            lines = []
        elif '(:action' in line and len(functions) == 0:
            if len(predicates) == 0:
                predicates += lines
                lines = []
                continue
            functions += lines
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
    streams = remove_the_last_end_bracket(lines)
    return header, streams


def save_pddl_file(blocks, domain_name):
    text = ''.join(blocks) + '\n)'
    with open(get_full_pddl_path(domain_name), 'w') as f:
        f.write(text)
    print(f'Saved {domain_name}')
    print(text)


def create_domain(base_domain, extensions, output_domain):
    header, new_predicates, functions, new_operators_axioms = parse_domain_pddl(base_domain)
    for extension in extensions:
        _, predicates, _, operators_axioms = parse_domain_pddl(extension)
        new_predicates.extend([comment.format(key='predicates', extension=extension)] + predicates)
        new_operators_axioms.extend([comment.format(key='operators & axioms', extension=extension)] + operators_axioms)
    blocks = header + ['\n  (:predicates\n'] + new_predicates + \
             ['\n  (:functions\n'] + functions + new_operators_axioms
    save_pddl_file(blocks, output_domain)


def create_stream(base_domain, extensions, output_domain):
    header, new_streams = parse_stream_pddl(base_domain)
    for extension in extensions:
        _, streams = parse_stream_pddl(extension)
        new_streams.extend([comment.format(key='streams', extension=extension)] + streams)
    save_pddl_file(header + new_streams, output_domain)


if __name__ == '__main__':
    create_domain(base_domain='mobile_domain', extensions=['_namo_domain'], output_domain='mobile_namo_domain')
    create_stream(base_domain='mobile_stream', extensions=['_namo_stream'], output_domain='mobile_namo_stream')
