from os.path import join, abspath, dirname


borderline = '  ;;' + '-' * 70 + '\n'
comment = '\n' + borderline + """  ;;      extended {key} from {extension}.pddl\n""" + borderline + '\n'
EXTENSIONS_DIR = join(dirname(abspath(__file__)), 'extensions')

STREAM_KEY = '(:stream '
OPERATOR_KEY = '(:action '
AXIOM_KEY = '(:derived '

empty_stream_body = '(define (stream symbolic)\n )'


def get_full_pddl_path(name):
    return abspath(join(dirname(__file__), f"{name}.pddl"))


## ---------------------------------------------------------------------------------------------


def get_uncommented_splits_by_key(pddl_content, key):
    """ avoids commented """
    all_lines = pddl_content.split(key)
    new_lines = []
    for i in range(len(all_lines)):
        if i > 0 and all_lines[i - 1].endswith(';'):
            continue
        line = all_lines[i]
        if line.endswith(';'):
            line = line[:-1]
        new_lines.append(line)
    return new_lines


def remove_all_streams_except_name(stream_pddl, stream_name):
    key = STREAM_KEY
    lines = get_uncommented_splits_by_key(stream_pddl, key)
    text = key.join([lines[0]]+[l for l in lines if l.startswith(f'{stream_name}\n')])
    return text + '\n)'


def remove_stream_by_name(stream_pddl, stream_name):
    key = STREAM_KEY
    lines = get_uncommented_splits_by_key(stream_pddl, key)
    return key.join([l for l in lines if not l.startswith(f'{stream_name}\n')])


def remove_operator_by_name(domain_pddl, operator_name):
    key = OPERATOR_KEY
    lines = get_uncommented_splits_by_key(domain_pddl, key)
    new_lines = []
    for i, line in enumerate(lines):
        if line.startswith(f'{operator_name}\n'):
            if line.endswith(';') or line.endswith('; '):
                new_lines[i-1] += ';'
        else:
            new_lines.append(line)
    return key.join(new_lines)


def remove_predicate_by_name(domain_pddl, predicate_name):
    for key in [f"(not ({predicate_name}))", f"({predicate_name})"]:
        domain_pddl = domain_pddl.replace(key, '')
    return domain_pddl


## ---------------------------------------------------------------------------------------------


def remove_the_last_end_bracket(lines):
    for i in range(len(lines)):
        index = len(lines) - i - 1
        if lines[index].strip() == ')':
            lines = lines[:index]
            break
    return lines


def _get_operator_axiom_blocks(lines):
    key1 = OPERATOR_KEY
    key2 = AXIOM_KEY
    lines = ''.join(lines).replace(key2, key1 + key2)
    blocks = get_uncommented_splits_by_key(lines, key1)
    return blocks


def clean_operator_lines(lines, return_blocks=False):
    key1 = OPERATOR_KEY
    key2 = AXIOM_KEY
    lines = ''.join(lines).replace(key2, key1 + key2)
    blocks = get_uncommented_splits_by_key(lines, key1)
    new_blocks = []
    for block in blocks:
        old_lines = block.split('\n')
        new_lines = []
        for line in old_lines:
            if ';' in line:
                line = line[:line.index(';')]
                if len(line.strip()) == 0:
                    continue
            new_lines.append(line)
        if len(new_lines) > 2 and ')' in new_lines[-2]:
            new_lines.append(' ')
        new_blocks.append('\n'.join(new_lines))
    if return_blocks:
        return [key1+s if not s.startswith(key2) else s for s in blocks[1:]]
    lines = key1.join(new_blocks).replace(key1 + key2, key2)
    lines = [l+'\n' for l in lines.split('\n')]
    return lines


def parse_domain(domain_lines):
    header = []
    predicates = []
    functions = None

    lines = []
    for line in domain_lines:
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
    operators_axioms = clean_operator_lines(operators_axioms)
    return header, predicates, functions, operators_axioms


def parse_domain_pddl(domain_name):
    return parse_domain(open(get_full_pddl_path(domain_name), 'r').readlines())


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


## ----------------------------------------------------------------------------------------------


def _create_domain_from_parts(header, predicates, functions, operators_axioms):
    return header + ['\n  (:predicates\n'] + predicates + \
             ['\n  )\n\n  (:functions\n'] + functions + operators_axioms


def create_domain(base_domain, extensions, output_domain):
    header, new_predicates, functions, new_operators_axioms = parse_domain_pddl(base_domain)
    for extension in extensions:
        _, predicates, _, operators_axioms = parse_domain_pddl(join(EXTENSIONS_DIR, extension))
        new_predicates.extend([comment.format(key='predicates', extension=extension)] + predicates)
        new_operators_axioms.extend([comment.format(key='operators & axioms', extension=extension)] + operators_axioms)
    blocks = _create_domain_from_parts(header, new_predicates, functions, new_operators_axioms)
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
    create_domain_and_stream('mobile', ['pull_decomposed', 'cooking'], 'mobile_v2')


def update_kitchen_nudge_pddl():
    """
    nudge_v1 (working): use plan-base-nudge-door to do motion planning
    nudge_v1b (best):   use plan-base-motion to do motion planning considering all world geometries
    nudge_v2 (bad):     use fluents when generating bconf to nudge, based on v1
    nudge_v3 (testing): open then nudge, based on v1b
    """
    extensions = ['pull_decomposed', 'cooking', 'nudge_v1b']
    create_domain_and_stream('mobile', extensions, 'mobile_v3')


def update_kitchen_action_pddl():
    extensions = ['pull_decomposed', 'cooking', 'nudge_v1b', 'arrange']
    create_domain_and_stream('mobile', extensions, 'mobile_v4')


def update_kitchen_pull_pddl():
    extensions = ['cooking', 'arrange']
    # extensions += ['pull_decomposed']
    extensions += ['pull']
    create_domain_and_stream('mobile', extensions, 'mobile_v5')


## ----------------------------------------------------------------------------------------------


def get_predicate_name_from_literal(l):
    """ (Name ?x ?y) -> Name """
    return l.strip().replace('(', '').replace(')', '').split(' ')[0]  ## .replace('\n', '')


def _get_predicate_arity(predicate_lines):
    lines = [l for l in predicate_lines if '(' in l]
    arity = {get_predicate_name_from_literal(l): len(l) - len(l.replace('?', '')) for l in lines}
    return arity


def _remove_predicates_in_operators_axioms(operators_axioms, predicates, include_axioms=False):
    blocks = clean_operator_lines(operators_axioms, return_blocks=True)
    axiom_blocks = [b for b in blocks if b.startswith(AXIOM_KEY)]
    operator_blocks = [b for b in blocks if b.startswith(OPERATOR_KEY)]

    new_operator_blocks = []
    for block in operator_blocks:
        # if ':parameters ()' in block:
        #     continue
        for literal in _get_literals_to_remove_in_operator(block, predicates) + ['(not )', '(not (=))']:
            block = block.replace(literal, '')
        block = block.replace('(and  ', '(and ')
        block = '\n'.join([l for l in block.split('\n') if len(l.strip()) > 0] + ['', ''])
        new_operator_blocks.append(block)

    new_operators = ''.join(new_operator_blocks)  ##  + axiom_blocks
    if include_axioms:
        new_operators = ''.join(new_operator_blocks+ axiom_blocks)
    new_operators = [l+'\n' for l in new_operators.split('\n')]

    ## -------------------------------------------------------------------------
    ## TODO: extract mapping from axiom and stream
    axioms_mapping = {}
    # for block in axiom_blocks:
    #     goal = block.split('\n')[0].replace(AXIOM_KEY, '')
    #     for literal in _get_literals_to_remove_in_axiom(block, predicates) + ['(not )', '(not (=))']:
    #         block = block.replace(literal, '')
    #     print('-'*40)
    #     print(block)
    #     print('-'*40)

    return new_operators, axioms_mapping


def _get_literals_to_remove_in_axiom(old_block, predicates_to_remove):
    ## TODO: extract mapping from axiom and stream
    conditions = old_block.split('(exists ')[1:]
    return old_block


def _get_literals_to_remove_in_operator(old_block, predicates_to_remove):
    block = old_block.split(':precondition ')[1]
    for k in ['not', 'and', 'or']:
        block = block.replace(f'({k} ', '')
    for k in [') )', '))']:
        block = block.replace(k, ')')
    for k in [':effect ', '(increase (total-cost) 1)', '\n']:
        block = block.replace(k, '')
    literals = [l.strip() + ')' for l in block.split(')') if '(' in l]
    literals_to_remove = [l for l in literals if get_predicate_name_from_literal(l) in predicates_to_remove]
    return literals_to_remove


def _remove_continuous_vars(body):
    body = body.replace(')', ' )')
    for v in ['q', 'bq', 'aq', 'hg', 'g', 'p', 'lp', 'rp', 'pstn', 't', 'at', 'bt']:
        for e in ['', '1', '2', '3']:
            body = body.replace(f'?{v}{e} ', '')
    return body.replace(' )', ')')


def _make_symbolic_domain(body, header, functions, predicates_arity, **kwargs):
    ## find all predicates that contain continuous variables
    body = [l + '\n' for l in body.split('\n')]
    _, new_predicates, _, new_operators_axioms = parse_domain(body)
    new_predicates_arity = _get_predicate_arity(new_predicates)
    predicates_to_remove = [k for k in predicates_arity if predicates_arity[k] != new_predicates_arity[k]]
    predicates_to_remove += ['Identical']
    # predicates_to_remove += ['CanMove']  ## because move_base is removed
    predicates_to_keep = [k for k in predicates_arity if k not in predicates_to_remove]

    new_operators, axioms_mapping = _remove_predicates_in_operators_axioms(new_operators_axioms, predicates_to_remove, **kwargs)

    new_predicates = [l for l in new_predicates if '(' not in l or \
                      get_predicate_name_from_literal(l) not in predicates_to_remove]
    functions = [l for l in functions if '?' not in l]
    symbolic_domain_pddl = _create_domain_from_parts(header, new_predicates, functions, new_operators)
    return symbolic_domain_pddl, predicates_to_keep, axioms_mapping


def make_symbolic_pddl(input_name, output_name):
    """ read file and generate file """
    header, predicates, functions, operators_axioms = parse_domain_pddl(f'{input_name}_domain')
    predicates_arity = _get_predicate_arity(predicates)

    for suffix in ['domain', 'stream']:

        ## first remove all continuous variables
        body = open(get_full_pddl_path(f'{input_name}_{suffix}'), 'r').read()
        body = _remove_continuous_vars(body)

        if suffix == 'domain':
            body, predicates_to_keep, _ = _make_symbolic_domain(body, header, functions, predicates_arity,
                                                                include_axioms=True)

        save_pddl_file(body, f'{output_name}_{suffix}')


def make_symbolic_pddl_inplace(domain_pddl):
    """ take in domain pddl string, output string """
    body = [l+'\n' for l in domain_pddl.split('\n')]
    header, predicates, functions, operators_axioms = parse_domain(body)
    predicates_arity = _get_predicate_arity(predicates)

    ## first remove all continuous variables
    body = domain_pddl
    body = _remove_continuous_vars(body)
    body, predicates_to_keep, _ = _make_symbolic_domain(body, header, functions, predicates_arity)
    if body[-1] == '\n':
        body.append(')')
    predicates_to_keep = [s.lower() for s in predicates_to_keep]
    return ''.join(body), predicates_to_keep

## ----------------------------------------------------------------------------------------------


def load_num_args_from_domain_pddl(body):
    body = [l + '\n' for l in body.split('\n')]
    _, predicates, _, operators_axioms = parse_domain(body)
    blocks = clean_operator_lines(operators_axioms, return_blocks=True)
    operator_blocks = [b.split('\n')[:2] for b in blocks if b.startswith(OPERATOR_KEY)]
    return {b[0].replace(OPERATOR_KEY, ''): len(b[1]) - len(b[1].replace('?', '')) for b in operator_blocks}


if __name__ == '__main__':
    # update_namo_pddl()
    # update_kitchen_pddl()
    # update_kitchen_action_pddl()
    # update_kitchen_pull_pddl()
    make_symbolic_pddl('mobile_v5', 'symbolic_mobile_v5')
