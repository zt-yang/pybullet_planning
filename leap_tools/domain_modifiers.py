from pybullet_tools.logging_utils import myprint as print

def initialize_domain_modifier(predicates):
    """ e.g., 'atseconf' """
    if predicates is None:
        return None

    if isinstance(predicates, str):
        if ',' in predicates:
            predicates = predicates.split(',')
        else:
            predicates = [predicates]

    return PostponePredicate(predicates)


class DomainModifier(object):

    def __init__(self):
        pass

    def __call__(self, domain, **kwargs):
        return self.modify(domain, **kwargs)

    def modify(self, domain, **kwargs):
        raise NotImplementedError('should implement this for DomainModifier')


class PostponePredicate(DomainModifier):

    def __init__(self, predicates, **kwargs):
        super(PostponePredicate, self).__init__()
        self.predicates = [l.lower() for l in predicates]
        # if 'atbconf' in self.predicates:
        #     self.predicates.append('canmove')

    def modify(self, domain, **kwargs):
        """ modifies both for the planner and preimage computation """
        from pddlstream.algorithms.downward import Domain
        from pddlgym.parser import PDDLDomainParser

        print()
        if isinstance(domain, tuple):
            domain, externals = domain
            domain = modify_pddlstream_domain(domain, self.predicates)
            externals = modify_pddlstream_stream(externals, self.predicates)
            return domain, externals

        elif isinstance(domain, PDDLDomainParser):
            domain = modify_pddlgym_domain(domain, self.predicates)
            return domain


def modify_operator_name(name, removed, orders):
    preds = list(set(removed))
    preds = [p for p in orders if p in preds]
    for p in preds:
        name += f'--no-{p}'
    return name


def modify_pddlgym_domain(domain, predicates):
    from pddlgym.structs import Literal, LiteralConjunction
    operators = {}
    for name, op in domain.operators.items():
        removed = []
        new_name = op.name
        if op.name.startswith('move'):
            operators[name] = op
            continue

        ## remove from preconditions
        literals = []
        for lit in op.preconds.literals:
            if isinstance(lit, Literal) and lit.predicate.name in predicates:
                removed.append(lit)
            else:  ## TODO: in ForALl
                literals.append(lit)
        if len(literals) != len(op.preconds.literals):
            rr = [r.predicate.name for r in removed]
            new_name = modify_operator_name(op.name, rr, predicates)
            op.preconds = LiteralConjunction(literals)

        ## remove from effects
        literals = []
        for lit in op.effects.literals:
            if isinstance(lit, Literal) and lit.predicate.name in predicates:
                removed.append(lit)
            else:  ## TODO: in ForALl
                literals.append(lit)
        if len(literals) != len(op.effects.literals):
            rr = [r.predicate.name for r in removed]
            new_name = modify_operator_name(op.name, rr, predicates)
            op.effects = LiteralConjunction(literals)

        if op.name != new_name:
            print(f'   modify_pddlgym_domain({op.name})\t{removed}')
            op.name = new_name
            for a in domain.actions:
                if a.name == name:
                    print(f'   modify_pddlgym_domain({a.name})\trenamed {new_name}')
                    domain.predicates.pop(a.name)
                    a.name = new_name
                    domain.predicates[new_name] = a
            operators[new_name] = op
        else:
            operators[name] = op
    domain.operators = operators
    return domain


remove_operators = {
    'basemotion': ['move_base']
}


def modify_pddlstream_domain(domain, predicates):
    """ add to path/source root: pddlstream.downward.src.translate """
    import pddl

    to_remove_operator_names = []
    for pred in predicates:
        if pred in remove_operators:
            to_remove_operator_names.extend(remove_operators[pred])

    to_remove_operators = []
    for op in domain.actions:
        removed = []
        new_name = op.name

        ## remove from preconditions
        parts = []
        for atom in op.precondition.parts:
            if atom.predicate not in predicates:
                parts.append(atom)
            else:
                removed.append(atom)
        if len(parts) != len(op.precondition.parts):
            rr = [r.predicate for r in removed]
            new_name = modify_operator_name(op.name, rr, predicates)
            op.precondition = pddl.Conjunction(parts)

        ## remove from effects
        effects = []
        for eff in op.effects:
            atom = eff.literal
            if atom.predicate not in predicates:
                effects.append(eff)
            else:
                removed.append(atom)
        if len(effects) != len(op.effects):
            rr = [r.predicate for r in removed]
            new_name = modify_operator_name(op.name, rr, predicates)
            op.effects = effects

        if op.name in to_remove_operator_names:
            to_remove_operators.append(op)
        if op.name != new_name:
            ## remove the whole action
            print(f'   modify_pddlstream_domain({op.name})\t{removed}')
            op.name = new_name

    for op in to_remove_operators:
        domain.actions.remove(op)

    return domain


def modify_pddlstream_stream(externals, predicates):
    to_remove = []
    for external in externals:
        found = [t[0] for t in external.certified if t[0] in predicates]
        if len(found) > 0:
            to_remove.append(external.name)
            print(f'   remove_pddlstream_stream({external.name})\t{found}')
    print()
    return [e for e in externals if e.name not in to_remove]
