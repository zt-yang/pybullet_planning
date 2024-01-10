

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
        if isinstance(domain, Domain):
            domain = modify_pddlstream_domain(domain, self.predicates)

        if isinstance(domain, PDDLDomainParser):
            domain = modify_pddlgym_domain(domain, self.predicates)
        print()
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
            print('   modify_pddlgym_domain |', op.name, removed)
            op.name = new_name
            for a in domain.actions:
                if a.name == name:
                    print('   modify_pddlgym_domain |   renamed ', a.name, new_name)
                    domain.predicates.pop(a.name)
                    a.name = new_name
                    domain.predicates[new_name] = a
            operators[new_name] = op
        else:
            operators[name] = op
    domain.operators = operators
    return domain


def modify_pddlstream_domain(domain, predicates):
    """ add to path/source root: pddlstream.downward.src.translate """
    import pddl
    for op in domain.actions:
        removed = []
        new_name = op.name
        if op.name.startswith('move'):
            continue

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

        if op.name != new_name:
            print('   modify_pddlstream_domain | ', op.name, removed)
            op.name = new_name
    return domain