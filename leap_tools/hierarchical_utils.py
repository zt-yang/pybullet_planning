import sys
import os
from os.path import join, dirname, abspath
ROOT_DIR = abspath(join(dirname(__file__), os.pardir))
sys.path.extend([
    join(ROOT_DIR, '..'),
    join(ROOT_DIR, '..', 'pddlgym'),
    join(ROOT_DIR, '..', 'pddlstream'),
    join(ROOT_DIR, '..', 'bullet', 'pybullet-planning'),
    join(ROOT_DIR, '..', 'bullet', 'pybullet-planning', 'examples'),
    '/Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages',  ## to use imgeio-ffmpeg
    # join(ROOT_DIR, '..', 'pddlstream'),
    # join(ROOT_DIR, '..', 'pddlstream', 'examples', 'pybullet', 'utils'),
])
import shutil
import copy
import time
from pprint import pprint

from pddlgym.core import PDDLEnv, _check_domain_for_strips
from pddlgym.parser import PDDLDomainParser
from pddlgym.inference import find_satisfying_assignments
from pddlgym.structs import Literal, LiteralConjunction, \
    Predicate, Type, TypedEntity, Exists, ForAll, LiteralDisjunction, DerivedPredicate
from pddlgym.spaces import LiteralSpace, LiteralSetSpace

from world_builder.entities import Object

from pybullet_tools.logging_utils import summarize_facts
from pybullet_tools.pr2_primitives import Pose, Grasp, Conf, APPROACH_DISTANCE
from pybullet_tools.pr2_utils import TOP_HOLDING_LEFT_ARM
from pybullet_tools.utils import get_unit_vector, multiply, unit_quat
from pybullet_tools.logging_utils import myprint as print, summarize_state_changes, print_list

from pddlstream.language.constants import Action
from pddlstream.algorithms.algorithm import parse_problem

DEFAULT_TYPE = Type("default")
DOMAIN_DIR = join(dirname(os.getcwd()), '..', 'bullet', 'assets', 'pddl', 'domains')


def get_required_pred(domain):
    ## YANG's way to reduce inference problem_sets
    for pred in domain.predicates.values():
        if not pred.is_derived:
            continue

        ## -------------------------------------
        ## YANG: avoid the prolog run if
        required_preds = []

        # required_types = []

        def add(lit):
            required_preds.append(lit.predicate)
            # required_types.extend([v.var_type for v in lit.variables])

        def add_exist(body):
            if isinstance(body, Literal):
                add(body)
            elif isinstance(body, LiteralConjunction):
                for lit in body.literals:
                    add(lit)
            else:
                print('YANG7, what is this derived definition in Exist')

        def add_literal(body):
            if isinstance(body, Literal):
                add(body)
            elif isinstance(body, Exists) or isinstance(body, ForAll):
                add_exist(body.body)
            else:
                print('YANG8, what is this AND/OR inside AND/OR')

        if isinstance(pred.body, Literal) or isinstance(pred.body, Exists) or isinstance(pred.body, ForAll):
            add_literal(pred.body)
        elif isinstance(pred.body, LiteralConjunction) or isinstance(pred.body, LiteralDisjunction):
            for lit in pred.body.literals:
                add_literal(lit)
        else:
            print('YANG6, what is this derived definition')
        pred.required_preds = required_preds
        # pred.required_types = required_types
    # print(f'computed required preds in {round(time.time()-start, 3)} sec')
    return domain


def get_preconds(pred):
    literals = []
    if isinstance(pred, DerivedPredicate):  ## axioms
        if isinstance(pred.body, Exists):
            literals.extend(pred.body.body.literals)
        elif isinstance(pred.body, LiteralConjunction):
            literals.extend(pred.body.literals)
        if isinstance(pred.body, Literal):
            literals.append(pred.body)
    else:
        literals.extend(pred.preconds.literals)
    return literals


def get_effects(action):
    literals = []
    if isinstance(action.effects, Literal):
        literals.append(action.effects)
    elif isinstance(action.effects, LiteralConjunction):
        literals.extend(action.effects.literals)
    return literals


def get_dynamic_literals(domain):
    literals = []
    for operator in domain.operators.values():
        for eff in get_effects(operator):
            if eff.predicate.name not in literals:
                literals.append(eff.predicate.name)
    for name, pred in domain.predicates.items():
        if pred.is_derived:
            literals.append(name)
    return literals


def find_pred_in_state(state, name):
    literals = []
    for lit in list(state.literals):
        if lit.predicate.name == name:
            literals.append(lit)
    return literals


def find_poses(facts):
    poses = {}
    for literal in facts:
        if literal.predicate.name == 'atpose':
            body, pose = literal.variables
            poses[body] = pose
    return poses


def find_on_pose(on_facts, facts):
    on_map = {}
    for on in on_facts:
        for fa in facts:
            if on.variables[0] == fa.variables[0]:
                on_map[on] = fa
    return on_map


def empty_temp():
    tmp_dir = join('../../leap/temp')
    shutil.rmtree(tmp_dir)
    os.mkdir(tmp_dir)


def get_elem_in_tups(tups):
    elems = []
    for tup in tups:
        elems.extend([n for n in tup[1:] if n not in elems])
    elems.sort()
    return elems


def get_updated_facts(facts, info):
    print('hierarchical.get_updated_facts')

    from_bconf = [f for f in facts if f[0] == 'atbconf'][0]
    print(f'\tfrom_bconf {from_bconf}')

    new_facts = []
    add_effects = [str(f) for f in info['add_effects']]
    del_effects = [str(f) for f in info['del_effects']]
    for f in facts:
        if str(f) in del_effects:
            print(f'\tDeleted fact: {f}')
            del_effects.remove(str(f))
            continue

        new_facts.append(f)
        if str(f) in add_effects:
            print(f'\tCannot add fact because already in facts: {f}')
            add_effects.remove(str(f))

    for f in info['add_effects']:
        if str(f) in add_effects:
            new_facts.append(f)
            print(f'\tAdded fact: {f}')

    return new_facts


def fix_facts_due_to_loaded_agent_state(facts):
    """ when reloading agent state, the variable in atgrasp is not the same once loaded in pickle,
        so there may be (handempty left) and (atgrasp left), so manually remove the later """
    handempty = [f[1] for f in facts if f[0] == 'handempty']
    for arm in handempty:
        found_conflicting = [f for f in facts if f[0].startswith('at') and f[0].endswith('grasp') and f[1] == arm]
        if found_conflicting:
            for f in found_conflicting:
                print(f'[hierarchical_utils.fix_facts_due_to_loaded_agent_state] removed {f} because handempty')
                facts.remove(f)
    return facts


def remove_unparsable_lines(original_domain_path, temp_dir):
    with open(original_domain_path, 'r') as f:
        lines = f.read()
    new_domain_path = join(temp_dir, 'domain.pddl')
    lines = lines.replace('(increase (total-cost) 1)', '')
    with open(new_domain_path, 'w') as f:
        f.write(lines)
    return new_domain_path
