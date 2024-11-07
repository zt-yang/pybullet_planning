import base64
import copy
import os
from os.path import join, dirname, isfile, abspath, pardir, isdir
from os import listdir

import random
import json
import time
import numpy as np

from examples.pybullet.utils.pybullet_tools.utils import get_link_pose
from pybullet_tools.bullet_utils import nice
from pybullet_tools.pose_utils import draw_pose3d_path
from pybullet_tools.utils import get_aabb, get_joint_position, get_joint_limits, WorldSaver, \
    get_pose, interpolate_poses, remove_handles, set_pose, set_joint_position, set_renderer, \
    wait_if_gui, wait_for_user
from pybullet_tools.pr2_primitives import Pose
from pybullet_tools.general_streams import Position
from pybullet_tools.logging_utils import print_debug

from world_builder.entities import Surface


PACKAGE_DIR = abspath(dirname(__file__))
PBP_DIR = abspath(join(PACKAGE_DIR, pardir))
EXP_DIR = abspath(join(PBP_DIR, pardir, 'experiments'))
EXP_REL_PATH = join(pardir, pardir, 'experiments')
DEFAULT_TEMP_DIR = abspath(join(PBP_DIR, pardir, 'examples'))
DEFAULT_IMAGE_DIR = join(DEFAULT_TEMP_DIR, 'log', 'media')
RESULTS_DIR = abspath(join(PBP_DIR, pardir, 'results'))
RESULTS_DATA_DIR = join(RESULTS_DIR, 'data')
cache_name = 'llm_memory.json'

## ---------------------------------------------------------------------------

SUBGOALS_GROUP = ['sequence', 'sequence-reprompt1', 'sequence-reprompt']
ACTIONS_GROUP = ['actions', 'actions-reprompt1', 'actions-reprompt']
SUBGOAL_CONSTRAINTS_GROUP = ['constraints', 'soft_constraints']
PLANNING_MODES = SUBGOALS_GROUP + ACTIONS_GROUP + SUBGOAL_CONSTRAINTS_GROUP + [None]

IMAGE_MODES = ['llm', 'pybullet', 'gym']

## ---------------------------- llamp_agent ----------------------------
END = 'end'

STARTED = 'started'
ALREADY = 'already'
SOLVED = 'solved'
FAILED = 'failed'
RESTART = 'restart'
UNGROUNDED = 'ungrounded'
SUCCEED = 'succeed'
ALL_LLAMP_AGENT_STATUS = [STARTED, ALREADY, SOLVED, FAILED, RESTART, UNGROUNDED, SUCCEED]

STATUS_TO_COLOR = {
    STARTED: 'purple',
    ALREADY: 'blue',
    SOLVED: 'green',
    FAILED: 'red',
    RESTART: 'purple',
    UNGROUNDED: 'orange',
    SUCCEED: 'yellow',
    END: 'yellow'
}

# status_to_color = {SOLVED: '#27ae60', FAILED: '#c0392b', ALREADY: '#3498db'}
## ---------------------------------------------------------------------------

## names in subgoals are initially described to be easy for VLM to read
## contains many versions of similar actions or predicates
preds_rename = {
    'sprinkled-into': 'sprinkledto',
    'sprinkled-to': 'sprinkledto',
    'poured-into': 'pouredto',
    'poured-to': 'pouredto',
    'opened': 'openedjoint',
    'closed': 'closedjoint',
    'pulled-open': 'openedjoint',
    'pulled-close': 'closedjoint',
    'opened-door': 'openedjoint',
    'opened-drawer': 'openedjoint',
    'closed-door': 'closedjoint',
    'closed-drawer': 'closedjoint',
    'turned-on': 'openedjoint',
    'turned-off': 'closedjoint',
}
preds_rename.update({
    'place': 'arrange'
})
pseudo_pull_actions = ['open', 'close', 'turn-on', 'turn-off']
# preds_rename.update({
#     k: pull_actions for k in pseudo_pull_actions
# })

preds_skipped = ['pressed', 'stirred', 'chopped']
preds_skipped += ['stir', 'chop']


## --------------------------------------------------------------------------

ch = '_'


def idx_branch_from_subgoal(subgoal):
    idx, branch = subgoal[:subgoal.index('[')-1].split('_')
    branch = branch.strip()
    return idx, branch


def grow_branch_from_node(node, plan, n):
    from anytree import Node

    def get_alphabet(n):
        alphabets = 'xyz'
        assert n < 3
        return alphabets[n]

    current = node
    for i, action in enumerate(plan + [END]):
        key = f"{i + 1}{ch}{get_alphabet(n)}{ch}{action}"
        current = Node(key, parent=current)


# def append_to_end_node(current_node, new_node):
#     while current_node:
#         start_node = self_


def get_alphabet(n):
    alphabets = 'abcdefghijklmnopqrstuvwxyz'
    if n < 26:
        return alphabets[n]
    mod = n // 26
    remaining = n % 26
    return alphabets[mod-1]+alphabets[remaining]


def plans_to_tree(lists_of_plans, export_path, no_end=False):
    from anytree import Node
    root = Node("start")
    loc_to_node = {"start": root}
    node_counts = {"start": 0}

    end = f'{ch}end' if not no_end else ''
    for j, plan in enumerate(lists_of_plans):
        current = root
        node_counts[current] = node_counts.get(current, 0) + 1
        for i, action in enumerate(plan + [END]):
            if i == len(plan):
                key = f"{get_alphabet(j)}{end}"
            else:
                key = f"{i+1}{ch}{get_alphabet(j)}{ch}{action}"
                for k in range(j):
                    key2 = f"{i+1}{ch}{get_alphabet(k)}{ch}{action}"
                    if key2 in loc_to_node and loc_to_node[key2].parent == current:
                        key = key2
                        break
            if key not in loc_to_node:
                loc_to_node[key] = Node(key, parent=current)
            node_counts[key] = node_counts.get(key, 0) + 1
            current = loc_to_node[key]

    export_tree_png(root, export_path)
    return root, node_counts


def export_tree_png(root, export_path, print_color=True, verbose=False):
    from anytree.exporter import UniqueDotExporter

    if not print_color:
        UniqueDotExporter(root).to_picture(export_path)

    else:

        if verbose:
            print('export_tree_png')

        def nodenamefunc(node):
            return node.name

        def nodeattrfunc(node):
            attrs = []
            attrs += [f'color={node.color}'] if hasattr(node, 'color') else []
            attrs += [f'shape={node.shape}'] if hasattr(node, 'shape') else []
            if verbose:
                print('\t', node.name, '\t', node.depth, '\t', attrs)
            return ', '.join(attrs)

        UniqueDotExporter(root, nodenamefunc=nodenamefunc, nodeattrfunc=nodeattrfunc).to_picture(export_path)

        if verbose:
            print()


def load_agent_memory(agent_memory_path):
    """ e.g.
    "load_memory": null,
    "load_agent_state": null,
    "objects": [],
    "observed": "the pot body is on the stove on the right,\nthe pot lid is on the stove on the left,\nthe chicken leg is on the fridge shelf,\nthe salt shaker is on the cabinet,\nthe pepper shaker is on the cabinet,\nthe fork is on the top drawer space,\ncabinet right door is partially open,\ncabinet left door is partially open,\ntop drawer is fully closed,\nfridge door is partially open,\nstove knob on the right is turned off,\nstove knob on the left is turned off",
    "replan_memory": [],
    "goal": "make chicken soup",
    "lists_of_subgoals": [],
    "lists_of_subgoals_to_print": []
    """
    if not agent_memory_path.endswith('agent_memory.json'):
        agent_memory_path = join(agent_memory_path, 'agent_memory.json')
    return json.load(open(agent_memory_path, 'r'))


def fix_server_path(path):
    """ in order to replay the plans generated on other hosts, may need to fix the abspath saved """
    hostname = os.uname()[1]
    if hostname == 'meraki':
        path = path.replace('/home/zhutiany/Documents/', '/home/yang/Documents/nvidia/')
    elif hostname == '223da4c-lcedt':
        path = path.replace('/home/yang/Documents/nvidia/', '/home/zhutiany/Documents/')
    return path


def fix_experiment_path(path):
    ## manual fix path because it's changed to reprompt1 manually
    run_name = path
    if '/states/' in run_name:
        run_name = run_name.split('/states/')[0]
    elif '/log' in run_name:
        run_name = run_name.split('/log')[0]

    if not isdir(path) and run_name.split('/')[-2].endswith('_reprompt'):
        run_name_possible = run_name.replace('_reprompt', '_reprompt1')
        if isdir(run_name_possible):
            path = path.replace('_reprompt', '_reprompt1')
    return path


def load_vlm_memory(memory_path, verbose=True):
    if '.json' not in memory_path:
        memory_path = join(memory_path, cache_name)
    if not isfile(memory_path):
        print(f'[vlm_planning_api._load_memory]\t loading from file but not found: {memory_path}')
        return None

    memory = json.load(open(memory_path, 'r'))
    memory['is_loaded'] = True
    responses = {key: value['response'] for key, value in memory.items() if key != 'is_loaded'}
    if verbose:
        print(f"\nLLAMPApi | Loaded memory from {abspath(memory_path)}\n")

    return memory, responses


def process_test_for_html(v):
    return '<br>\n'.join(v.replace('<', '&lt;').replace('>', '&gt;').split('\n'))


def add_answer_to_chat(lst, answer, key='answer'):
    if isinstance(answer, list):
        lst += [(key, process_test_for_html(s)) for s in answer]
    else:
        lst += [(key, process_test_for_html(answer))]


def add_prompt_answer_to_chat(lst, memory_dict):
    from vlm_tools.visualize.viz_run_utils import image_row
    prompt = process_test_for_html(memory_dict['prompt'])
    image_path = memory_dict['image_path']
    if isinstance(image_path, str) and isfile(image_path):
        prompt += '<br>' + image_row.format(img_path=image_path.split('/log/')[-1])
    lst += [('prompt', prompt)]
    add_answer_to_chat(lst, memory_dict['response'])


# Function to encode the image
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def show_rgb(rgba_matrix):
    from PIL import Image
    image = Image.fromarray(rgba_matrix, 'RGBA')
    image.show()


def save_rgb_jpg(rgba_matrix, jpg_path='observation.jpg'):
    from PIL import Image
    image = Image.fromarray(rgba_matrix, 'RGBA')
    image_rgb = image.convert('RGB')
    image_rgb.save(jpg_path)


def rindex(lst, item):
    return len(lst) - lst[::-1].index(item) - 1


def is_interactive():
    import __main__ as main
    return not hasattr(main, '__file__')


def repair_facts(world, g, p, subgoal=None):
    """ given a LLM generated fact g : (pred arg1 arg2)
        and printable version p : (pred arg1_english arg2_english)
        return modified g and p such that the typing is correct """

    pred = g[0]
    if pred in ['c']:
        return g, p

    movable_types = world.get_type(g[1])
    container_types = world.get_type(g[-1])

    # if subgoal is not None:
    #     if subgoal == 'in(chicken leg, fridge shelf)':
    #         print('[repair_facts]')

    ## TODO: hack for correcting VLM translation mistakes
    if subgoal in ['in(chicken leg, fridge shelf)', 'in(salt shaker, cabinet)', 'in(pepper shaker, cabinet)']:
        pred = 'picked'
        g = [pred, g[1]]
        p = [pred, p[1]]
        print_debug(f'[vlm_utils.repair_facts] repaired {subgoal} to {p}')

    ## in(object, surface) -> on
    elif pred == 'in' and 'surface' in container_types and 'space' not in container_types:
        g[0] = p[0] = pred = 'on'

    ## on(object, space) -> in
    elif pred == 'on' and 'space' in container_types and 'surface' not in container_types:
        g[0] = p[0] = pred = 'in'

    ## in/on(space/surface, container) -> in/on(object, container)
    elif pred in ['in', 'on'] and len(set(movable_types).intersection({'space', 'surface'})) != 0:
        obj = world.BODY_TO_OBJECT[g[1]]
        g[1] = obj.body
        p[1] = world.get_english_name(g[1])

    ## in/on(object, obj/joint) -> in/on(object, space/surface)
    elif pred in ['in', 'on'] and len(set(container_types).intersection({'space', 'surface'})) == 0:
        obj = world.BODY_TO_OBJECT[g[-1]]
        regions = [s for s in world.cat_to_bodies('surface') + world.cat_to_bodies('space') \
                   if s == obj.body or (isinstance(s, tuple) and s[0] == obj.body)]
        regions = [s for s in regions if s not in world.not_stackable[g[1]] and s not in world.not_containable[g[1]]]
        if len(regions) > 1:
            # surfaces = [random.choice(regions)]
            regions = _sort_surfaces_by_distance(g[-1][0], g[-1][-1], regions)
        if len(regions) > 0:
            g[0] = p[0] = pred = 'on' if regions[0] in world.cat_to_bodies('surface') else 'in'
            g[-1] = regions[0]
            p[-1] = world.get_english_name(g[-1])

    ## on(object, space) -> in
    elif pred == 'on' and 'braiserbody' in world.get_name(g[-1]) and 'braiserlid' not in world.get_name(g[1]):
        g[-1] = world.name_to_body('braiser_bottom')
        p[-1] = world.get_english_name(g[-1])

    elif pred in ['openedjoint', 'closedjoint']:
        obj = world.BODY_TO_OBJECT[g[-1]]

        if 'movable' in container_types:
            if pred == 'openedjoint':
                g[0] = p[0] = 'holding'
            else:
                g[0] = p[0] = 'moved'

        if len(set(container_types).intersection({'surface', 'space'})) > 0:
            if len(obj.governing_joints) > 0:
                joint = obj.governing_joints[0]
            else:
                joints = {j: world.get_english_name(j) for j in world.cat_to_bodies('door') if j[0] == obj.body}
                ## find which joint has best matching name
                if len(joints) > 1:
                    obj_elems = set(p[-1].split(' '))
                    matching = {j: len(obj_elems.intersection(set(v.split(' ')))) for j, v in joints.items()}
                    joints = sorted(matching.keys(), key=lambda x: matching[x], reverse=True)
                joint = joints[0]
            g[-1] = joint
            p[-1] = world.get_english_name(g[-1])

    return g, p


def strip(line):
    line = line.replace('- ', '').replace("'", '').replace("[", '').replace("]", '')
    if '. ' in line:
        line = line[line.index('. ') + 1:]
    return line.strip()


def _sort_surfaces_by_distance(body, link, surfaces):

    def get_link_center(l):
        return np.asarray(get_link_pose(body, l)[0])

    def get_link_dist(l1, l2):
        return np.linalg.norm(get_link_center(l1) - get_link_center(l2))

    dist = {x: get_link_dist(link, x[-1]) for x in surfaces}
    surfaces = sorted(surfaces, key=lambda x: dist[x])[:1]
    return surfaces


###############################################################################


def parse_lines_into_lists_claude3(string, n=1):
    """ given a list of lines (subgoal_english or subgoal_strings) """

    lists_of_parsed = []
    lists = string.split('[\n')[1:]
    for lst in lists:
        elems = lst.replace('\n]', '').split("',\n'")
        elems = [elem.replace("'", '').replace("\n", '') for elem in elems]
        lists_of_parsed.append(elems)
    return lists_of_parsed


def parse_lines_into_lists_gpt4(string, n=1, planning_mode='sequence'):
    """ given a list of lines (subgoal_english or subgoal_strings) """

    def filter(lst):
        return [strip(s) for s in lst if 'plaintext' not in s and '```' not in s and 'python' not in s]

    string = string.replace('"', "'").replace(' \n', "\n")

    lists_of_parsed = []
    if n != 1:
        string = string.replace("[\n'", "").replace("',\n]", "").replace("'\n]", "")
        string = string.replace('- ', '').replace("'", '').replace(",\n", "\n")
        parsed_list = [s.strip() for s in string.split("\n")]

        for k in range(1, n + 1):
            # key_term_1 = f"List {k}:"
            # key_term_2 = f"List {k + 1}:"
            # if model_name == 'gpt-4o':
            key_term_1 = f"subgoals_{k} = ["
            key_term_2 = f"subgoals_{k + 1} = ["
            if planning_mode in ACTIONS_GROUP:
                key_term_1 = f"actions_{k} = ["
                key_term_2 = f"actions_{k + 1} = ["

            if key_term_1 not in parsed_list:
                n = 1
                break
            start = parsed_list.index(key_term_1)

            if key_term_2 not in parsed_list:
                n = 1
                end = -1
            else:
                end = -1 if k == n else parsed_list.index(key_term_2)
            content = parsed_list[start + 1: end]
            if len(content) > 0:
                lists_of_parsed.append(filter(content))

    if n == 1 and len(lists_of_parsed) == 0:
        if '[' in string:
            string = string[string.index('[')+1:string.index(']')]
        if ',\n' in string:
            elems = string.split(',\n')
        elif "',\n'" in string:
            elems = string.split("',\n'")
        elif ".\n" in string:
            elems = string.split(".\n")
        elif ";\n" in string:
            elems = string.split(";\n")
        else:
            elems = string.split("\n")
        parsed_list = filter(elems)
        lists_of_parsed = [parsed_list]

    # for parsed in lists_of_parsed:
    #     for i, line in enumerate(parsed):
    #         parsed[i] = strip(line)
    return lists_of_parsed


###############################################################################


def parse_subgoals(world, subgoals_string, parse_lines_into_lists_fn, subgoals_english=None, **kwargs):
    """ parse_lines_into_lists_gpt4() """
    if isinstance(subgoals_string, list) and len(subgoals_string) == 1:
        subgoals_string = subgoals_string[0]
    lists_of_subgoals = parse_lines_into_lists_fn(subgoals_string, **kwargs)
    # lists_of_subgoals_english = [parse_lines_into_lists_fn(string, n=1)[0] for string in subgoals_english]

    parsed_lists_of_subgoals = []
    parsed_lists_of_subgoals_to_print = []
    for subgoals_list in lists_of_subgoals:
        subgoals = []
        subgoals_to_print = []
        for subgoal in subgoals_list:
            if len(subgoal) < 4:
                continue
            pred, args = subgoal.replace(')', '').split('(')
            if pred in preds_skipped:
                continue
            if pred in preds_rename:
                pred = preds_rename[pred]

            g = [pred]
            p = [pred]

            found_unknown_object = False
            for n in args.split(', '):
                if n not in world.english_name_to_body or world.english_name_to_body[n] not in world.BODY_TO_OBJECT:
                    if n in ['arm', 'hand'] and len(g) == 2:
                        g = ['holding', 'left', g[-1]]
                        p = ['holding', 'left', p[-1]]
                        break

                    print(f'\t skipping unknown object {n} in {subgoal}')
                    found_unknown_object = True
                    break
                g.append(world.english_name_to_body[n])
                p.append(n)

            if not found_unknown_object:
                g, p = repair_facts(world, g, p, subgoal=subgoal)
                # if isinstance(pred, list):
                #     subgoals.extend([[gg] + g[1:] for gg in g[0]])
                #     subgoals_to_print.extend([[pp] + p[1:] for pp in p[0]])
                # else:
                subgoals.append(g)
                subgoals_to_print.append(p)  ## TODO: [s.replace(' ', '__') for s in p] causes HPN inference problem

        print_subgoals(subgoals_to_print, subgoals=subgoals, text_list=subgoals_list)
        parsed_lists_of_subgoals.append(subgoals)
        parsed_lists_of_subgoals_to_print.append(subgoals_to_print)

    return parsed_lists_of_subgoals, parsed_lists_of_subgoals_to_print


def print_subgoals(subgoals_to_print, subgoals=None, text_list=None):
    from pybullet_tools.logging_utils import myprint as print
    print('\n[vlm_utils.print_subgoals]')
    for i, lst in enumerate(subgoals_to_print):
        if subgoals is not None:
            subgoals_lst = subgoals[i]
            lst = lst[:1] + [f"{lst[j]}|{subgoals_lst[j]}" for j in range(1, len(lst))]
        line = f"{i}\t{lst}"
        if text_list is not None:
            line += f"  <-- {text_list[i]}"
        print(line)
    print()


def parse_state_changes(world, subgoals_strings, parse_lines_into_lists_fn, subgoals_english, objects=None, n=1, visualize=True):
    def parse_lines(lines):
        changes = []
        for line in lines:
            if ' is at ' not in line:
                break
            name, value = line.split(' is at ')
            body = world.english_name_to_body[name]
            value = eval(value)
            typ = 'pose' if isinstance(value, tuple) else 'position'
            changes.append({'type': typ, 'body': body, 'value': value})
        return changes

    parsed_states_english = [parse_lines_into_lists_fn(string, n=1)[0] for string in subgoals_english]
    parsed_lists_of_subgoals = []
    parsed_lists_of_subgoals_to_print = []
    for i in range(n):
        subgoals_string = subgoals_strings[i]

        states = []
        if '\n\n' in subgoals_string:
            subgoals_lines = subgoals_string.split('\n\n')
            for j, lines in enumerate(subgoals_lines):
                lines = [strip(line) for line in lines.split('\n')]
                english = lines[0]
                states.append({'english': english, 'changes': parse_lines(lines[1:])})
        else:
            subgoals_lines = subgoals_string.split('\n')
            for j, lines in enumerate(subgoals_lines):
                lines = [strip(line) for line in lines.split('; ')]
                english = parsed_states_english[i][j]
                states.append({'english': english, 'changes': parse_lines(lines)})

        if visualize:
            visualize_state_changes(world, states, objects=objects, timestep=0.1)

        subgoals = []
        subgoals_to_print = []
        for state in states:
            for change in state['changes']:
                body = change['body']
                value = change['value']
                if change['type'] == 'pose':
                    pred = 'AtPose'
                    obj_value = Pose(body, (value, get_pose(body)[1]))
                else:
                    pred = 'AtPosition'
                    if value > 1.57:  ## debug
                        value = 1.57
                    obj_value = Position(body, value)
                subgoals.append([pred, body, obj_value])
                subgoals_to_print.append([pred, world.body_to_english_name[body], value])

        print(subgoals_to_print)
        parsed_lists_of_subgoals.append(subgoals)
        parsed_lists_of_subgoals_to_print.append(subgoals_to_print)

    return parsed_lists_of_subgoals, parsed_lists_of_subgoals_to_print


def visualize_state_changes(world, states, objects=None, timestep=0.1):

    def interpolate_numbers(start, end, num_steps=8):
        step_size = (end - start) / num_steps
        interpolated_values = [start + step_size * i for i in range(num_steps + 1)]
        return interpolated_values

    def interpolate_points(old_pose, pose, pos_step_size=0.1):
        poses = []
        for pose in interpolate_poses(old_pose, pose, pos_step_size=pos_step_size):
            poses.append(pose)
        return poses

    def set_body_pose(body, pose, old_handles):
        old_pose = get_pose(body)
        poses = interpolate_points(old_pose, (pose, old_pose[1]))
        remove_handles(old_handles)
        handles = draw_pose3d_path(poses)
        for obj_pose in poses:
            set_pose(body, obj_pose)
            time.sleep(timestep)
        return handles

    def set_body_joint_position(body_joint, position):
        body, joint = body_joint
        old_position = get_joint_position(body, joint)
        positions = interpolate_numbers(old_position, position)
        for jp in positions:
            set_joint_position(body, joint, jp)
            time.sleep(timestep)

    handles = []
    # current_state = get_current_object_state(world, objects, return_strings=False)
    with WorldSaver():
        set_renderer(True)
        wait_for_user("start visualizing state changes?")
        for i, state in enumerate(states):
            print(f"\n{i+1}: "+state['english'])
            for change in state['changes']:
                body = change['body']
                value = change['value']
                print(f"\t{body}: {value}")
                if change['type'] == 'position':
                    set_body_joint_position(body, value)
                if change['type'] == 'pose':
                    handles = set_body_pose(body, value, handles)
                wait_if_gui()


def get_current_object_state(world, objects, return_strings=True):
    pose_template = "{name} is at {pose}"
    position_template = "{name} is at {position}"

    movable_object_poses = []
    static_object_poses = []
    joint_positions = []
    joint_limits = []
    for body in objects:
        name = world.get_english_name(body)
        obj = world.BODY_TO_OBJECT[body]
        if isinstance(body, int):
            line = pose_template.format(name=name, pose=nice(obj.get_pose()[0]))
            if body in world.movable:
                movable_object_poses.append(line)
            else:
                static_object_poses.append(line)

        elif isinstance(body, tuple) and len(body) == 3:
            aabb = get_aabb(body[0], link=body[-1])
            pose = (np.asarray(aabb.lower) + np.asarray(aabb.upper)) / 2
            if isinstance(obj, Surface):
                pose[2] = aabb.upper[2]
            static_object_poses.append(pose_template.format(name=name, pose=nice(pose)))

        elif isinstance(body, tuple) and len(body) == 2:
            position = get_joint_position(body[0], body[1])
            limits = get_joint_limits(body[0], body[1])
            joint_positions.append(position_template.format(name=name, position=nice(position)))
            joint_limits.append(f"{name}: {nice(limits)}")

    if return_strings:
        sep = '; '
        current_state = sep.join(movable_object_poses) + '\n' + sep.join(joint_positions)
        static_object_state = sep.join(static_object_poses)
        joint_limits = sep.join(joint_limits)
        return dict(current_state=current_state, static_object_state=static_object_state, joint_limits=joint_limits)
    else:
        return dict(movable_object_poses=movable_object_poses, static_object_poses=static_object_poses,
                    joint_positions=joint_positions, joint_limits=joint_limits)


## -----------------------------------------------------------------------------------------


def get_subdir_name(planning_mode='sequence', dual_arm=True, mode='eval',
                    version='', problem_name='kitchen_chicken_soup',
                    llamp_planning_mode=None, difficulty=None, **kwargs):
    """ e.g. {llm_only}_{kitchen_chicken_soup}_{v0}_{actions}_{dual_arm} """

    if llamp_planning_mode is not None:
        planning_mode = llamp_planning_mode
    if difficulty is not None:
        version = f'v{difficulty}'

    exp_subdir = {'llm': 'llm_only', 'eval': 'run'}[mode]
    exp_subdir += f'_{problem_name}'.replace('test_', '')

    if len(version) > 0:
        exp_subdir += f'_{version}'

    exp_subdir += '_actions' if planning_mode in ACTIONS_GROUP else '_subgoals'

    if mode == 'eval':
        if 'reprompt1' in planning_mode:
            exp_subdir += '_reprompt1'
        elif 'reprompt' in planning_mode:
            exp_subdir += '_reprompt'

    if dual_arm:
        exp_subdir += '_dual_arm'

    return exp_subdir


def load_prompts(**kwargs):
    exp_name = get_subdir_name(mode='llm', **kwargs)
    exp_dir = join(EXP_DIR, exp_name)
    subdirs = [join(exp_dir, f) for f in listdir(exp_dir) if isdir(join(exp_dir, f))]
    subdirs.sort()
    print(f'Loaded {len(subdirs)} prompt dirs from {exp_dir}')
    return subdirs


def sample_prompt(randomly=True, **kwargs):
    """ kwargs could come from the args for run_agent(), which slightly differ from those taken by load_prompts """
    mod_kwargs = copy.deepcopy(kwargs)
    if 'llamp_planning_mode' in mod_kwargs:
        mod_kwargs['planning_mode'] = mod_kwargs.pop('llamp_planning_mode').replace('-reprompt', '')
    if 'difficulty' in mod_kwargs:
        difficulty = mod_kwargs.pop('difficulty')
        mod_kwargs['version'] = f"v{difficulty}"
    if 'exp_subdir' in mod_kwargs:
        mod_kwargs.pop('exp_subdir')

    subdirs = load_prompts(**mod_kwargs)
    if randomly:
        return random.choice(subdirs)
    return subdirs[0]


def enumerate_exp_names(
        planning_modes=tuple(SUBGOALS_GROUP + ACTIONS_GROUP),
        if_dual_arm=tuple([False, True]),
        versions=tuple(["v0", "v1"]),
        problem_names=tuple(["test_kitchen_chicken_soup"]),
        **kwargs
    ):
    for problem_name in problem_names:
        for version in versions:
            for dual_arm in if_dual_arm:
                for planning_mode in planning_modes:
                    yield get_subdir_name(planning_mode, dual_arm, version=version, problem_name=problem_name, **kwargs)
