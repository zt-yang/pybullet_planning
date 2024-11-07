import copy
import shutil
from os.path import basename
from pathlib import Path
import json

import os
import sys
from os.path import join, abspath, dirname, isdir, isfile
R = abspath(join(dirname(__file__), os.pardir, os.pardir))
sys.path.extend([R] + [join(R, name) for name in ['pddlstream', 'pybullet_planning', 'lisdf']])


from pigi_tools.data_utils import aabb_placed_in_aabb

from lisdf_tools.image_utils import merge_center_crops_of_images

from pybullet_tools.logging_utils import TXT_FILE as default_log_path, myprint as print

from world_builder.paths import PBP_PATH
from world_builder.init_utils import add_joint_status_facts
from world_builder.world_utils import get_camera_image, make_camera_collage, \
    make_camera_image_with_object_labels, sort_body_indices, KNOB, JOINT, \
    SURFACE, SPACE, MOVABLE
from world_builder.world import World

from vlm_tools.vlm_api import GPT4vApi, Claude3Api
from vlm_tools.prompts_gpt4v import *
# from vlm_tools.examples.prompts_playground import *
from vlm_tools.visualize.viz_run_utils import output_html, launch_log_page, media_dir, content_dir
from vlm_tools.vlm_utils import plans_to_tree, rindex, export_tree_png, grow_branch_from_node, \
    add_answer_to_chat, IMAGE_MODES, add_prompt_answer_to_chat, process_test_for_html, \
    parse_subgoals, cache_name, get_current_object_state, parse_state_changes, load_vlm_memory, \
    parse_lines_into_lists_claude3, parse_lines_into_lists_gpt4, PLANNING_MODES, ACTIONS_GROUP, \
    STARTED, ALREADY, SOLVED, FAILED, UNGROUNDED, SUCCEED, RESTART, ALL_LLAMP_AGENT_STATUS, END, \
    STATUS_TO_COLOR, fix_server_path, idx_branch_from_subgoal


def get_llamp_api(api_class_name):
    if 'gpt' in api_class_name:
        return GPT4PlanningApi
    elif 'claude' in api_class_name:
        return Claude3PlanningApi
    return LLAMPApi


class LLAMPApi(object):
    """ LLM-Advised Planning """

    llm = None

    def __init__(self, open_goal, planning_mode='sequence', image_mode='pybullet',
                 load_memory=None, seed=None, log_dir=abspath('log')):
        """
        planning_mode:
            sequence:           solve each subgoal in sequence
            sequence-reprompt:  reprompt when failed
            actions:            refining each action
            actions-reprompt:   reprompt when failed
        image_mode:
            llm:        query with text only
            pybullet:   + image rendered from pybullet
            gym:        + image rendered from issac gym
        """

        self.obs_idx = 0  ## the number of total planner runs
        self.log_rounds = []  ## planner runs grouped by query responses
        self.log_rows = ''  ## the log html table

        self.open_goal = open_goal
        assert planning_mode in PLANNING_MODES
        self.planning_mode = planning_mode
        assert image_mode in IMAGE_MODES
        self.image_mode = image_mode

        self.load_memory = load_memory if isinstance(load_memory, str) and load_memory.lower() != 'null' else None
        self.agent_memory = {
            'load_memory': self.load_memory,
            'load_agent_state': None,
            'seed': seed
        }
        self.suffix = ''
        self.agent_state_path = None

        self.world = None  ## initiate during the first query
        self.body_to_english_names = None  ## dict of object (cat, body) to english name
        self.allowed_num_reprompting = 1

        ## initially in vlm-tamp/examples/, later reset by self.init_log_dir()
        self.log_dir = log_dir
        self.obs_dir = join(log_dir, media_dir)
        self.planning_tree_path = join(log_dir, os.pardir, "planning_tree.png") ## abspath("planning_tree.png")

        self.plan_tree_root = None
        self.node_counts = None
        self.visited_junction_nodes = []
        self.visited_nodes = []
        self.succeeded_nodes = []
        self.failed_nodes = []
        self.current_planning_node = None
        self.current_planning_option = None
        self.all_lists_of_subgoals = None

    def parse_lines_into_lists_fn(self, string: str, **kwargs):
        assert NotImplementedError

    def ask(self, prompt: str, **kwargs):
        return self.llm.ask(prompt, **kwargs)

    ##############################################################################

    def new_session(self, image_dir=None, cache_dir=None):
        if cache_dir is not None:
            cache_file = join(self.llm.cache_dir, cache_name)
            if isfile(cache_file):
                os.rename(cache_file, join(cache_dir, cache_name))
        self.llm.new_session(image_dir=image_dir, cache_dir=cache_dir)

    def save_agent_memory(self):
        memory_path = abspath(join(self.log_dir, '..', 'agent_memory.json'))
        print(f'llamp_api.save_agent_memory to {memory_path}')
        with open(memory_path, 'w') as f:
            json.dump(self.agent_memory, f, indent=3)

    def _load_memory(self):
        """ the given path is either a subdir in experiments or a file in vlm_tools"""
        load_memory = fix_server_path(self.load_memory)
        memory_path = abspath(join(PBP_PATH, load_memory))

        results = load_vlm_memory(memory_path)
        if results is None:
            return None

        self.llm.memory, responses = results

        if '.json' not in self.load_memory:
            self.llm.save_memory()
        return responses

    def set_planning_world(self, world: World = None, objects: list = None, observed: str = None):
        """
            goal:       in natural english
            objects:    a list of pybullet bodies
        """
        if objects is None:
            objects = world.BODY_TO_OBJECT

        all_body_to_english_names = {body: world.body_to_object(body).name for body in objects}
        if observed is None:
            body_to_english_names, description = get_observed_objects(world, objects)
            observed = ",\n".join(description)
            print('-'*80)
            print(observed)
            print('-'*80)
            self.body_to_english_names = body_to_english_names

        ## the main query
        if self.world is None:
            self.world = world
            self.body_to_english_names[self.world.robot.body] = 'robot'
            self.agent_memory.update({'objects': objects, 'observed': observed, 'replan_memory': []})

        ## reprompts
        else:
            self.agent_memory['replan_memory'].append({'observed': observed})
            objects = self.agent_memory['objects']
            self.suffix = "__reprompt_" + str(len(self.agent_memory['replan_memory']))

        ## generate image for querying vlm
        if self.image_mode == 'pybullet':
            self.generate_query_images(self.body_to_english_names)

        return world, objects, observed

    def load_llm_answers(self):
        subgoals_english, subgoals_string, responses = None, None, None
        if self.load_memory is not None:
            responses = self._load_memory()
            if responses is not None:
                key_1 = 'subgoals_english'
                key_2 = 'subgoals_string'
                if self.planning_mode in ACTIONS_GROUP:
                    key_1 = 'actions_english'
                    key_2 = 'actions_string'
                if key_1 in responses:
                    subgoals_english = responses[key_1]
                if key_2 in responses:
                    subgoals_string = responses[key_2]
        return subgoals_english, subgoals_string, responses

    ## -----------------------------------------------------------------------------------------

    def get_subgoals(self, world: World = None, objects: list = None, observed: str = None,
                     history=None, **kwargs):
        """ given a goal expressed in English, decompose it into possible sequences of subgoals using LLM """

        goal = self.open_goal

        query_fn = self._query_subgoals
        key = 'lists_of_subgoals'
        if self.planning_mode in ACTIONS_GROUP:
            query_fn = self._query_actions
            key = 'lists_of_actions'

        n_arms = 'one arm' if not world.robot.dual_arm else 'two arms'
        world, objects, observed = self.set_planning_world(world, objects, observed)
        object_names = [world.get_english_name(body) for body in objects]
        history_content = '' if history is None else history
        world_args = dict(objects=object_names, observed=observed, n_arms=n_arms, history=history_content)

        first_query_kwargs = dict(image_name=None, image_description='')
        if self.image_mode == 'pybullet':
            first_query_kwargs = dict(
                image_name=f'query_{self.obs_idx}.png',
                image_description=composed_annotated_image_description
            )

        lists_of_subgoals, lists_of_subgoals_to_print, chat = query_fn(
            goal, world, world_args=world_args, objects=objects,
            first_query_kwargs=first_query_kwargs, **kwargs
        )
        self.log_rounds.append({'chat': chat, 'subplans': []})

        ## main prompt
        if history is None:
            self.plan_tree_root, self.node_counts = plans_to_tree(lists_of_subgoals_to_print, self.planning_tree_path)
            self.current_planning_node = self.plan_tree_root
            self.visited_junction_nodes = [self.plan_tree_root]
            self.visited_nodes = [self.plan_tree_root]
            self.all_lists_of_subgoals = lists_of_subgoals
            self.current_planning_option = 'a'
            self.agent_memory.update({
                'goal': goal,
                key: lists_of_subgoals,
                f'{key}_to_print': lists_of_subgoals_to_print
            })

        ## reprompt
        else:

            node = self.current_planning_node = self.current_planning_node.parent
            n = len(self.agent_memory['replan_memory']) - 1
            grow_branch_from_node(node, lists_of_subgoals_to_print[0], n)
            export_tree_png(self.plan_tree_root, self.planning_tree_path)
            self.visited_junction_nodes.append(node)
            self.current_planning_option = idx_branch_from_subgoal(node.children[-1].name)[1]

            self.agent_memory['replan_memory'][-1].update({
                key: lists_of_subgoals,
                f'{key}_to_print': lists_of_subgoals_to_print
            })
            self.agent_memory['replan_memory'][-1].update({
                'already_succeeded': self.get_succeeded_subgoals_to_print(),
            })
        return lists_of_subgoals[0]

    ## -----------------------------------------------------------------------------------------------

    def _query_actions(self, goal, world, world_args, temperature: float = 0.2, first_query_kwargs=dict(),
                       include_preconditions=True, objects=None):

        reprompted = len(world_args['history']) > 0

        ## ------------- question 1: actions in english -------------
        prompt = prompt_actions_english.format(goal=goal, **world_args)

        ## ------------- question 2: actions in predicates -------------
        actions = list_of_actions_with_preconditions if include_preconditions else list_of_actions
        description = world.get_objects_by_type(objects)
        prompt_translation = prompt_english_to_actions.format(set_of_actions=actions, objects=description)

        actions_english, actions_string, responses = self.load_llm_answers()
        first_prompt_name = 'actions_english'
        if actions_english is None or reprompted:
            first_prompt_name += self.suffix
            actions_english = self.ask(prompt, prompt_name=first_prompt_name, temperature=temperature, **first_query_kwargs)
            if actions_string is None or reprompted:
                actions_string = self.ask(prompt_translation, prompt_name=f'actions_string{self.suffix}')

        print(actions_string)
        print('-' * 40 + '\n')

        # chat = [('prompt', prompt)]
        # add_answer_to_chat(chat, actions_english)
        chat = []
        add_prompt_answer_to_chat(chat, self.llm.memory[first_prompt_name])
        chat += [('prompt', process_test_for_html(prompt_translation))]
        add_answer_to_chat(chat, actions_string)

        ## ------------- process the response generated by GPT-4  -------------
        actions, actions_to_print = parse_subgoals(world, actions_string, self.parse_lines_into_lists_fn,
                                                   planning_mode=self.planning_mode)

        for one in actions_to_print:
            chat += [('processed', '<br>'.join([str(s) for s in one]))]

        return actions, actions_to_print, chat

    def _query_subgoals(self, goal, world, world_args, temperature: float = 0.2, first_query_kwargs=dict(),
                        predict_object_states=False, objects=None, verify_subgoals=False, repair_subgoals=False):

        reprompted = len(world_args['history']) > 0

        ## ------------- question 1: subgoals in english -------------
        prompt = prompt_subgoals_english.format(goal=goal, **world_args)

        ## ------------- question 2: subgoals in predicates -------------
        prompt_translation = prompt_english_to_subgoals.format(objects=world.get_objects_by_type(objects))

        subgoals_english, subgoals_string, responses = self.load_llm_answers()
        first_prompt_name = 'subgoals_english'
        if subgoals_english is None or reprompted:
            first_prompt_name += self.suffix
            subgoals_english = self.ask(prompt, prompt_name=first_prompt_name, temperature=temperature, **first_query_kwargs)
            if not predict_object_states or reprompted:
                subgoals_string = self.ask(prompt_translation, prompt_name=f'subgoals_string{self.suffix}')

        ## ------------- question 2b: query changes to object poses and joint positions -------------
        if predict_object_states:
            key = 'state_changes'
            current_state_args = get_current_object_state(world, objects)
            prompt_state_changes = prompt_english_to_object_states.format(**current_state_args)
            prompt_translation = prompt_state_changes
            if self.load_memory is not None and responses is not None and key in responses:
                subgoals_string = responses[key]
            else:
                subgoals_string = self.ask(prompt_state_changes, prompt_name=key)

        print(subgoals_string)
        print('-' * 40 + '\n')

        # chat = [('prompt', prompt)]
        # add_answer_to_chat(chat, subgoals_english)
        chat = []
        add_prompt_answer_to_chat(chat, self.llm.memory[first_prompt_name])
        chat += [('prompt', process_test_for_html(prompt_translation))]
        add_answer_to_chat(chat, subgoals_string)

        ## ------------- process the response generated by GPT-4  -------------
        if predict_object_states:
            subgoals, subgoals_to_print = parse_state_changes(world, subgoals_string, self.parse_lines_into_lists_fn,
                                                              subgoals_english, objects=objects)
            verify_subgoals = False

        else:
            subgoals, subgoals_to_print = parse_subgoals(world, subgoals_string, self.parse_lines_into_lists_fn,
                                                         subgoals_english=subgoals_english)

        for one in subgoals_to_print:
            chat += [('processed', '<br>'.join([str(s) for s in one]))]

        ##  -----------------------------------------------------------------

        if verify_subgoals:
            question = prompt_translate_subgoals_to_english
            subgoals = self.ask(question, promp_name='subgoals_translate')
            chat += [('prompt', question), ('answer', subgoals)]

            question = prompt_goal_achieved.format(goal=goal)
            subgoal_results = self.ask(question, promp_name='subgoals_verify')
            chat += [('prompt', question), ('answer', subgoal_results)]

            if repair_subgoals:
                question = prompt_goal_repair
                subgoals_repaired = self.ask(question, promp_name='subgoals_repair')
                chat += [('prompt', question), ('answer', subgoals_repaired)]

        return subgoals, subgoals_to_print, chat

    ## -----------------------------------------------------------------------------------------

    def backtrack_planning_tree(self, failed_but_continue_branch=False, jump_to_reprompt=False, **kwargs):
        """
        failed_but_continue_branch = True:
            when not in reprompting mode, after failed current node, continue current branch
        jump_to_reprompt = True:
            when in reprompting mode, after failed current node, prompt to start a new branch
        """
        failed_node = False
        chosen_index = -1

        self.failed_nodes.append(self.current_planning_node)

        if failed_but_continue_branch:
            failed_node = True
            # _, branch, goal = self.current_planning_node.name.split('_')
            # append_to_end_node(self.current_planning_node, self.current_planning_node, color='pink')

        elif not jump_to_reprompt:
            if len(self.visited_junction_nodes) == 0:
                failed_node = True

            else:

                last_junction_node = self.visited_junction_nodes[-1]

                ## previous problem with updating self.visited_nodes
                if last_junction_node not in self.visited_nodes or len(self.agent_memory['replan_memory']) > 0:
                    failed_node = True

                else:
                    ## the following are for when there are multiple branches queried at the beginning
                    index = rindex(self.visited_nodes, last_junction_node) + 1
                    if index >= len(self.visited_nodes):
                        failed_node = True

                    else:
                        chosen_children = self.visited_nodes[index]
                        while chosen_children not in last_junction_node.children:
                            chosen_children = chosen_children.parent
                        chosen_index = last_junction_node.children.index(chosen_children)
                        while chosen_index == len(last_junction_node.children) - 1:  ## or chosen_children in self.failed_nodes
                            self.visited_junction_nodes.remove(last_junction_node)
                            if len(self.visited_junction_nodes) == 0:
                                failed_node = True
                                break

                            last_junction_node = self.visited_junction_nodes[-1]
                            chosen_children = self.visited_nodes[rindex(self.visited_nodes, last_junction_node) + 1]
                            chosen_index = last_junction_node.children.rfind(chosen_children)

        if failed_node or jump_to_reprompt:
            reprompt = self.planning_mode.endswith('-reprompt') and len(self.agent_memory['replan_memory']) < 2
            if reprompt:
                history, failure = self.get_action_history_and_failure(**kwargs)
                history_content = include_history.format(actions=history, failure=failure)
                new_plan = [self.suffix] + self.get_subgoals(self.world, history=history_content)
                self.save_agent_memory()
                return new_plan
            return FAILED

        next_chosen = last_junction_node.children[chosen_index + 1]
        self.visited_nodes.append(last_junction_node)

        if next_chosen.name.endswith('_end'):
            return SUCCEED

        self.current_planning_node = last_junction_node
        step, option, _ = next_chosen.name.split('_')
        self.current_planning_option = option
        subgoals = self.all_lists_of_subgoals['abcdefg'.index(option)][int(step)-1:]

        return [RESTART] + subgoals

    def get_action_history_and_failure(self, collision_bodies=None, holding_objects=None):
        subgoals = self.get_succeeded_subgoals_to_print()

        # if previous_goals is not None:
        #     print('[get_action_history_and_failure]')
        #     print(f'\tsubgoals in vlm_api\t {len(subgoals)}\n\t'+'\n\t'.join([str(s) for s in subgoals]))
        #     print(f'\tprevious_goals in llamp_agent\t {len(previous_goals)}\n\t'+'\n\t'.join([str(s) for s in previous_goals]))

        history = [f"{i + 1}. {action}" for i, action in enumerate(subgoals)]
        history = '\n'.join(history)
        if holding_objects is not None:
            holding_line = '\nCurrently, the robot is holding some objects. '
            for arm, obj in holding_objects:
                holding_line += f'The {arm} hand is holding {obj}. '
            history += holding_line

        last_failed = self.get_last_failed_to_print()
        key = 'actions' if self.planning_mode in ACTIONS_GROUP else 'subgoals'
        failure = f"{key} {last_failed}. So please do not list this {key} as the first {key} to achieve in your answer."
        # collisions = self.world.summarize_collisions()
        # if len(collisions) > 0:
        #     print('Collisions:', collisions)
        #     failure += ''
        if collision_bodies is not None and len(collision_bodies) > 0:
            failure += (f'\nWhen trying to solve the previous problem in simulation. '
                        f'The robot has collided with these objects: {collision_bodies}')

        return history, failure

    def get_all_lists_of_subgoals(self):
        key = 'actions' if self.planning_mode in ACTIONS_GROUP else 'subgoals'
        all_subgoals = copy.deepcopy(self.agent_memory[f'lists_of_{key}_to_print'])
        for memory in self.agent_memory['replan_memory']:
            all_subgoals.extend(memory[f'lists_of_{key}_to_print'])
        return all_subgoals

    def get_succeeded_subgoals_to_print(self):
        succeeded_subgoals = []
        all_subgoals = self.get_all_lists_of_subgoals()
        for i, round in enumerate(self.log_rounds):
            for j, problem in enumerate(round['subplans']):
                if problem[-1] == SOLVED:
                    problem_idx = int(problem[0].split('_')[0]) - 1
                    succeeded_subgoals.append(all_subgoals[i][problem_idx])
        return succeeded_subgoals

    def get_last_failed_to_print(self):
        failed = self.log_rounds[-1]['subplans'][-1][0]
        failed = failed[failed.index('_[')+1: failed.index(';')]
        return failed

    #########################################################################

    def init_log_dir(self, log_dir=abspath('log'), serve_page=True):
        """ only when low-level planning problem starts running, before that it's in default """
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        os.makedirs(join(self.log_dir, content_dir))

        ## for viewing the log at http://0.0.0.0:9000/{exp_subdir}/{datetime_expname}/log/
        if serve_page:
            launch_log_page(log_dir)

        ## for viewing the log at http://0.0.0.0:9000/latest_run/log/
        last_log_dir = join(dirname(dirname(dirname(self.log_dir))), 'latest_run')
        if Path(last_log_dir).is_symlink():
            os.unlink(last_log_dir)
        os.symlink(dirname(self.log_dir), last_log_dir)

        ## move images over from default directory
        images = [join(self.obs_dir, f) for f in os.listdir(self.obs_dir) if f.endswith('.png')]

        ## for accessing from cogarch agent
        self.obs_dir = join(self.log_dir, media_dir)
        os.makedirs(self.obs_dir)
        for img in images:
            shutil.move(img, join(self.obs_dir, basename(img)))

        abs_path = abspath(join(self.log_dir, "media", "planning_tree.png"))
        shutil.move(self.planning_tree_path, abs_path)
        self.planning_tree_path = abs_path

        ## save LLAMP agent state
        self.save_agent_memory()

    def log_obs_image(self, cameras, **kwargs):
        camera_images = [get_camera_image(camera, include_rgb=True) for camera in cameras]
        obs_path = join(media_dir, f'observation_{self.obs_idx}.png')
        make_camera_collage(camera_images, output_path=join(self.log_dir, obs_path), **kwargs)
        return obs_path

    def generate_query_images(self, body_to_english_names):
        """ before subgoal planning starts, generate an image for the main query """

        def is_index_category_tuple(item):
            return isinstance(item, tuple) and isinstance(item[-1], str)

        def belongs_to_first_image(item):
            return is_index_category_tuple(item) and item[1] in [MOVABLE, JOINT]

        def belongs_to_second_image(item):
            return is_index_category_tuple(item) and item[1] in [MOVABLE, SURFACE, SPACE]

        def sort_dict(bodies):
            sorted_bodies = sort_body_indices(list(bodies.keys()))
            return {k: bodies[k] for k in sorted_bodies}

        os.makedirs(self.obs_dir, exist_ok=True)
        img_path = join(self.obs_dir, f'query_{self.obs_idx}.png')

        ## add object labels in sim
        world = self.world
        world.initiate_observation_cameras()

        camera = world.cameras[1]  ## camera tilted downward
        # camera_image = get_camera_image(camera, include_rgb=True, include_segment=True)
        # make_camera_image(camera_image.rgbPixels, output_path=img_path)

        from world_builder.loaders_nvidia_kitchen import object_label_locations
        kwargs = dict(object_label_locations=object_label_locations)
        dict_both = {k: v for k, v in body_to_english_names.items() if not is_index_category_tuple(k)}
        dict_1 = {k[0]: v for k, v in body_to_english_names.items() if belongs_to_first_image(k)}
        dict_2 = {k[0]: v for k, v in body_to_english_names.items() if belongs_to_second_image(k)}
        dict_1.update(dict_both)
        dict_2.update(dict_both)
        img_path_1 = img_path.replace('.png', '_1.png').replace('.pdf', '_1.png')
        img_path_2 = img_path.replace('.png', '_2.png').replace('.pdf', '_1.png')
        make_camera_image_with_object_labels(camera, sort_dict(dict_1), output_path=img_path_1, verbose=False, **kwargs)
        make_camera_image_with_object_labels(camera, sort_dict(dict_2), output_path=img_path_2, verbose=False, **kwargs)
        merge_center_crops_of_images([img_path_1, img_path_2], output_path=img_path, spacing=20)

    ## -----------------------------------------------------------------------------------------------

    def _get_english_goal(self, goal):
        args = []
        for arg in goal[1:]:
            if isinstance(arg, int) or isinstance(arg, tuple):
                args.append(self.world.get_english_name(arg))
            elif isinstance(arg, str):
                args.append(arg)
            else:
                arg = arg.value if hasattr(arg, 'value') else arg.values
                if isinstance(arg, tuple) and len(arg) == 2:
                    arg = arg[0]  ## only point
                args.append(arg)
        return str(list(goal[:1]) + args)

    def log_subgoal(self, subgoal, obs_path, status, suffix="", trial_index=0):
        """ update the interactive html page before and after planning
            before:
                subgoal: name of subgoal, status = STARTED
            after:
                subgoal: name of subgoal, status = ALL_LLAMP_AGENT_STATUS - {STARTED}
            finished all planning:
                subgoal: END, status = STARTED
        """
        assert status in ALL_LLAMP_AGENT_STATUS

        if isinstance(subgoal, list) and not isinstance(subgoal[0], str):
            subgoal = subgoal[-1]

        finished_planning = False
        skipped_planning = (status in [ALREADY, UNGROUNDED])
        subgoal_english = None
        if subgoal != END:  ## deal with the last planning problem solved
            subplans = self.log_rounds[-1]['subplans']
            subgoal_english = self._get_english_goal(subgoal)
            finished_planning = len(subplans) > 0 and (subgoal_english in str(subplans[-1][0])) and subplans[-1][-1] == STARTED

        round_idx = len(self.log_rounds) - 1
        subgoal_idx = len(self.log_rounds[-1]['subplans'])

        ## add the button when the planning starts, and replace it planning finishes
        if subgoal == END:
            log_path = join(content_dir, f'log_{round_idx}_{subgoal_idx}.txt')
            os.rename(default_log_path, join(self.log_dir, log_path))
            # self.current_planning_node = self.current_planning_node.children[0]
            status = END

        elif finished_planning or skipped_planning:
            # log_path = self.log_rounds[-1]['subplans'][-1][1]  ## image of init scene for the problem
            log_path = join(content_dir, f'log_{round_idx}_{subgoal_idx-1}.txt')

            if finished_planning:
                self.log_rounds[-1]['subplans'] = self.log_rounds[-1]['subplans'][:-1]  ## remove the line with STARTED
                self.obs_idx += 1

            ## skipped planning
            else:
                i = 0
                while self.current_planning_node.children[i] in self.failed_nodes:
                    i += 1

                ## TODO, may not be enough to skip self.visited_junction_nodes,
                # check last block in _update_next_planning_node
                self.visited_nodes.append(self.current_planning_node)

                self.current_planning_node = self.current_planning_node.children[i]
                log_path = join(content_dir, f'log_{round_idx}_{subgoal_idx}.txt')

            ## if file already exist, append default log to it instead of moving over and overwriting
            log_file = join(self.log_dir, log_path)
            if isfile(log_file):
                old_text = open(log_file, 'r').read()
                new_text = open(default_log_path, 'r').read()
                with open(log_file, 'w') as f:
                    f.write(old_text + '\n' + new_text)
                os.remove(default_log_path)
            else:
                os.rename(default_log_path, log_file)

        ## start a planning problem
        else:
            ## get current planning node
            found = self._update_next_planning_node(subgoal_english, status, trial_index)
            if found is None:
                return None

            ## able to monitor the current world state while waiting for planning result
            log_path = join(content_dir, f'log_{round_idx}_{subgoal_idx}.txt')
            shutil.copy(default_log_path, join(self.log_dir, log_path))

        self.current_planning_node.color = STATUS_TO_COLOR[status]
        export_tree_png(self.plan_tree_root, self.planning_tree_path)

        node_name = self.current_planning_node.name
        self.log_rounds[-1]['subplans'].append((f"{node_name}{suffix}", log_path, obs_path, status))
        self.save_log_rounds()
        return node_name

    def _update_next_planning_node(self, subgoal_english, status, trial_index):
        found = [c for c in self.current_planning_node.children if subgoal_english in c.name]
        if len(found) == 0:
            ## sometimes the predicate name has been changed for debugging
            if ', ' not in subgoal_english:
                print(f'subgoal_english {subgoal_english}')
            subgoal_english = subgoal_english[subgoal_english.index(', '):]
            found = [c for c in self.current_planning_node.children if subgoal_english in c.name]
            if len(found) == 0:
                if subgoal_english in self.current_planning_node.name:
                    print(f'log_subgoal({status}, trial_index={trial_index})'
                          f'\t {subgoal_english} in {self.current_planning_node.name}')
                    found = [self.current_planning_node]
                else:
                    print(f'vlm_planning_api._find_next_planning_node({subgoal_english}) returns None')
                    return None
        self.current_planning_node = found[-1]
        if trial_index == 0 and status == STARTED:
            self.visited_nodes.append(self.current_planning_node)
            if len(self.current_planning_node.children) > 1:
                self.visited_junction_nodes.append(self.current_planning_node)
        return found

    def save_log_rounds(self):
        keys = ['subgoal', 'log_path', 'obs_path', 'result']
        for i, dic in enumerate(self.log_rounds):
            log_path = join(self.log_dir, '..', f'subgoals_{i}.json')
            subgoals = [{keys[j]: subplan[j] for j in range(len(keys))} for subplan in dic['subplans']]
            with open(log_path, 'w') as f:
                json.dump(subgoals, f, indent=3)

    def output_html(self, progress_table=None):
        output_path = join(self.log_dir, 'index.html')
        memory_path = None if self.load_memory is None else self.load_memory.split('/experiments/')[-1]
        output_html(self.log_rounds, output_path, memory_path=memory_path, progress_table=progress_table)


###########################################################


class GPT4PlanningApi(LLAMPApi):

    def __init__(self, open_goal, vlm_kwargs=dict(), **kwargs):
        super(GPT4PlanningApi, self).__init__(open_goal, **kwargs)
        self.llm = GPT4vApi(**vlm_kwargs)

    def parse_lines_into_lists_fn(self, string: str, **kwargs):
        return parse_lines_into_lists_gpt4(string, **kwargs)


class Claude3PlanningApi(LLAMPApi):

    def __init__(self, open_goal, vlm_kwargs=dict(), **kwargs):
        super(Claude3PlanningApi, self).__init__(open_goal, **kwargs)
        self.llm = Claude3Api(**vlm_kwargs)

    def parse_lines_into_lists_fn(self, string: str, **kwargs):
        return parse_lines_into_lists_claude3(string, **kwargs)


###########################################################


def get_observed_objects(world, objects, verbose=True):
    """ objects are pybullet bodies instead of strings """
    def name_from_obj(obj):
        return world.get_english_name(obj.pybullet_name)

    if verbose:
        print('vlm_planning_api.get_observed_objects')

    bodies = {}
    descriptions = []
    for attachment in world.attachments.values():
        if attachment.parent == world.robot:
            continue
        parent_name = name_from_obj(attachment.parent)
        parent_body = attachment.parent.pybullet_name
        child_name = name_from_obj(attachment.child)
        child_body = attachment.child.pybullet_name

        prop = 'on'
        category = SURFACE
        parent_aabb = attachment.parent.aabb()
        child_aabb = attachment.child.aabb()
        if aabb_placed_in_aabb(child_aabb, parent_aabb):
            prop = 'in'
            category = SPACE

        description = f"the {child_name} is {prop} the {parent_name}"
        if child_body in objects or parent_body in objects:
            descriptions.append(description)
            bodies[(parent_body, category)] = parent_name
            bodies[(child_body, MOVABLE)] = child_name
        else:
            if verbose:
                print(f'\tskipping {description} because not in objects')

    joints = world.cat_to_bodies(KNOB)
    joints = [j for j in world.cat_to_bodies(JOINT) if j not in joints] + joints
    for joint in joints:
        obj = world.get_object(joint)
        description = add_joint_status_facts(joint, categories=obj.categories, return_description=True)
        description = name_from_obj(obj) + ' is ' + description
        descriptions.append(description)
        bodies[(obj.pybullet_name, JOINT)] = name_from_obj(obj)

    surfaces = [s for s in world.cat_to_bodies(SURFACE) if s not in bodies]
    for surface in surfaces:
        obj = world.get_object(surface)
        bodies[(obj.pybullet_name, SURFACE)] = name_from_obj(obj)

    if verbose:
        print()
    return bodies, descriptions
