from cogarch_tools.processes.pddlstream_agent import *

from pybullet_tools.logging_utils import print_heading, get_success_rate_string, print_debug, print_green
from pybullet_tools.stream_agent import fix_init_given_goals, make_init_lower_case, print_skeleton

from world_builder.actions import pull_actions, pull_with_link_actions, pick_arrange_actions, nudge_actions
from world_builder.world import State
from world_builder.init_utils import check_subgoal_achieved, get_objects_at_grasp, \
    remove_unnecessary_movable_given_goal

from vlm_tools.visualize.viz_run_utils import media_dir
from vlm_tools.vlm_utils import pseudo_pull_actions, preds_rename, ACTIONS_GROUP, \
    STARTED, ALREADY, SOLVED, FAILED, UNGROUNDED, ALL_LLAMP_AGENT_STATUS, SUCCEED, END, \
    fix_server_path, idx_branch_from_subgoal, fix_experiment_path

VLM_AGENT_CONFIG_ROOT = abspath(join(dirname(__file__), 'configs'))


class LLAMPAgent(PDDLStreamAgent):

    def __init__(self, world, **kwargs):
        super(LLAMPAgent, self).__init__(world, **kwargs)

        self.goal_sequence = None
        self.llamp_api = None

        self.debug_step = None

        self.last_obs_path = None
        self.last_node = None

        self.object_reducer_names = []
        self.object_reducers = []
        self.object_reducer_state = -1
        self.custom_object_reducer = None
        self.custom_object_reducer_name = None
        self.trial_count = defaultdict(list)

        ## loading cache and planning again
        self.facts_to_update_pddlstream_problem = [[], []]
        self.all_log_collisions = {}

        # self.newly_added_commands = []

    def set_pddlstream_problem(self, problem_dict, state):
        super(LLAMPAgent, self).set_pddlstream_problem(problem_dict, state)

        if 'llamp_api' in problem_dict and problem_dict['llamp_api'] is not None:
            self.llamp_api = problem_dict['llamp_api']
        if 'goal_sequence' in problem_dict and problem_dict['goal_sequence'] is not None:
            self.goal_sequence = problem_dict['goal_sequence']
            self.goal_sequence.insert(0, 'start')

        self.pddlstream_problem = self.world.robot.modify_pddl(self.pddlstream_problem, remove_operators=['place'])

    def init_experiment(self, args, domain_modifier=None, object_reducer=None, comparing=False, serve_page=True):
        super(LLAMPAgent, self).init_experiment(args, domain_modifier, object_reducer, comparing)

        if self.llamp_api is not None:
            self.llamp_api.init_log_dir(join(self.exp_dir, 'log'), serve_page=serve_page)
            self.llamp_api.output_html()

            obs_dir = join(self.exp_dir, 'log', media_dir)
            self.llamp_api.new_session(image_dir=obs_dir, cache_dir=self.exp_dir)

            state_dir = join(self.exp_dir, 'states')
            os.makedirs(state_dir)

            if self.llamp_api.agent_state_path is not None:
                self = self.load_agent_state(self.llamp_api.agent_state_path)
                self._modify_problem_for_debug()

        # ## LLAMP debugging
        # if hasattr(args, 'debug_step'):
        #     self.debug_step = args.debug_step

        return self

    def save_agent_state(self):
        custom_object_reducer = self.custom_object_reducer
        self.custom_object_reducer = None

        super().save_agent_state()

        self.custom_object_reducer = custom_object_reducer

    def load_agent_state(self, agent_state_path):

        agent_state_path = fix_experiment_path(agent_state_path)

        ## for indexing using -1 in given path
        if '-1.pkl' in agent_state_path:
            agent_state_dir = dirname(agent_state_path)
            indices = [f for f in listdir(agent_state_dir) if f.startswith('commands_')]
            indices = [eval(f.replace('commands_', '').replace('.pkl', '')) for f in indices]
            index = max(indices)
            agent_state_path = agent_state_path.replace('-1.pkl', f'{index}.pkl')

        world = self.world
        planning_mode = self.llamp_api.planning_mode

        self = super().load_agent_state(agent_state_path)

        world.remove_all_object_labels_in_pybullet()

        ## move over all media and log files
        for key in ['content', 'media']:
            old_log_dir = fix_server_path(self.llamp_api.log_dir)
            old_log_dir = fix_experiment_path(old_log_dir)
            copy_all_files_to(join(old_log_dir, key), join(self.exp_dir, 'log', key))

        self.llamp_api.world = self.world
        self.llamp_api.log_dir = join(self.exp_dir, 'log')
        self.llamp_api.obs_dir = join(self.llamp_api.log_dir, media_dir)
        self.llamp_api.planning_tree_path = join(self.llamp_api.obs_dir, 'planning_tree.png')
        self.llamp_api.agent_state_path = agent_state_path
        self.llamp_api.planning_mode = planning_mode
        self.llamp_api.llm.image_dir = join(self.llamp_api.log_dir, media_dir)
        self.llamp_api.llm.cache_dir = self.llamp_api.log_dir

        self.llamp_api.agent_memory['load_agent_state'] = agent_state_path
        self.llamp_api.save_agent_memory()

        ## allows trying again
        # for k, v in self.trial_count.items():
        #     if len(v) > 0 and v[-1] != SOLVED:
        #         self.trial_count[k] = []
        # self.object_reducer_state = 0

        ## for replay
        self.save_commands()
        lisdf_path = join(agent_state_path.split('/states/')[0], 'scene.lisdf')
        if isfile(lisdf_path):
            shutil.copy(lisdf_path, join(self.exp_dir, 'scene.lisdf'))
        return self

    def _init_object_reducer(self, args, object_reducer, exp_name, append_object_reducer_name=False):
        """ assumes multiple object reducers are separated by ';' """
        if hasattr(args, 'object_reducer'):
            object_reducer = args.object_reducer
        if object_reducer is not None:
            if append_object_reducer_name:
                exp_name += '_' + object_reducer.replace(';', '_')
            self.object_reducer_state = 0
        self.object_reducer_names = object_reducer.split(';') if ';' in object_reducer else [object_reducer]
        self.object_reducers = [initialize_object_reducer(o) for o in self.object_reducer_names]
        return exp_name

    def replan(self, observation, smaller_world=None, **kwargs):
        """ make new plans given a pddlstream_probelm """

        if self.llamp_api is not None:
            self._replan_preprocess(observation)
            self._init_env_execution()
        debug_just_fail = False

        # set_renderer(True)
        # wait_if_gui()
        # debug_just_fail = True

        pddlstream_problem = self._heuristic_reduce_pddlstream_problem()
        result = super(LLAMPAgent, self).replan(observation, pddlstream_problem=pddlstream_problem,
                                                debug_just_fail=debug_just_fail, **kwargs)

        ## move the txt_file.txt to log directory
        if self.llamp_api is not None:
            if self.llamp_api.planning_mode in ACTIONS_GROUP:
                self.time_log[-1]['goal'] = self.goal_sequence[0]
            status = UNGROUNDED if result is not None and isinstance(result, str) and result == UNGROUNDED else None
            self._replan_postprocess(smaller_world=smaller_world, status=status)

        return self.plan

    def _replan_preprocess(self, observation):

        ## before each planning run, log the current observation and subgoal
        assert len(self.world.cameras) > 0, "Please use world.set_camera_points() or world.add_camera() to add"
        obs_path = self.llamp_api.log_obs_image(self.world.cameras)
        object_reducer_name = self._get_object_reducer_name()
        subgoal = self.goal_sequence[0] ## self.pddlstream_problem.goal[1]
        self.llamp_api.log_subgoal(subgoal, obs_path, STARTED, suffix=f";{object_reducer_name}",
                                   trial_index=self.object_reducer_state)
        self.llamp_api.output_html(self._get_progress_table())
        self.last_obs_path = obs_path

        # ## change pddlstream stream arguments such as debug and visualization
        # if self.debug_step != -1 and self.problem_count != self.debug_step:
        #     self.pddlstream_problem = self._modify_stream_map(observation.state)

        ## change pddlstream planner args such as skeleton and subgoals
        if self.llamp_api.planning_mode in ACTIONS_GROUP:
            from pigi_tools.feasibility_checkers import ChooseSkeleton
            skeleton = self.get_skeleton(subgoal)
            self.pddlstream_kwargs.update({
                'skeleton': skeleton,
                'fc': ChooseSkeleton(self.world, {}, skeleton),
            })
            # to_del = reduce_init_given_skeleton(self.pddlstream_problem.init, skeleton)
            # if len(to_del) > 0:
            #     self.pddlstream_problem = self._modify_init(to_del=to_del)

    def _replan_postprocess(self, goal=None, status=None, smaller_world=None):
        """ after each planning run, log the planning result """
        title = '[llamp_agent._replan_postprocess]\t'
        if goal is None:
            goal = self.goal_sequence[0]  ## self.pddlstream_problem.goal[1]
        if status is None:
            status = FAILED if self.plan is None else SOLVED
        assert status in ALL_LLAMP_AGENT_STATUS

        object_reducer_name = self._get_object_reducer_name()
        obs_path = self.last_obs_path
        if obs_path is None:  ## when the first subgoal already fulfilled
            obs_path = self.llamp_api.log_obs_image(self.world.cameras)
        self.last_node = self.llamp_api.log_subgoal(goal, obs_path, status, suffix=f";{object_reducer_name}",
                                                    trial_index=self.object_reducer_state)
        self.trial_count[self.last_node].append(status)

        log, objects_to_add = self._get_log_collisions(smaller_world)
        self.all_log_collisions[(self.last_node, self._get_object_reducer_name())] = log

        if len(self.time_log) > 0:
            result_dict = {
                'status': status, 'object_reducer': object_reducer_name, 'last_node': self.last_node,
                'agent_state': basename(self.get_state_file_path(key='agent_state'))
            }
            if self.llamp_api.planning_mode in ACTIONS_GROUP:
                result_dict['actual_goal'] = self.get_action_goal([goal])
            self._update_last_time_log_dict(result_dict)

            self.save_stats(final=False, failed_time=True)
            self.llamp_api.output_html(self._get_progress_table())

        if (self.plan and status != END) or status in [ALREADY, UNGROUNDED]:
            self.problem_count += 1
            # self.save_commands(join(self.exp_dir, 'commands.pkl'))

        # self.save_agent_state()

        ## save the collisions that resulted in failure
        if not self.plan and status not in [END, ALREADY, UNGROUNDED]:
            self._modify_object_reducer_based_on_log_collisions(log, objects_to_add)
        else:
            self._reset_custom_object_reducer()

        self.world.robot.reset_log_collisions()
        if smaller_world is not None:
            smaller_world.robot.reset_log_collisions()

        ## otherwise door attachment won't work in action execution
        if smaller_world is not None and len(smaller_world.robot.ROBOT_CONF_TO_OBJECT_CONF) > 0:
            self.world.robot.ROBOT_CONF_TO_OBJECT_CONF = smaller_world.robot.ROBOT_CONF_TO_OBJECT_CONF

    def _update_facts_to_update_pddlstream_problem(self, added=[], deled=[]):
        title = 'facts_to_update_pddlstream_problem.added'
        added_here = [a for a in added if a not in self.facts_to_update_pddlstream_problem[0]]
        deled_here = [a for a in deled if a not in self.facts_to_update_pddlstream_problem[1]]
        print_lists([(added_here, f'{title}.added'), (deled_here, f'{title}.deled')])

        self.facts_to_update_pddlstream_problem[0] += added_here
        self.facts_to_update_pddlstream_problem[1] += deled_here
        title = 'facts_to_update_pddlstream_problem.now'
        print_lists([(self.facts_to_update_pddlstream_problem[0], f'{title}.added'),
                     (self.facts_to_update_pddlstream_problem[1], f'{title}.deled')])

    def _execution_postprocess(self, observation, verbose=True):
        facts = make_init_lower_case(observation.facts)
        facts_from_observation = facts
        title = f'llamp.sequential_policy(problem_count={self.problem_count})'

        if self.last_plan_state is not None:

            ## self.state_facts is updated by env.step(action) while self.last_plan_state = self.pddlstream_problem.init
            ## both from the smaller planning world, thus the differences are added to the current set of all fact
            tag = f'{title}(small problem)'
            added_sm, deled_sm = summarize_state_changes(self.state_facts, self.last_plan_state, verbose=False)
            added_sm = filter_dynamic_facts(added_sm)
            deled_sm = filter_dynamic_facts(deled_sm)
            if verbose:
                print_lists([(added_sm, f'{tag}.added'), (deled_sm, f'{tag}.deled')])
            self._update_facts_to_update_pddlstream_problem(added_sm, deled_sm)

            # added_lg, deled_lg = process_facts_to_add_del(self.state_facts, facts, verbose=verbose,
            #                                               title=f'{title}(differ from observed facts)')
            # facts = update_facts(facts, added_sm + added_lg , deled_sm + deled_lg)
            # self._update_facts_to_update_pddlstream_problem(added_lg, deled_lg)

            ## so that it loads the state before the next planning run
            self.save_agent_state()

        ## loaded facts to be added
        # if self.llamp_api.agent_state_path is not None:
        added, deled = self.facts_to_update_pddlstream_problem
        facts = update_facts(facts, added=added, deled=deled)

        summarize_state_changes(facts, facts_from_observation, verbose=True, title=f'{title}(whole problem)')

        return facts

    def _get_object_reducer(self):
        if self.custom_object_reducer is not None:
            return self.custom_object_reducer
        object_reducer_state = self.object_reducer_state
        if object_reducer_state >= 3:
            object_reducer_state = 2
        return self.object_reducers[object_reducer_state] if self.object_reducer_state >= 0 else None

    def _get_object_reducer_name(self):
        if self.custom_object_reducer_name is not None:
            names_of_added = [self.world.body_to_name[str(k)] for k in self.custom_object_reducer_name if str(k) in self.world.body_to_name]
            custom_object_reducer_name = 'custom_add=' + '+'.join(names_of_added)
            return custom_object_reducer_name
        return self.object_reducer_names[self.object_reducer_state] if self.object_reducer_state >= 0 else None

    def policy(self, observation):

        observation.assign()
        self.world.attachments = observation.state.attachments

        action = self.process_plan(observation)
        if action is not None:
            self.record_command(action)
            return action

        """ if no more action to execute, check success or replan """
        while not self.plan:

            next_goal = None
            smaller_world = None
            facts = self._execution_postprocess(observation)

            seq_planning_mode = self.goal_sequence is not None and len(self.goal_sequence) > 1

            ## self.object_reducer_state == len(self.object_reducer_names) indicates loading back to the state right before backtracking
            if self.object_reducer_state < len(self.object_reducer_names):
                if seq_planning_mode:

                    self.goal_sequence.pop(0)
                    next_goal = self.goal_sequence[0]
                    while self._check_subgoal_achieved(facts, next_goal):

                        ## the last goal is already achieved
                        self.goal_sequence.pop(0)

                        ## a quick wrapper that skips running a planner
                        # self.llamp_api.log_subgoal(next_goal, self.last_obs_path, STARTED)
                        self._record_skipped_time_log(next_goal, status=ALREADY)
                        self._replan_postprocess(next_goal, status=ALREADY)

                        if len(self.goal_sequence) == 0:
                            seq_planning_mode = False
                            break
                        next_goal = self.goal_sequence[0]

                    ## ready for replanning
                    if seq_planning_mode:
                        smaller_world = self._update_pddlstream_problem(facts, goals=[next_goal])

                if not seq_planning_mode and self.goal_achieved(observation):
                    ## add the last observation and action trace to the html
                    obs_path = self.llamp_api.log_obs_image(self.world.cameras)
                    self.last_obs_path = obs_path
                    self._replan_postprocess(END, status=STARTED)
                    return None

                self.replan(observation, smaller_world=smaller_world)
                # self.plan = None  ## TODO debug

            ## planning failed
            if not self.plan:

                ## backtrack planning tree to use other subgoals
                if seq_planning_mode:
                    # print('!!! self.trial_count[self.last_node] ' + str(self.trial_count[self.last_node]))
                    try_again = len(self.object_reducers) > 1 and self.last_node is not None and \
                                len(self.trial_count[self.last_node]) < len(self.object_reducers)
                    # try_again = False  ## TODO debug
                    jump_to_reprompt = 'reprompt' in self.llamp_api.planning_mode and self.last_node is not None and \
                                       UNGROUNDED in self.trial_count[self.last_node]

                    ## self.object_reducer_state == len(self.object_reducer_names) indicates loading back to the state right before backtracking
                    if self.object_reducer_state < len(self.object_reducer_names):
                        self.object_reducer_state += 1

                    if try_again and not jump_to_reprompt:
                        self.goal_sequence.insert(0, next_goal)
                        self.llamp_api.current_planning_node = self.llamp_api.current_planning_node.parent
                    else:
                        failed_but_continue_branch = not 'reprompt' in self.llamp_api.planning_mode

                        ## makes reloading agent state and debugging easier
                        self.save_agent_state()
                        self.save_commands()

                        ## making args for query after failure
                        kwargs = dict()
                        if not failed_but_continue_branch and not jump_to_reprompt:
                            # previous_goals, failed_count = self._get_succeeded_subgoals()
                            collision_bodies = self._get_recent_collision_bodies_english()
                            holding_objects = [(f[1], self.world.get_english_name(f[2])) for f in
                                               self.pddlstream_problem.init if f[0] == 'atgrasp']
                            kwargs = dict(
                                # previous_goals=previous_goals,
                                collision_bodies=collision_bodies,
                                holding_objects=holding_objects
                            )
                        status = self.llamp_api.backtrack_planning_tree(
                            failed_but_continue_branch=failed_but_continue_branch,
                            jump_to_reprompt=jump_to_reprompt, **kwargs
                        )
                        self.problem_count += 1
                        self.object_reducer_state = 0
                        self._reset_custom_object_reducer()
                        if status in [SUCCEED, FAILED]:
                            object_reducer_name = self._get_object_reducer_name()
                            self._update_last_time_log_dict({
                                'status': status, 'object_reducer': object_reducer_name, 'last_node': self.last_node
                            })
                            solved = (status == SUCCEED)
                            if not solved and failed_but_continue_branch:
                                return self.policy(observation)
                            self.save_stats(solved=solved, final=solved, failed_time=True, save_csv=True)
                            return None

                        ## after replanning
                        else:
                            self.goal_sequence = status
                            self.trial_count = defaultdict(list)
                else:
                    break
            else:
                self.object_reducer_state = 0
                self.llamp_api.succeeded_nodes.append(next_goal)

        return self.process_plan(observation)

    def _update_last_time_log_dict(self, result_update):
        ## the last row may be summary
        if 'num_success' in self.time_log[-1] and 'num_success' not in self.time_log[-2]:
            self.time_log[-2].update(result_update)
        else:
            self.time_log[-1].update(result_update)

    ## ----------------------------------------------------------------------------------

    def _get_problem_count_suffix(self):
        return '' if self.object_reducer_state == 0 else 'abcr'[self.object_reducer_state]

    def _check_subgoal_achieved(self, facts, next_goal):
        return check_subgoal_achieved(facts, next_goal, self.world)

    def _update_pddlstream_problem(self, init, goals, reduce_objects=True):
        from pybullet_tools.logging_utils import myprint as print

        title = '[llamp_agent._update_pddlstream_problem]'
        print_heading(f'{title} problem_count = {self.problem_count}')

        """ create a version of the world with less planning objects """
        self.world.remove_all_object_labels_in_pybullet()

        cached_attributes = self.remove_unpickleble_attributes()
        world = copy.deepcopy(self.world)
        world.recover_unpickleble_attributes(cached_attributes)
        self.recover_unpickleble_attributes(cached_attributes)

        ## debug log_collisions
        world.summarize_collisions(title=f'{title} world.collisions')
        self.world.summarize_collisions(title=f'{title} self.world.collisions')

        """ make init """
        ## loaded facts to be added
        # if self.llamp_api.agent_state_path is not None:
        added, deled = self.facts_to_update_pddlstream_problem
        init = update_facts(init, added=added, deled=deled)

        ## fix for domain
        goals = self.fix_goals(goals)
        init = fix_init_given_goals(goals, init)
        init, exceptions = fix_init(init, goals, self.world)

        ## reduce world
        world.name = 'reduced_world'
        exceptions += self.custom_object_reducer_name if self.custom_object_reducer_name is not None else []
        world.remove_bodies_from_planning(goals=goals, exceptions=exceptions)
        if reduce_objects:
            object_reducer = self._get_object_reducer()
            objects = world.objects + world.constants + list(world.robot.arms)
            init = object_reducer(facts=init, objects=objects, goals=goals, world=world)
            # ## sometimes the obstacle is put on the original surface
            # if self.custom_object_reducer is not None:
            #     init = fix_planning_inefficiency(init, world)
        init = remove_unnecessary_movable_given_goal(init, goals, self.world)

        ## if some object is both movable and surface, when not moving, remove the category
        ## TODO: not working for moving braiser then cook
        # world.remove_object_categories_by_init(init)

        """ make goal """
        ## when the subgoal sequence is actually action sequence
        if self.llamp_api.planning_mode in ACTIONS_GROUP:
            goals = self.get_action_goal(goals)

        goal = [AND] + goals

        """ make pddlstream problem """
        domain_pddl, constant_map, stream_pddl, stream_map, _, _ = self.pddlstream_problem
        ## TODO: actually causes problem because world.objects didn't include
        # stream_map = self._get_stream_map(State(world))
        stream_map = self._update_obstacles_in_stream_map(stream_map, world)
        self.pddlstream_problem = PDDLProblem(domain_pddl, constant_map, stream_pddl, stream_map, init, goal)

        """ print out info about planning problem """
        print(f'{SEPARATOR} problem_count = {self.problem_count} {SEPARATOR}')
        if self.llamp_api.planning_mode in ACTIONS_GROUP:
            print_skeleton(self.pddlstream_kwargs['skeleton'])
        world.summarize_all_objects()
        object_reducer_name = self._get_object_reducer_name()
        summarize_facts(init, world, name=f'Facts updated (object_reducer = {object_reducer_name})', print_fn=print)
        print_goal(goal, world=world, print_fn=print)
        world.summarize_body_indices(print_fn=print)
        # world.print_ignored_pairs()

        return world

    def fix_goals(self, goals):
        if isinstance(goals, list) and len(goals) == 1 and isinstance(goals[0], list) and not isinstance(goals[0][0], str):
            goals = goals[0]

        for i, g in enumerate(goals):

            ## fixing some of the misnaming
            if g[0] in preds_rename:
                g[0] = preds_rename[g[0]]

            ## sometimes the arguments don't involve the arm, so favor the empty arm
            if g[0] == 'holding' and len(g) == 2:
                arm = self.get_available_arm(g)
                goals[i] = ['holding', arm, g[-1]]

        return goals

    def _get_stream_map(self, state, collisions=True, teleport=False, debug=False, **kwargs):
        """ may need to change debug info for refinement problems
        movable_collisions=True, motion_collisions=True, pull_collisions=True, base_collisions=True """
        robot = state.world.robot
        return robot.get_stream_map(state, collisions, self.custom_limits, teleport, debug=debug, **kwargs)

    def _modify_stream_map(self, state, **kwargs):
        stream_map = self._get_stream_map(state, **kwargs)
        domain_pddl, constant_map, stream_pddl, _, init, goal = self.pddlstream_problem
        return PDDLProblem(domain_pddl, constant_map, stream_pddl, stream_map, init, goal)

    def _update_obstacles_in_stream_map(self, stream_map, world):
        from pddlstream.language.generator import from_gen_fn
        from pybullet_tools.mobile_streams import get_ik_pull_gen, get_ik_pull_with_link_gen, \
            get_ik_gen_old, get_ik_rel_gen_old
        from pybullet_tools.stream_agent import pull_kwargs, ir_kwargs
        kwargs = dict(collisions=True, teleport=False, custom_limits=self.custom_limits)
        problem = State(world)

        print_green(f'[llamp_agent._update_obstacles_in_stream_map]\t using obstacles {problem.fixed}')
        # world.print_ignored_pairs()

        stream_map.update({
            'inverse-reachability': from_gen_fn(get_ik_gen_old(problem, **ir_kwargs, **kwargs)),
            'inverse-reachability-rel': from_gen_fn(get_ik_rel_gen_old(problem, learned=ir_kwargs['learned'], **kwargs)),
            'inverse-kinematics-pull': from_gen_fn(get_ik_pull_gen(problem, **kwargs, **pull_kwargs)),
            'inverse-kinematics-pull-with-link': from_gen_fn(get_ik_pull_with_link_gen(problem, **kwargs, **pull_kwargs)),
        })
        return stream_map

    def _modify_goal(self, goals=None):
        if goals is None:
            goals = [('Debug1',)]
        domain_pddl, constant_map, stream_pddl, stream_map, init, _ = self.pddlstream_problem
        pddlstream_problem = PDDLProblem(domain_pddl, constant_map, stream_pddl, stream_map, init, [AND] + goals)
        # self.goal_sequence.insert(1, goals)
        return pddlstream_problem

    def _modify_init(self, to_add=[], to_del=[]):
        domain_pddl, constant_map, stream_pddl, stream_map, init, goal = self.pddlstream_problem
        init = update_facts(init, added=to_add, deled=to_del)
        return PDDLProblem(domain_pddl, constant_map, stream_pddl, stream_map, init, goal)

    def _init_env_execution(self, pddlstream_problem=None):
        if self.env_execution is not None:
            return
        if self.llamp_api.planning_mode in ACTIONS_GROUP:
            pddlstream_problem = self._modify_goal()
        temp_dir = self.llamp_api.llm.cache_dir
        super()._init_env_execution(pddlstream_problem, temp_dir=temp_dir)

    ## ----------------------------------------------------------------------------------

    def record_command(self, action):
        self.commands.append(action)
        # self.newly_added_commands.append(action)

    def save_commands(self, commands_path=None, **kwargs):
        if commands_path is None:
            commands_path = join(self.exp_dir, 'commands.pkl')
        if len(self.commands) > 0:
            super().save_commands(commands_path)
            new_commands_path = self.get_state_file_path(key='commands')
            shutil.copy(commands_path, new_commands_path)
        # debug_path = join(self.exp_dir, 'states', f'new_commands_{self.problem_count}.pkl')
        # super().save_commands(debug_path, commands=self.newly_added_commands)
        # self.newly_added_commands = []

    ## ----------------------------------------------------------------------------------

    def get_action_goal(self, actions):
        if self.env_execution is None:
            self._init_env_execution()

        preds_one = ['picked', 'placed', 'graspedhandle', 'nudgeddoor',
                     'grasp_pull_ungrasp_handle', 'grasp_pull_ungrasp_handle_with_link']
        preds_two = ['sprinkledto', 'stacked']

        action = actions[0]
        action_name = action[0]
        print(f'[llamp_agent.get_action_goal] action = {action}')
        if 'open' in action:
            print('debug get_action_goal')
        if action_name in pseudo_pull_actions:
            ## action_name = pull_actions[-1]  ## _pull_decomposed.pddl
            joint = action[1]
            categories = self.world.body_to_object(joint).categories
            action_name = 'grasp_pull_ungrasp_handle' ## pull_actions[-1]
            if 'drawer' in categories:
                action_name = 'grasp_pull_ungrasp_handle_with_link'
            if 'movable' in categories:
                action_name = 'pick'
        operator = self.env_execution.domain.operators[action_name]

        effects = []
        for l in operator.effects.literals:

            ## one-arity effects such as (picked ?o)
            if l.predicate.name in preds_one:
                effects += [(l.predicate.name, action[1])]

            if l.predicate.name in preds_two:
                effects += [(l.predicate.name, action[1], action[2])]

        if len(effects) == 0:
            for l in operator.effects.literals:

                ## zero-arity effects such as (canmove)
                if l.predicate.arity == 0 and l.predicate.name != 'increase':
                    effects += [self.env_execution.from_literal(l)]

        return list(set(effects))

    def get_skeleton(self, subgoal):
        arm = self.get_available_arm(subgoal)
        action_name = subgoal[0]
        action_args = subgoal[1:]
        skeleton = [[action_name, arm] + action_args]

        if action_name in pseudo_pull_actions:
            init = self.pddlstream_problem.init
            joint = subgoal[1]
            if ('unattachedjoint', joint) in init:
                skeleton = [(a, arm, subgoal[1]) for a in pull_actions]
            else:
                skeleton = [(a, arm, subgoal[1]) for a in pull_with_link_actions]

        return skeleton

    ## ------------------------------------------------------------------------------------

    def get_available_arm(self, subgoal):
        arms_used = [f[1] for f in self.state_facts if f[0].startswith('at') and f[0].endswith('grasp')]
        if subgoal[0] in ['arrange']:  ## has to be the same arm
            return arms_used[0]
        arms_left = [a for a in self.robot.arms if a not in arms_used]
        if len(arms_left) == 0:
            print('llamp_agent.get_available_arm: no available arm')
            return self.robot.arms[0]
        return arms_left[0]

    def _get_recent_collision_bodies_english(self):
        collision_counts = defaultdict(int)
        for k, v in self.all_log_collisions.items():
            if self.last_node == k[0]:
                for body, count in v.items():
                    collision_counts[body] += count

        collisions = []
        if len(collision_counts) > 0:
            collision_counts = {self.world.get_english_name(k): v for k, v in
                          sorted(collision_counts.items(), key=lambda item: item[1], reverse=True)}
            max_count = list(collision_counts.values())[0]
            collisions = [k for k, v in collision_counts.items() if v >= max_count * 0.3]
        return collisions

    def _get_log_collisions(self, smaller_world=None):
        title = '_get_log_collisions'
        world = smaller_world if smaller_world is not None else self.world
        world_name = 'smaller_world' if smaller_world is not None else 'self.world'
        log = world.summarize_collisions(title=f'{title} {world_name}')
        if smaller_world is not None:
            ## TODO: sometimes smaller_world is used for planning while sometimes self.world
            log.update(self.world.summarize_collisions(title=f'{title} self.world'))
        objects_to_add = [o for o in log.keys() if o not in world.BODY_TO_OBJECT]
        return log, objects_to_add

    def _modify_object_reducer_based_on_log_collisions(self, log, objects_to_add):
        """ after failure, change the object reducer of the next run """

        if len(objects_to_add) > 0:
            from leap_tools.object_reducers import reduce_facts_given_objects

            ## ignore those accidental or trivial collisions (e.g.g braiser lid compared to braiser)
            if len(objects_to_add) > 1:
                max_occurrence = max(list(log.values())[0] // 3, 1)
                try_add_objects = list([k for k, v in log.items() if v > max_occurrence])
                if len(try_add_objects) > 0:
                    add_objects = try_add_objects

            self.custom_object_reducer_name = objects_to_add

            ## if add_objects included movable objects, also add surfaces to potentially move them to
            add_surfaces = []
            added_movables = [o for o in self.world.cat_to_bodies('movable') if o in objects_to_add]
            if len(added_movables) > 0:
                from leap_tools.heuristic_utils import add_surfaces_given_obstacles
                add_surfaces = add_surfaces_given_obstacles(self.world, added_movables, title='custom_object_reducer\t')

            def custom_object_reducer(facts, objects, **kwargs):
                return reduce_facts_given_objects(facts, objects+objects_to_add+add_surfaces, **kwargs)

            self.custom_object_reducer = custom_object_reducer
            print(f'self.custom_object_reducer_name: {self._get_object_reducer_name()}')

    ## ========================================================================================

    def _reset_custom_object_reducer(self):
        self.custom_object_reducer = None
        self.custom_object_reducer_name = None

    def _modify_problem_for_debug(self):
        name_to_body = self.world.name_to_body
        testcase = 12

        if testcase == 12:
            self.pddlstream_kwargs.update({
                'evaluation_time': 10,
                'max_plans': 1,
                'max_evaluation_plans': 1
            })

        if testcase == 11:
            self.pddlstream_kwargs.update({
                'evaluation_time': 30,
                'max_plans': 12,
                'max_evaluation_plans': 12
            })

        if testcase == 10:
            from world_builder.loaders_nvidia_kitchen import fix_braiser_orientation
            fix_braiser_orientation(self.world)

        if testcase == 9:
            if not hasattr(self, 'problem_count_suffix'):
                self.problem_count_suffix = ''

        if testcase == 7:
            self.pddlstream_kwargs['visualization'] = True
            self.pddlstream_kwargs['stream_planning_timeout'] = 60
            print('stream_planning_timeout = 60')

        if testcase == 6:
            self.llamp_api.suffix = ''

        if testcase == 5:
            self.llamp_api.succeeded_nodes = []

        ## sprinkling
        if testcase == 3:
            adding = [('Sprinkler', name_to_body(k)) for k in ['salt-shaker', 'pepper-shaker']]
            self._update_facts_to_update_pddlstream_problem(added=adding)

        ## the attribute is not in previous saved worlds
        if testcase == 1:
            self.all_log_collisions = {}
            self.custom_object_reducer = None
            self.custom_object_reducer_name = None

        ## testing taking things from the cabinet
        if testcase in [0, 2, 4, 8]:
            arm = 'left'
            door = name_to_body('chewie_door_left_joint')
            salter = name_to_body('salt-shaker')

            skeleton = []
            if testcase == 0:
                skeleton = [(k, arm, door) for k in pull_actions]
                # skeleton += [(k, arm, salter) for k in pick_arrange_actions]

                goals = [('OpenedJoint', door)]
                self.pddlstream_problem = self._modify_goal(goals)
                self.goal_sequence.insert(1, goals[0])

            if testcase == 2:
                skeleton = [(k, arm, door) for k in pull_actions]
                skeleton += [(k, arm, door) for k in nudge_actions]
                skeleton += [(k, arm, salter) for k in pick_arrange_actions]

            if testcase == 4:
                braiser = name_to_body('braiserbody')
                skeleton = [(k, arm, door) for k in pull_actions]
                skeleton += [('pick', arm, salter), ('sprinkle', arm, salter)]

            if testcase == 8:
                braiserlid = name_to_body('braiserlid')
                counter = name_to_body('indigo_tmp')
                goals = [('On', braiserlid, counter)]
                # goals = [['On', braiserlid, counter], ['Pulled', door]]
                self.pddlstream_problem = self._modify_goal(goals)
                self.goal_sequence.insert(1, goals)
                # skeleton = [(k, arm, door) for k in pull_actions]
                # skeleton += [('pick', arm, salter), ('sprinkle', arm, salter)]

            self.pddlstream_kwargs['skeleton'] = skeleton

    def _get_succeeded_subgoals(self):
        previous_goals = []
        failed_count = defaultdict(int)
        for g in self.time_log:
            if 'num_success' in g:
                continue
            if g['status'] == FAILED:
                failed_count[str(g['goal_original'])] += 1
                continue
            if len(previous_goals) > 0 and previous_goals[-1] == g['goal_original']:
                continue
            previous_goals.append(g['goal_original'])
        return previous_goals, failed_count

    def _get_current_whole_goal_sequence(self, verbose=True):
        title = '[llamp_agent._get_current_whole_goal_sequence]'

        previous_goals, failed_count = self._get_succeeded_subgoals()

        future_goals = [[g] for g in self.goal_sequence]
        if len(future_goals) > 0 and len(previous_goals) > 0 and future_goals[0] == previous_goals[-1]:
            future_goals = future_goals[1:]
        future_goals_print = copy.deepcopy(future_goals)
        if len(future_goals) > 0:
            if str(future_goals[0]) in failed_count:
                future_goals_print[0] = f"{future_goals_print[0]}\t <- failed {failed_count[str(future_goals[0])]} times"

        if verbose:
            print(f'{title} previous_goals')
            print('\t' + '\n\t'.join([f"{i}\t{g}" for i, g in enumerate(previous_goals)]))
            print(f'{title} future_goals')
            print('\t' + '\n\t'.join([f"{i+len(previous_goals)}\t{g}" for i, g in enumerate(future_goals_print)]))
        return previous_goals + future_goals

    def _get_progress_table(self, include_summary=False, verbose=True):
        header = ['idx', 'goal', 'task_idx', 'status', 'plan_len', 'planning_time',
                  'planning_objects', 'object_reducer', 'planning_node', 'agent_state']
        whole_goal_sequence = self._get_current_whole_goal_sequence(verbose=verbose)
        state_dir = join(self.exp_dir, 'states')
        loaded_state_dir = None if self.llamp_api.agent_state_path is None else \
            abspath(dirname(self.llamp_api.agent_state_path))
        rows, summary = get_progress_summary_from_time_log(self.time_log, whole_goal_sequence,
                                                           state_dir, loaded_state_dir)
        table = [header] + rows
        if include_summary:
            return table, summary
        return table

    def _record_skipped_time_log(self, goal, **kwargs):
        super()._record_skipped_time_log(goal, last_node=self.last_node, **kwargs)

    def save_time_log(self, csv_name, final=False, solved=True, failed_time=False):
        """ compare the planning time and plan length across runs """
        from tabulate import tabulate

        table, summary = self._get_progress_table(include_summary=True)

        self.time_log = [l for l in self.time_log if 'num_success' not in l]
        self.time_log.append(summary)

        print(f'[llamp_agent.save_time_log]')
        csv_name = join(self.exp_dir, basename(csv_name))
        with open(csv_name, mode='w') as csv_file:
            writer = csv.writer(csv_file)
            writer.writerows(table)

        print(tabulate(table[1:], headers=table[0], tablefmt='psql'))

        return summary['total_planning_time']

# def process_facts_to_add_del(facts_sm, facts, verbose=False, title='process_facts_to_add_del'):
#     added_lg, deled_lg = summarize_state_changes(facts_sm, facts, verbose=False)
#
#     added_atgrasp = find_facts_by_pair(added_lg, 'atgrasp', 'grasp')
#     deled_atgrasp = []  ## this unfortunately will always be empty
#     # deled_lg = deled + [f for f in deled_lg if f[0] in ['handempty']]  ## TODO: hands are always empty in facts?
#
#     added_lg = added_atgrasp + filter_dynamic_facts(added_lg)
#     deled_lg = deled_atgrasp ## + filter_dynamic_facts(deled_lg)
#
#     if verbose:
#         print_lists([(added_lg, f'{title}.added'), (deled_lg, f'{title}.deled')])
#
#     return added_lg, deled_lg


def fix_init(init, goals, world):
    title = '[llamp_agent.fix_init]\t'
    for k in ["CanMove", "CanUngrasp"]:
        if (k,) not in init and (k.lower(),) not in init:
            init += [(k,)]

    ## in principle should be the job of Observation.get_facts()
    arms_at_grasp = [(f[1], f[2]) for f in init if f[0].lower() == 'atgrasp']
    for a, o in arms_at_grasp:
        for k in ['handempty', 'canpull']:
            empty = (k, a)
            if empty in init:
                init.remove(empty)
                atpose = [f for f in init if f[0] in ['atpose', 'atrelpose'] and f[1] == o]
                [init.remove(f) for f in atpose]
                print(f'{title} removed {empty} and atpose {atpose}')

    ## include the objects at hand and potential areas to place
    objs_at_grasp = get_objects_at_grasp(init, goals, world)
    return init, objs_at_grasp


def get_progress_summary_from_time_log(time_log, whole_goal_sequence,
                                       state_dir=None, loaded_state_dir=None):
    header = ['idx', 'goal', 'task_idx', 'status', 'plan_len', 'planning_time',
              'planning_objects', 'object_reducer', 'planning_node', 'agent_state']
    rows = []

    ## summarize planning time
    effective_planning = 0
    wasted_planning = 0

    total_num_problems = len(whole_goal_sequence)
    task_failure = False
    problems_failed = defaultdict(int)  ## all planning failure counted for different object reduction method
    problems_succeeded = []
    problems_passed = []
    subgoal_planning_time = []
    subgoal_plan_len = []
    previous_branch = None
    previous_starting_index = 0
    task_idx = 0
    last_status = None
    last_node = None

    continuous_plan_len = None
    continuous_planning_time = None
    num_problems_solved_continuously = None
    continuous_subgoal_plan_len = []
    reprompt_after_subgoals = []
    ungrounded_subgoals = []
    failed_nodes = defaultdict(int)  ## only count that those that result in reprompting or failure

    for i, log in enumerate(time_log):
        if 'goal' not in log or 'last_node' not in log:
            continue
        if log['last_node'] is None:
            print_debug(f'progress stuck {log}')
            return rows, {}

        task_idx_and_branch = idx_branch_from_subgoal(log['last_node'])
        if task_idx_and_branch[1] != previous_branch:
            previous_starting_index = task_idx  ## would have already minused one if the previous one failed
            task_failure = False
            num_problems_solved_continuously = None
            continuous_plan_len = None
            continuous_planning_time = None
            if previous_branch is not None:
                reprompt_after_subgoals.append(last_node)

        last_status = log['status']
        last_node = log['last_node']

        task_idx = int(task_idx_and_branch[0]) + previous_starting_index
        previous_branch = task_idx_and_branch[1]

        num_planning_objects = ''
        if 'objects_by_category' in log:
            data = log['objects_by_category']
            total = sum([len(v) for v in data.values()])
            items = ', '.join(f"{k}: {len(v)}" for k, v in data.items() if len(v) > 0)
            num_planning_objects = f"{total} ({items})"

        goal = copy.deepcopy(log['goal'])
        if 'actual_goal' in log:
            goal = str(goal) + ' -> ' + str(log['actual_goal'])

        title = '[get_progress_table] agent state not found in'
        agent_state = ''
        if 'agent_state' in log:
            if state_dir is not None:
                agent_state_potential = join(state_dir, log['agent_state'])
                if isfile(agent_state_potential):
                    agent_state = log['agent_state']
                else:
                    # print_debug(f'{title} current state dir {agent_state_potential}')
                    if loaded_state_dir is not None:
                        agent_state_potential = join(loaded_state_dir, log['agent_state'])
                        if isfile(agent_state_potential):
                            dir_name = loaded_state_dir.split('/experiments/')[-1]
                            agent_state = join(dir_name, log['agent_state'])
                        # else:
                        #     print_debug(f'{title} loaded state dir {agent_state_potential}')

        object_reducer = log['object_reducer'].replace('counter#1::', '')
        rows.append([i + 1, goal, task_idx, last_status, log['plan_len'], log['planning'],
                     num_planning_objects, object_reducer, log['last_node'], agent_state])

        subgoal_planning_time.append(log['planning'])
        subgoal_plan_len.append(log['plan_len'])

        if last_status in [SOLVED, STARTED]:  ## the last one is marked as STARTED somehow
            problems_succeeded.append(last_node)
            problems_passed.append(log['last_node'])
            effective_planning += log['planning']
            continuous_subgoal_plan_len.append(log['plan_len'])
            if last_node in failed_nodes:
                failed_nodes.pop(last_node)

        if last_status in [ALREADY]:
            problems_passed.append(last_node)

        if last_status in [UNGROUNDED]:
            ungrounded_subgoals.append(last_node)

        if last_status in [FAILED]:
            problems_failed[last_node] += 1
            failed_nodes[last_node] += 1
            wasted_planning += log['planning']
            task_idx -= 1

        if problems_failed[last_node] == 3 and num_problems_solved_continuously is None:
            task_failure = True
            num_problems_solved_continuously = len(problems_passed)
            continuous_plan_len = len(subgoal_plan_len)
            continuous_planning_time = effective_planning

    ## sometimes the experiment is cut off early because the last chance the agent has is an infeasible problem
    if not task_failure and (last_status == FAILED or len(problems_passed) != total_num_problems):
        task_failure = True
        num_problems_solved_continuously = len(problems_passed)
        continuous_plan_len = len(subgoal_plan_len)
        continuous_planning_time = effective_planning

    num_success = len(problems_succeeded)
    problems_failed = [pb for pb, v in problems_failed.items() if pb not in problems_succeeded and v > 0]
    num_problems = len(problems_failed) + num_success

    num_completed_problems = task_idx
    total_planning = sum(subgoal_planning_time)
    total_plan_len = sum(subgoal_plan_len)

    # print(f'[get_progress_summary_from_time_log]\tlen(whole_goal_sequence) == {len(whole_goal_sequence)}')
    if num_problems_solved_continuously is None:
        num_problems_solved_continuously = total_num_problems
        continuous_plan_len = total_plan_len
        continuous_planning_time = total_planning

    # print(f'total_num_problems = {total_num_problems}\t continuously solved {num_problems_solved_continuously}\n\n')
    summary = {
        'continuous_success_string': get_success_rate_string(num_problems_solved_continuously, total_num_problems, roundto=2),
        'task_progress_string': get_success_rate_string(num_completed_problems, total_num_problems, roundto=2),
        'total_problems_solved_string': get_success_rate_string(num_success, num_problems, roundto=2),

        'total_plan_len': total_plan_len,
        'total_planning_time': round(total_planning, 2),

        'effective_planning_time': effective_planning,
        'wasted_planning_time': wasted_planning,
    }
    data = list(summary.values())
    rows.append([''] + data[:-2] +
                [f'{round(data[-2], 2)} (effective time)', f'{round(data[-1], 2)} (wasted time)', '', ''])

    num_reprompts = 0 if previous_branch is None else 'axy'.index(previous_branch)
    summary.update({
        'task_success': int(1-task_failure),
        'num_solved_problems': num_problems,  ## solved or failed
        'num_success': num_success,  ## solved
        'num_problems_solved_continuously': num_problems_solved_continuously,
        'num_completed_problems': num_completed_problems,  ## solved, failed, already, or ungrounded
        'total_num_problems': total_num_problems,  ## all predicted problems

        'continuous_success_rate': num_problems_solved_continuously / total_num_problems if total_num_problems > 0 else 0,
        'task_progress': num_completed_problems / total_num_problems if total_num_problems > 0 else 0,
        'total_problems_solved': num_success / num_problems if num_problems > 0 else 0,

        'whole_goal_sequence': whole_goal_sequence,
        # 'subgoal_planning_time': subgoal_planning_time,
        'subgoal_plan_len': subgoal_plan_len,

        'continuous_plan_len': continuous_plan_len,
        'continuous_planning_time': continuous_planning_time,
        'continuous_subgoal_plan_len': continuous_subgoal_plan_len,

        'ungrounded_subgoals': list(set(ungrounded_subgoals)),
        'reprompt_after_subgoals': reprompt_after_subgoals,
        'num_reprompts': num_reprompts,
        'failed_nodes': dict(failed_nodes)
    })
    return rows, summary


def copy_all_files_to(old_dir, new_dir):
    os.makedirs(new_dir, exist_ok=True)
    for f in listdir(old_dir):
        shutil.copy(join(old_dir, f), join(new_dir, f))
