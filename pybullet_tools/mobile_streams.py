from __future__ import print_function

from pybullet_tools.utils import invert, get_all_links, get_name, set_pose, get_link_pose, is_placement, \
    pairwise_collision, set_joint_positions, get_joint_positions, sample_placement, get_pose, waypoints_from_path, \
    unit_quat, plan_base_motion, plan_joint_motion, base_values_from_pose, pose_from_base_values, \
    uniform_pose_generator, add_fixed_constraint, remove_debug, remove_fixed_constraint, \
    disable_real_time, enable_gravity, joint_controller_hold, get_distance, Point, Euler, set_joint_position, \
    get_min_limit, user_input, step_simulation, get_body_name, get_bodies, BASE_LINK, get_joint_position, \
    add_segments, get_max_limit, link_from_name, BodySaver, get_aabb, interpolate_poses, wait_for_user, \
    plan_direct_joint_motion, has_gui, create_attachment, wait_for_duration, get_extend_fn, set_renderer, \
    get_custom_limits, all_between, remove_body, draw_aabb, GREEN, MAX_DISTANCE

from pybullet_tools.bullet_utils import multiply, has_tracik, visualize_bconf
from pybullet_tools.ikfast.pr2.ik import pr2_inverse_kinematics
from pybullet_tools.ikfast.utils import USE_CURRENT
from pybullet_tools.pr2_primitives import Conf, Commands, create_trajectory, State
from pybullet_tools.pr2_streams import DEFAULT_RESOLUTION
from pybullet_tools.pr2_utils import open_arm, arm_conf, learned_pose_generator
from pybullet_tools.utils import WorldSaver
from pybullet_tools.general_streams import *


def get_ir_sampler(problem, custom_limits={}, max_attempts=40, collisions=True, learned=True, verbose=False):
    robot = problem.robot
    world = problem.world
    obstacles = [o for o in problem.fixed if o not in problem.floors] if collisions else []
    grippers = {arm: problem.get_gripper(arm=arm, visual=False) for arm in robot.arms}
    heading = f'   mobile_streams.get_ir_sampler | '

    def gen_fn(arm, obj, pose, grasp):

        gripper = grippers[arm]
        pose.assign()
        if isinstance(obj, tuple):  ## may be a (body, joint) or a body with a marker
            obj = obj[0]
        if 'pstn' in str(pose): ## isinstance(pose, Position): ## path problem
            pose_value = linkpose_from_position(pose)
        else:
            pose_value = pose.value

        if hasattr(world, 'refine_marker_obstacles'):
            approach_obstacles = problem.world.refine_marker_obstacles(obj, obstacles)
            ## {obst for obst in obstacles if obst != obj}  ##{obst for obst in obstacles if not is_placement(obj, obst)}
            if set(obstacles) != set(approach_obstacles):
                print(f'approach_obstacles = {approach_obstacles}')
        else:
            approach_obstacles = obstacles

        for _ in robot.iterate_approach_path(arm, gripper, pose_value, grasp):
            # if verbose:
            for b in approach_obstacles:
                if pairwise_collision(gripper, b):
                    if verbose:
                        print(f'{heading} in approach, gripper {nice(get_pose(gripper))} collide with {b} {nice(get_pose(b))}')
                    return
                if obj == b: continue

        # gripper_pose = multiply(pose_value, invert(grasp.value)) # w_f_g = w_f_o * (g_f_o)^-1
        tool_from_root = robot.get_tool_from_root(arm)
        gripper_pose = multiply(robot.get_grasp_pose(pose_value, grasp.value, arm, body=grasp.body), invert(tool_from_root))

        default_conf = grasp.carry
        arm_joints = robot.get_arm_joints(arm)
        base_joints = robot.get_base_joints()
        if learned:
            base_generator = learned_pose_generator(robot, gripper_pose, arm=arm, grasp_type=grasp.grasp_type)
        else:
            base_generator = uniform_pose_generator(robot, gripper_pose)

        lower_limits, upper_limits = get_custom_limits(robot, base_joints, custom_limits)
        aconf = nice(get_joint_positions(robot, arm_joints))
        while True:
            count = 0
            for base_conf in islice(base_generator, max_attempts):
                count += 1
                if not all_between(lower_limits, base_conf, upper_limits):
                    continue

                ## added by YANG for adding torso value
                if robot.use_torso:
                    base_joints = robot.get_base_joints()
                    z = gripper_pose[0][-1]
                    z = random.uniform(z - 0.7, z - 0.3)
                    x, y, yaw = base_conf
                    base_conf = (x, y, z, yaw)

                bq = Conf(robot, base_joints, base_conf)
                pose.assign()
                bq.assign()
                set_joint_positions(robot, arm_joints, default_conf)
                if any(pairwise_collision(robot, b) for b in obstacles + [obj]):
                    continue
                if verbose:
                    print(f'{heading} IR attempt {count} | bconf = {nice(base_conf)}, aconf = {aconf}')

                yield (bq,)
                break
            else:
                yield None
    return gen_fn


## --------------------------------------------------------------------------------


def get_ik_fn_old(problem, custom_limits={}, collisions=True, teleport=False,
                  ACONF=False, verbose=True, visualize=False, resolution=DEFAULT_RESOLUTION):
    robot = problem.robot
    world = problem.world
    obstacles = problem.fixed if collisions else []
    ignored_pairs = world.ignored_pairs
    world_saver = WorldSaver()
    title = 'mobile_streams.get_ik_fn_old:\t'

    def fn(arm, obj, pose, grasp, base_conf, fluents=[]):

        obstacles_here = copy.deepcopy(obstacles)
        ignored_pairs_here = copy.deepcopy(ignored_pairs)

        if fluents:
            attachments = process_motion_fluents(fluents, robot)
            attachments = {a.child: a for a in attachments}
            obstacles_here.extend([p[1] for p in fluents if p[0] in ['atpose', 'atrelpose'] if isinstance(p[1], int)])
        else:
            world_saver.restore()
            attachment = grasp.get_attachment(robot, arm, visualize=False)
            attachments = {attachment.child: attachment}  ## {}  ## TODO: problem with having (body, joint) tuple

        if 'pstn' in str(pose):  ## isinstance(pose, Position):
            pose_value = linkpose_from_position(pose)
        else:
            pose_value = pose.value
        pose.assign()

        return solve_approach_ik(
            arm, obj, pose_value, grasp, base_conf,
            world, robot, custom_limits, obstacles_here, ignored_pairs_here, resolution,
            title=title, ACONF=ACONF, teleport=teleport, verbose=verbose, visualize=visualize
        )
    return fn


def get_ik_rel_fn_old(problem, custom_limits={}, collisions=True, teleport=False,
                      ACONF=False, verbose=True, visualize=False):
    robot = problem.robot
    world = problem.world
    obstacles = problem.fixed if collisions else []
    ignored_pairs = world.ignored_pairs
    world_saver = WorldSaver()
    title = 'mobile_streams.get_ik_rel_fn_old:\t'

    def fn(arm, obj, relpose, obj2, pose2, grasp, base_conf, fluents=[]):

        obstacles_here = copy.deepcopy(obstacles)
        ignored_pairs_here = copy.deepcopy(ignored_pairs)

        if fluents:
            attachments = process_motion_fluents(fluents, robot)
            attachments = {a.child: a for a in attachments}
            obstacles_here.extend([p[1] for p in fluents if p[0] in ['atpose', 'atrelpose'] if isinstance(p[1], int)])
        else:
            world_saver.restore()
            attachment = grasp.get_attachment(robot, arm, visualize=False)
            attachments = {attachment.child: attachment}  ## {}  ## TODO: problem with having (body, joint) tuple

        # pose2.assign()
        relpose.assign()
        pose_value = multiply(pose2.value, relpose.value)

        return solve_approach_ik(
            arm, obj, pose_value, grasp, base_conf,
            world, robot, custom_limits, obstacles_here, ignored_pairs_here,
            title, ACONF=ACONF, teleport=teleport, verbose=verbose, visualize=visualize
        )
    return fn


## --------------------------------------------------------------------


def get_ik_gen_old(problem, max_attempts=30, collisions=True, learned=True, teleport=False, ir_only=False,
                   soft_failures=False, verbose=False, visualize=False, ACONF=False, **kwargs):
    """ given grasp of target object at relative pose rp with regard to supporter at p2, return base conf and arm traj """
    ir_max_attempts = 40
    ## not using this if tracik compiled
    ir_sampler = get_ir_sampler(problem, collisions=collisions, learned=learned,
                                max_attempts=ir_max_attempts, verbose=verbose, **kwargs)
    ik_fn = get_ik_fn_old(problem, collisions=collisions, teleport=teleport, verbose=False, ACONF=ACONF, **kwargs)
    robot = problem.robot
    world = problem.world
    obstacles = problem.fixed if collisions else []
    heading = 'mobile_streams.get_ik_rel_gen | '

    def gen(a, o, p, g, context=None):
        process_ik_context(context)

        """ check if hand pose is in collision """
        p.assign()
        if 'pstn' in str(p):
            pose_value = linkpose_from_position(p)
        else:
            pose_value = p.value

        inputs = a, o, p, g
        return sample_bconf(
            world, robot, inputs, pose_value, obstacles, heading, ir_sampler=ir_sampler, ik_fn=ik_fn,
            verbose=verbose, visualize=visualize, soft_failures=soft_failures, learned=learned,
            ir_max_attempts=ir_max_attempts, max_attempts=max_attempts, ir_only=ir_only)

    return gen


def get_ik_rel_gen_old(problem, max_attempts=30, collisions=True, learned=True, teleport=False, ir_only=False,
                       soft_failures=False, verbose=False, visualize=False, ACONF=False, **kwargs):
    """ given grasp of target object at relative pose rp with regard to supporter at p2, return base conf and arm traj """
    ir_max_attempts = 40
    ## not using this if tracik compiled
    ir_sampler = get_ir_sampler(problem, collisions=collisions, learned=learned,
                                max_attempts=ir_max_attempts, verbose=verbose, **kwargs)
    ik_fn = get_ik_fn_old(problem, collisions=collisions, teleport=teleport, verbose=False, ACONF=ACONF, **kwargs)
    robot = problem.robot
    world = problem.world
    obstacles = problem.fixed if collisions else []
    heading = 'mobile_streams.get_ik_rel_gen | '

    def gen(a, o1, rp1, o2, p2, g, context=None):
        process_ik_context(context)

        p2.assign()
        rp1.assign()
        pose_value = multiply(p2.value, rp1.value)

        inputs = a, o1, rp1, o2, p2, g
        return sample_bconf(world, robot, inputs, pose_value, obstacles, heading, ir_sampler=ir_sampler, ik_fn=ik_fn,
                            verbose=verbose, visualize=visualize, soft_failures=soft_failures, learned=learned,
                            ir_max_attempts=ir_max_attempts, max_attempts=max_attempts, ir_only=ir_only)

    return gen


######################################################################################################


def process_ik_context(context, verbose=False):
    if context is None:
        return
    from pddlstream.language.object import OptimisticObject
    current_stream = context[0]
    if verbose:
        print('Stream:', current_stream)
    for i, stream in enumerate(context[1:]):
        if stream.output_objects:
            continue
        if not set(current_stream.output_objects) & set(stream.input_objects):
            continue
        if any(isinstance(obj, OptimisticObject) for obj in set(stream.input_objects) - set(current_stream.output_objects)):
            continue
        if verbose:
            print('{}/{}) {}'.format(i, len(context), stream))
        inputs = stream.instance.get_input_values()
        if stream.name in ['inverse-kinematics', 'sample-pose', 'sample-pose-inside', 'sample-grasp',
                           'test-cfree-pose-pose', 'test-cfree-approach-pose', 'plan-base-motion']:
            pass
        elif stream.name == 'test-cfree-traj-pose':
            _, o, p = inputs
            p.assign()
        elif stream.name == 'test-cfree-traj-position':
            _, o, p = inputs
            p.assign()
        else:
            raise ValueError(stream.name)


def sample_bconf(world, robot, inputs, pose_value, obstacles, heading,
                 ir_sampler=None, ik_fn=None, ir_only=False, learned=False,
                 ir_max_attempts=40, max_attempts=30, soft_failures=False, verbose=False, visualize=False):
    a, o = inputs[:2]
    g = inputs[-1]
    robot.open_arm(a)
    context_saver = WorldSaver(bodies=[robot, o])
    title = f'sample_bconf({o}, learned=True) | start sampling '

    # set_renderer(enable=False)
    if visualize:
        # set_renderer(enable=True)
        samples = []

    # gripper_grasp = robot.visualize_grasp(pose_value, g.value, arm=a, body=g.body)
    gripper_grasp = robot.set_gripper_pose(pose_value, g.value, arm=a, body=g.body)
    if collided(gripper_grasp, obstacles, articulated=True, world=world):  # w is not None
        # wait_unlocked()
        # robot.remove_gripper(gripper_grasp)
        if verbose:
            print(f'{heading} -------------- grasp {nice(g.value)} is in collision')
        return
    # robot.remove_gripper(gripper_grasp)

    arm_joints = robot.get_arm_joints(a)
    default_conf = g.carry

    ## use domain specific bconf databases
    if learned and world.learned_bconf_list_gen is not None:
        results = world.learned_bconf_list_gen(world, inputs)
        searched = False
        for bq in results:
            searched = True
            ir_outputs = (bq,)
            print('sample_bconf | found saved bconf', bq)
            if ir_only:
                yield ir_outputs
                continue

            ik_outputs = ik_fn(*(inputs + ir_outputs))
            if ik_outputs is None:
                continue
            yield ir_outputs + ik_outputs

        reason = 'beyond saved bconfs' if searched else 'because there arent saved bconfs'
        print(title + reason)

    ## solve IK for all 13 joints
    if robot.use_torso and has_tracik():
        from pybullet_tools.tracik import IKSolver
        tool_from_root = robot.get_tool_from_root(a)
        tool_pose = robot.get_grasp_pose(pose_value, g.value, a, body=g.body)
        gripper_pose = multiply(tool_pose, invert(tool_from_root))

        tool_link = robot.get_tool_link(a)
        ik_solver = IKSolver(robot, tool_link=tool_link, first_joint=None,
                             custom_limits=robot.custom_limits)  ## using all 13 joints

        attempts = 0
        for conf in ik_solver.generate(gripper_pose):  # TODO: islice
            if max_attempts <= attempts:
                if verbose:
                    print(f'sample_bconf failed after {attempts} attempts!')
                # wait_unlocked()
                if soft_failures:
                    attempts = 0
                    yield None
                    context_saver.restore()
                    continue
                else:
                    break
            attempts += 1
            if conf is None:
                continue

            joint_state = dict(zip(ik_solver.joints, conf))

            base_joints = robot.get_base_joints()
            bconf = list(map(joint_state.get, base_joints))
            bq = Conf(robot, base_joints, bconf, joint_state=joint_state)
            bq.assign()

            set_joint_positions(robot, arm_joints, default_conf)
            if collided(robot, obstacles, articulated=True, tag='ik_default_conf',
                        verbose=verbose, world=world):
                # wait_unlocked()
                continue

            ik_solver.set_conf(conf)
            if collided(robot, obstacles, articulated=True, tag='ik_final_conf',
                        visualize=visualize, verbose=verbose, world=world):
                continue

            if visualize:
                samples.append(visualize_bconf(bconf))
                # set_renderer(True)
                # Conf(robot, joints, conf).assign()
                # wait_for_user()

            ir_outputs = (bq,)
            if ir_only:
                if visualize:
                    [remove_body(samp) for samp in samples]
                yield ir_outputs
                continue

            ik_outputs = ik_fn(*(inputs + ir_outputs))
            if ik_outputs is None:
                continue
            if verbose: print('succeed after TracIK solutions:', attempts)

            if visualize:
                [remove_body(samp) for samp in samples]
            yield ir_outputs + ik_outputs
            context_saver.restore()

    ## do ir sampling of x, y, theta, torso, then solve ik for arm
    else:
        ir_generator = ir_sampler(*inputs)
        attempts = 0
        while True:
            if max_attempts <= attempts:
                # print(f'{heading} exceeding max_attempts = {max_attempts}')
                yield None
                # break # TODO(caelan): probably should be break/return

            attempts += 1
            if verbose: print(f'{heading} | attempt {attempts} | inputs = {inputs}')

            try:
                ir_outputs = next(ir_generator)
            except StopIteration:
                if verbose: print('    stopped ir_generator in', attempts, 'attempts')
                print(f'{heading} exceeding ir_generator ir_max_attempts = {ir_max_attempts}')
                return

            if ir_outputs is None:
                continue
            inp = ir_generator.gi_frame.f_locals
            inp = [inp[k] for k in ['pose', 'grasp', 'custom_limits']]
            if verbose:
                print(f'           ir_generator  |  inputs = {inp}  |  ir_outputs = {ir_outputs}')

            if visualize:
                bconf = ir_outputs[0].values
                samples.append(visualize_bconf(bconf))

            if ir_only:
                yield ir_outputs
                continue

            ik_outputs = ik_fn(*(inputs + ir_outputs))
            if ik_outputs is None:
                continue
            if verbose: print('succeed after IK attempts:', attempts)

            if visualize:
                [remove_body(samp) for samp in samples]
            yield ir_outputs + ik_outputs
            return
            # if not p.init:
            #    return


def solve_approach_ik(arm, obj, pose_value, grasp, base_conf,
                      world, robot, custom_limits, obstacles_here, ignored_pairs_here,
                      resolution=DEFAULT_RESOLUTION, max_distance=MAX_DISTANCE, title='solve_approach_ik',
                      ACONF=False, teleport=False, verbose=True, visualize=False):
    attachments = {}

    if isinstance(obj, tuple):  ## may be a (body, joint) or a body with a marker
        body = obj[0]
    else:
        body = obj

    ## TODO: change to world.get_grasp_parent
    addons = [body]
    if hasattr(world, 'BODY_TO_OBJECT') and body in world.BODY_TO_OBJECT and \
            world.BODY_TO_OBJECT[body].grasp_parent is not None:
        addons.append(world.BODY_TO_OBJECT[body].grasp_parent)

    approach_obstacles = obstacles_here
    ignored_pairs_here.extend([[(body, obst), (obst, body)] for obst in obstacles_here if is_placement(body, obst)])
    # approach_obstacles = problem.world.refine_marker_obstacles(obj, approach_obstacles)  ## for steerables

    tool_from_root = robot.get_tool_from_root(arm)
    gripper_pose = multiply(robot.get_grasp_pose(pose_value, grasp.value, body=obj), invert(tool_from_root))
    approach_pose = multiply(robot.get_grasp_pose(pose_value, grasp.approach, body=obj), invert(tool_from_root))

    arm_link = robot.get_tool_link(arm)
    arm_joints = robot.get_arm_joints(arm)

    default_conf = grasp.carry
    # sample_fn = get_sample_fn(robot, arm_joints)
    base_conf.assign()
    robot.open_arm(arm)
    set_joint_positions(robot, arm_joints, default_conf)  # default_conf | sample_fn()

    ## visualize the gripper
    gripper_grasp = None
    if visualize:
        print('grasp_value', nice(grasp.value))
        set_renderer(True)
        gripper_approach = robot.visualize_grasp(pose_value, grasp.approach,
                                                 body=obj, color=RED, new_gripper=True)
        gripper_grasp = robot.visualize_grasp(pose_value, grasp.value,
                                              body=obj, color=GREEN, new_gripper=True)
        set_camera_target_body(gripper_grasp)
        wait_unlocked('solve_approach_ik | visualized the gripper at grasp and approach poses')
        remove_body(gripper_approach)
        remove_body(gripper_grasp)

    if base_conf.joint_state is not None:
        grasp_conf = list(map(base_conf.joint_state.get, arm_joints))
        set_joint_positions(robot, arm_joints, grasp_conf)

        ## visualize the grasp conf
        if visualize:
            set_renderer(True)
            wait_unlocked('solve_approach_ik | visualized the arm')
    else:
        grasp_conf = robot.inverse_kinematics(arm, gripper_pose, obstacles_here, verbose=verbose)

    if (grasp_conf is None) or collided(robot, obstacles_here, articulated=True, world=world, tag=title,
                                        verbose=verbose, ignored_pairs=ignored_pairs_here,
                                        min_num_pts=3):  ## approach_obstacles): # [obj]
        # wait_unlocked()
        if verbose:
            if grasp_conf is not None:
                grasp_conf = nice(grasp_conf)
            print(
                f'{title}Grasp IK failure | {grasp_conf} <- pr2_inverse_kinematics({robot}, {nice(base_conf.values)}, '
                f'{arm}, {nice(gripper_pose[0])}) | pose {pose_value}, grasp {grasp}')
        # if grasp_conf is not None:
        #    print(grasp_conf)
        #    #wait_if_gui()
        if visualize:
            remove_body(gripper_grasp)
        return None
    # elif verbose:
    #     print(f'{title}Grasp IK success | {nice(grasp_conf)} = pr2_inverse_kinematics({robot} at {nice(base_conf.values)}, '
    #           f'{arm}, {nice(gripper_pose[0])}) | pose = {pose}, grasp = {grasp}')

    approach_conf = None
    if has_tracik():
        from pybullet_tools.tracik import IKSolver
        tool_link = robot.get_tool_link(arm)
        ik_solver = IKSolver(robot, tool_link=tool_link, first_joint=arm_joints[0],
                             custom_limits=custom_limits)  # TODO: cache
        approach_conf = ik_solver.solve(approach_pose, seed_conf=grasp_conf)

    if (not has_tracik() or approach_conf is None) and 'pr2' in robot.name.lower():
        # TODO(caelan): sub_inverse_kinematics's clone_body has large overhead
        approach_conf = pr2_inverse_kinematics(robot, arm, approach_pose, custom_limits=custom_limits,
                                               upper_limits=USE_CURRENT, nearby_conf=USE_CURRENT)
        if not has_tracik() and approach_conf is not None:
            print('\n\n FastIK succeeded after TracIK failed\n\n')
        # approach_conf = sub_inverse_kinematics(robot, arm_joints[0], arm_link, approach_pose, custom_limits=custom_limits)

    if (approach_conf is None) or collided(robot, obstacles_here, articulated=True, world=world, tag=title,
                                           verbose=verbose, ignored_pairs=ignored_pairs_here, min_num_pts=3):  ##
        if verbose:
            if approach_conf is not None:
                approach_conf = nice(approach_conf)
            print(f'{title}Approach IK failure', approach_conf)
        # wait_if_gui()
        if visualize:
            remove_body(gripper_grasp)
        return None
    # elif verbose:
    #     print(f'{title}Approach IK success | sub_inverse_kinematics({robot} at {nice(base_conf.values)}, '
    #           f'{arm}, {nice(approach_pose[0])}) | pose = {pose}, grasp = {nice(grasp.approach)} -> {nice(approach_conf)}')

    # ## -------------------------------------------
    # arm_joints = get_arm_joints(robot, 'left')
    # aconf = Conf(robot, arm_joints, get_joint_positions(robot, arm_joints))
    # print(f'@ mobile_streams.get_ik_fn() -> aconf = {aconf} | bconf = {base_conf}')
    # ## -------------------------------------------

    set_joint_positions(robot, arm_joints, approach_conf)
    # approach_conf = get_joint_positions(robot, arm_joints)

    motion_planning_kwargs = dict(attachments=attachments.values(), self_collisions=robot.self_collisions,
                                  use_aabb=True, cache=True, ignored_pairs=ignored_pairs_here,
                                  custom_limits=custom_limits, max_distance=robot.max_distance)

    if teleport:
        path = [default_conf, approach_conf, grasp_conf]
    else:
        resolutions = resolution * np.ones(len(arm_joints))
        if is_top_grasp(robot, arm, body, grasp) or True:
            grasp_path = plan_direct_joint_motion(robot, arm_joints, grasp_conf, obstacles=approach_obstacles,
                                                  resolutions=resolutions / 2., **motion_planning_kwargs)
            if grasp_path is None:
                if verbose: print(f'{title}Grasp path failure')
                if visualize:
                    remove_body(gripper_grasp)
                return None
            dest_conf = approach_conf
        else:
            grasp_path = []
            dest_conf = grasp_conf

        set_joint_positions(robot, arm_joints, default_conf)
        approach_path = plan_joint_motion(robot, arm_joints, dest_conf, obstacles=obstacles_here, resolutions=resolutions,
                                          restarts=2, iterations=25, smooth=0, **motion_planning_kwargs)  # smooth=25
        if approach_path is None:
            if verbose: print(f'{title}\tApproach path failure')
            if visualize:
                remove_body(gripper_grasp)
            return None
        path = approach_path + grasp_path

    mt = create_trajectory(robot, arm_joints, path)
    cmd = Commands(State(attachments=attachments), savers=[BodySaver(robot)], commands=[mt])

    set_joint_positions(robot, arm_joints, default_conf)  # default_conf | sample_fn()

    if visualize:
        remove_body(gripper_grasp)
    if ACONF:
        return (mt.path[-1], cmd)
    return (cmd,)


## ------------------------------------------------------------------------------


# def get_ik_gen(problem, max_attempts=100, collisions=True, learned=True, teleport=False,
#                ir_only=False, pick_up=True, given_grasp_conf=False,
#                soft_failures=False, verbose=False, visualize=False, **kwargs):
#     """ given grasp of target object p, return base conf and arm traj """
#     ir_max_attempts = 40
#     ir_sampler = get_ir_sampler(problem, collisions=collisions, learned=learned,
#                                 max_attempts=ir_max_attempts, verbose=verbose, **kwargs)
#     if not pick_up and given_grasp_conf:
#         ik_fn = get_ik_fn_old(problem, collisions=collisions, teleport=teleport, verbose=False,
#                               ACONF=True, **kwargs)
#     else:
#         ik_fn = get_ik_fn(problem, pick_up=pick_up, given_grasp_conf=given_grasp_conf,
#                           collisions=collisions, teleport=teleport, verbose=verbose, **kwargs)
#     robot = problem.robot
#     world = problem.world
#     obstacles = problem.fixed if collisions else []
#     heading = '\t\tmobile_streams.get_ik_gen | '
#
#     co_kwargs = dict(articulated=True, verbose=verbose, world=world)
#
#     def gen(a, o, p, g, context=None):
#         if isinstance(o, tuple):
#             obstacles_here = [obs for obs in obstacles if obs != o[0]]
#         else:
#             obstacles_here = [obs for obs in obstacles if obs != o]
#
#         if visualize:
#             samples = []
#
#         process_ik_context(context)
#
#         """ check if hand pose is in collision """
#         p.assign()
#         if 'pstn' in str(p):
#             pose_value = linkpose_from_position(p)
#         else:
#             pose_value = p.value
#         open_arm(robot, a)
#         context_saver = WorldSaver(bodies=[robot, o])
#
#         if visualize:
#             gripper_grasp = robot.visualize_grasp(pose_value, g.value, arm=a, body=g.body)
#             set_renderer(enable=True)
#             wait_unlocked()
#         else:
#             gripper_grasp = robot.set_gripper_pose(pose_value, g.value, arm=a, body=g.body)
#
#         # co_kwargs['verbose'] = True
#         if collided(gripper_grasp, obstacles_here, **co_kwargs):
#             if verbose:
#                 print(f'{heading} -------------- grasp {nice(g.value)} is in collision')
#             if visualize:
#                 set_renderer(enable=True)
#                 wait_unlocked()
#                 robot.remove_gripper(gripper_grasp)
#             return None
#         # co_kwargs['verbose'] = False
#
#         arm_joints = get_arm_joints(robot, a)
#         default_conf = grasp.carry
#
#         ## solve IK for all 13 joints
#         if robot.use_torso and has_tracik():
#             from pybullet_tools.tracik import IKSolver
#             tool_from_root = robot.get_tool_from_root(a)
#             tool_pose = robot.get_grasp_pose(pose_value, g.value, a, body=g.body)
#             gripper_pose = multiply(tool_pose, invert(tool_from_root))
#
#             # gripper_grasp = robot.visualize_grasp(pose_value, g.value, a, body=g.body)
#
#             tool_link = robot.get_tool_link(a)
#             ik_solver = IKSolver(robot, tool_link=tool_link, first_joint=None,
#                                  custom_limits=robot.custom_limits)  ## using all 13 joints
#
#             attempts = 0
#             for conf in ik_solver.generate(gripper_pose):
#                 joint_state = dict(zip(ik_solver.joints, conf))
#                 if max_attempts <= attempts:
#                     if verbose:
#                         print(f'\t\t{get_ik_gen.__name__} failed after {attempts} attempts!')
#                     # wait_unlocked()
#                     if soft_failures:
#                         attempts = 0
#                         yield None
#                         context_saver.restore()
#                         continue
#                     else:
#                         break
#                 attempts += 1
#
#                 base_joints = robot.get_base_joints()
#                 bconf = list(map(joint_state.get, base_joints))
#                 bq = Conf(robot, base_joints, bconf, joint_state=joint_state)
#                 bq.assign()
#
#                 set_joint_positions(robot, arm_joints, default_conf)
#                 if collided(robot, obstacles_here, tag='ik_default_conf', **co_kwargs):
#                     # set_renderer(True)
#                     # wait_for_user()
#                     continue
#
#                 ik_solver.set_conf(conf)
#                 if collided(robot, obstacles_here, tag='ik_final_conf', **co_kwargs):
#                     # robot.add_collision_grasp(a, o, g)
#                     # robot.add_collision_conf(Conf(robot.body, ik_solver.joints, conf))
#                     continue
#
#                 if visualize:
#                     samples.append(visualize_bconf(bconf))
#                     # set_renderer(True)
#                     # Conf(robot, joints, conf).assign()
#                     # wait_for_user()
#
#                 ir_outputs = (bq,)
#                 if ir_only:
#                     if visualize:
#                         [remove_body(samp) for samp in samples]
#                     yield ir_outputs
#                     continue
#
#                 inputs = a, o, p, g
#                 ik_outputs = ik_fn(*(inputs + ir_outputs))
#                 if ik_outputs is None:
#                     continue
#                 if verbose: print('succeed after TracIK solutions:', attempts)
#
#                 if visualize:
#                     [remove_body(samp) for samp in samples]
#                 yield ir_outputs + ik_outputs
#                 context_saver.restore()
#
#         ## do ir sampling of x, y, theta, torso, then solve ik for arm
#         else:
#             inputs = a, o, p, g
#             ir_generator = ir_sampler(*inputs)
#             attempts = 0
#             while True:
#                 if max_attempts <= attempts:
#                     # print(f'{heading} exceeding max_attempts = {max_attempts}')
#                     yield None
#                     # break # TODO(caelan): probably should be break/return
#
#                 attempts += 1
#                 if verbose: print(f'{heading} | attempt {attempts} | inputs = {inputs}')
#
#                 try:
#                     ir_outputs = next(ir_generator)
#                 except StopIteration:
#                     if verbose: print('    stopped ir_generator in', attempts, 'attempts')
#                     print(f'{heading} exceeding ir_generator ir_max_attempts = {ir_max_attempts}')
#                     return
#
#                 if ir_outputs is None:
#                     continue
#                 inp = ir_generator.gi_frame.f_locals
#                 inp = [inp[k] for k in ['pose', 'grasp', 'custom_limits']]
#                 if verbose:
#                     print(f'           ir_generator  |  inputs = {inp}  |  ir_outputs = {ir_outputs}')
#
#                 if visualize:
#                     bconf = ir_outputs[0].values
#                     samples.append(visualize_bconf(bconf))
#
#                 if ir_only:
#                     yield ir_outputs
#                     continue
#
#                 ik_outputs = ik_fn(*(inputs + ir_outputs))
#                 if ik_outputs is None:
#                     continue
#                 if verbose:
#                     print('succeed after IK attempts:', attempts)
#
#                 if visualize:
#                     [remove_body(samp) for samp in samples]
#                 yield ir_outputs + ik_outputs
#                 return
#                 #if not p.init:
#                 #    return
#     return gen