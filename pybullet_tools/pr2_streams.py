from __future__ import print_function

from pybullet_tools.bullet_utils import check_cfree_gripper
from pybullet_tools.ikfast.pr2.ik import pr2_inverse_kinematics
from pybullet_tools.ikfast.utils import USE_CURRENT
from pybullet_tools.pr2_primitives import Grasp, \
    APPROACH_DISTANCE, TOP_HOLDING_LEFT_ARM, get_tool_from_root, Conf, Commands, create_trajectory, \
    Trajectory, State
from pybullet_tools.pr2_problems import create_pr2
from pybullet_tools.pr2_utils import open_arm, arm_conf, get_gripper_link, get_arm_joints, \
    learned_pose_generator, PR2_TOOL_FRAMES, get_group_joints, get_group_conf, TOOL_POSE, MAX_GRASP_WIDTH, GRASP_LENGTH, \
    SIDE_HEIGHT_OFFSET, approximate_as_prism, set_group_conf
from pybullet_tools.utils import wait_unlocked, WorldSaver
from .general_streams import *

BASE_EXTENT = 3.5 # 2.5
BASE_LIMITS = (-BASE_EXTENT*np.ones(2), BASE_EXTENT*np.ones(2))
GRASP_LENGTH = 0.03
APPROACH_DISTANCE = 0.1 + GRASP_LENGTH
SELF_COLLISIONS = False
DEFAULT_RESOLUTION = math.radians(3) # 0.05
LINK_POSE_TO_JOINT_POSITION = {}

########################################################################


def pr2_grasp(body, value, grasp_type=None):
    if grasp_type == None:
        euler = euler_from_quat(value[1])
        grasp_type = 'top'
        if euler[0] == euler[1] == 0:
            grasp_type = 'side'
    approach_vector = APPROACH_DISTANCE * get_unit_vector([1, 0, 0])
    return Grasp(grasp_type, body, value, multiply((approach_vector, unit_quat()), value),
                 TOP_HOLDING_LEFT_ARM)


def get_handle_grasps(body_joint, tool_pose=TOOL_POSE, body_pose=unit_pose(),
                      max_width=MAX_GRASP_WIDTH, grasp_length=GRASP_LENGTH,
                      robot=None, obstacles=[], full_name=None, world=None):
    from pybullet_tools.utils import Pose
    from bullet_utils import find_grasp_in_db, add_grasp_in_db

    DEBUG = True

    PI = math.pi
    body = body_joint[0]
    handle_link = get_handle_link(body_joint)

    found, db, db_file = find_grasp_in_db('handle_grasps.json', full_name)
    if found is not None: return found

    handle_pose = get_handle_pose(body_joint)

    # TODO: compute bounding box width wrt tool frame
    center, (w, l, h) = approximate_as_prism(body, body_pose=body_pose, link=handle_link)
    translate_center = Pose(point=point_from_pose(body_pose)-center)
    num_h = 4
    top_offset = h / num_h
    gl = grasp_length / 3

    grasps = []
    rots = [[0,0,0], [0,0,PI/2], [0,0,PI], [0,PI/2,0], [0,PI,0], [PI/2,0,0], [PI,0,0]]
    title = f'pr2_streams.get_handle_grasps({full_name}): '
    all_rots = []
    for i in range(len(rots)):
        for j in range(len(rots)):
            r1 = Pose(euler=rots[i])
            r2 = Pose(euler=rots[j])
            all_rots.append(multiply(r1, r2))
            index = f'{i*len(rots)+j}/{len(rots)**2}'
    #
    # all_rots = list(set(all_rots))
    # for i in range(len(all_rots)):
    #     rot = all_rots[i]
    #     for gh in range(0, num_h):
    #         index = f"{i}/{len(all_rots)}"

            for gh in range(0, num_h):
                translate_z = Pose(point=[gh*top_offset, 0, w / 2 - gl])
                grasp = multiply(tool_pose, translate_z, r1, r2, translate_center, body_pose)
                result = check_cfree_gripper(grasp, world, handle_pose, obstacles, visualize=DEBUG) ##
                print(f'{title}test grasp ({index}, gh={gh}), {result}')
                if result:
                    if grasp not in grasps:
                        grasps += [grasp]
                else:
                    break

    ## lastly store the newly sampled grasps
    add_grasp_in_db(db, db_file, full_name, grasps)
    return grasps

## -------- moved to robot.
# def iterate_approach_path(robot, arm, gripper, pose_value, grasp, body=None):
#     tool_from_root = get_tool_from_root(robot, arm)
#     grasp_pose = multiply(pose_value, invert(grasp.value))
#     approach_pose = multiply(pose_value, invert(grasp.approach))
#     for tool_pose in interpolate_poses(grasp_pose, approach_pose):
#         set_pose(gripper, multiply(tool_pose, tool_from_root))
#         # if body is not None:
#         #     set_pose(body, multiply(tool_pose, grasp.value))
#         yield


def get_ir_sampler(problem, custom_limits={}, max_attempts=40, collisions=True,
                   learned=True, verbose=False):
    robot = problem.robot
    world = problem.world
    obstacles = [o for o in problem.fixed if o not in problem.floors] if collisions else []
    gripper = problem.get_gripper(visual=False)
    heading = f'   pr2_streams.get_ir_sampler | '

    def gen_fn(arm, obj, pose, grasp):

        pose.assign()
        if isinstance(obj, tuple): ## may be a (body, joint) or a body with a marker
            obj = obj[0]
        if 'pstn' in str(pose): ## isinstance(pose, Position): ## path problem
            pose_value = linkpose_from_position(pose)
        else:
            pose_value = pose.value

        if hasattr(world, 'refine_marker_obstacles'):
            approach_obstacles = problem.world.refine_marker_obstacles(obj, obstacles)
            ## {obst for obst in obstacles if obst !=obj} ##{obst for obst in obstacles if not is_placement(obj, obst)}
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
                # if pairwise_collision(obj, b):
                #     print(f'       get_ir_sampler | in approach, obj {obj} at {nice(get_pose(obj))} with obj {b} at {nice(get_pose(b))}')
                #     return
            # else:
            #     if any([pairwise_collision(gripper, b) or pairwise_collision(obj, b) for b in approach_obstacles]):
            #         return


        # gripper_pose = multiply(pose_value, invert(grasp.value)) # w_f_g = w_f_o * (g_f_o)^-1
        tool_from_root = get_tool_from_root(robot, arm)
        gripper_pose = multiply(robot.get_grasp_pose(pose_value, grasp.value, arm, body=grasp.body), invert(tool_from_root))

        default_conf = arm_conf(arm, grasp.carry)
        arm_joints = get_arm_joints(robot, arm)
        base_joints = get_group_joints(robot, 'base')
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
                if robot.USE_TORSO:
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
                # wconf = problem.get_wconf()
                # wconf.printout()
                # print(f'pr2_streams.get_ir_sampler() -> {bq} is c-free from {obstacles + [obj]}')
                yield (bq,)
                break
            else:
                yield None
    return gen_fn

##################################################

def solve_nearby_ik(robot, arm, approach_pose, custom_limits={}):
    arm_joints = get_arm_joints(robot, arm)
    grasp_conf = get_joint_positions(robot, arm_joints)
    if has_tracik():
        from pybullet_tools.tracik import IKSolver
        tool_link = robot.get_tool_link(arm)
        ik_solver = IKSolver(robot, tool_link=tool_link, first_joint=arm_joints[0],
                             custom_limits=custom_limits)  # TODO: cache
        approach_conf = ik_solver.solve(approach_pose, seed_conf=grasp_conf)
    else:
        # TODO(caelan): sub_inverse_kinematics's clone_body has large overhead
        approach_conf = pr2_inverse_kinematics(robot, arm, approach_pose, custom_limits=custom_limits,
                                               upper_limits=USE_CURRENT, nearby_conf=USE_CURRENT)
        # approach_conf = sub_inverse_kinematics(robot, arm_joints[0], arm_link, approach_pose, custom_limits=custom_limits)
    return approach_conf

def get_ik_fn(problem, custom_limits={}, collisions=True, teleport=False, verbose=False, ACONF=False):
    robot = problem.robot
    obstacles = problem.fixed if collisions else []
    world = problem.world
    world_saver = WorldSaver()
    title = 'pr2_streams.get_ik_fn:\t'

    def fn(arm, obj, pose, grasp, base_conf, wconf=None, fluents=[]):
        if wconf is not None:
            wconf.assign()
        if fluents:
            attachments = process_motion_fluents(fluents, robot) # TODO(caelan): use attachments
        else:
            world_saver.restore()

        if isinstance(obj, tuple): ## may be a (body, joint) or a body with a marker
            body = obj[0]
        else:
            body = obj

        if 'pstn' in str(pose): ## isinstance(pose, Position):
            pose_value = linkpose_from_position(pose)
        else:
            pose_value = pose.value

        ## TODO: change to world.get_grasp_parent
        addons = [body]
        if hasattr(world, 'BODY_TO_OBJECT') and world.BODY_TO_OBJECT[body].grasp_parent != None:
            addons.append(world.BODY_TO_OBJECT[body].grasp_parent)
        addon_obstacles = obstacles + addons if collisions else []

        approach_obstacles = {obst for obst in obstacles if not is_placement(body, obst)}
        # approach_obstacles = problem.world.refine_marker_obstacles(obj, approach_obstacles)  ## for steerables

        # gripper_pose = multiply(pose_value, invert(grasp.value))
        # #approach_pose = multiply(grasp.approach, gripper_pose)
        # approach_pose = multiply(pose_value, invert(grasp.approach))

        tool_from_root = get_tool_from_root(robot, arm)
        gripper_pose = multiply(robot.get_grasp_pose(pose_value, grasp.value, body=obj), invert(tool_from_root))
        approach_pose = multiply(robot.get_grasp_pose(pose_value, grasp.approach, body=obj), invert(tool_from_root))

        arm_link = get_gripper_link(robot, arm)
        arm_joints = get_arm_joints(robot, arm)

        default_conf = arm_conf(arm, grasp.carry)
        #sample_fn = get_sample_fn(robot, arm_joints)
        pose.assign()
        base_conf.assign()
        open_arm(robot, arm)
        set_joint_positions(robot, arm_joints, default_conf) # default_conf | sample_fn()

        if base_conf.joint_state is not None:
            grasp_conf = list(map(base_conf.joint_state.get, arm_joints))
            set_joint_positions(robot, arm_joints, grasp_conf)
        else:
            grasp_conf = pr2_inverse_kinematics(robot, arm, gripper_pose, custom_limits=custom_limits) #, upper_limits=USE_CURRENT)
                                                #nearby_conf=USE_CURRENT) # upper_limits=USE_CURRENT,

        if (grasp_conf is None) or collided(robot, addon_obstacles, articulated=False): ## approach_obstacles): # [obj]
            #wait_unlocked()
            if verbose:
                if grasp_conf != None:
                    grasp_conf = nice(grasp_conf)
                print(f'{title}Grasp IK failure | {grasp_conf} = pr2_inverse_kinematics({robot} at {nice(base_conf.values)}, '
                      f'{arm}, {nice(gripper_pose[0])}) | pose = {pose}, grasp = {grasp}')
                for b in addon_obstacles:
                    if pairwise_collision(robot, b):
                        # set_renderer(True)
                        print(f'                        robot at {nice(base_conf.values)} colliding with {b} at {nice(get_pose(b))}')
            #if grasp_conf is not None:
            #    print(grasp_conf)
            #    #wait_if_gui()
            return None
        elif verbose:
            print(f'{title}Grasp IK success | {nice(grasp_conf)} = pr2_inverse_kinematics({robot} at {nice(base_conf.values)}, '
                  f'{arm}, {nice(gripper_pose[0])}) | pose = {pose}, grasp = {grasp}')

        if has_tracik():
            from pybullet_tools.tracik import IKSolver
            tool_link = robot.get_tool_link(arm)
            ik_solver = IKSolver(robot, tool_link=tool_link, first_joint=arm_joints[0], custom_limits=custom_limits) # TODO: cache
            approach_conf = ik_solver.solve(approach_pose, seed_conf=grasp_conf)
        else:
            # TODO(caelan): sub_inverse_kinematics's clone_body has large overhead
            approach_conf = pr2_inverse_kinematics(robot, arm, approach_pose, custom_limits=custom_limits,
                                                   upper_limits=USE_CURRENT, nearby_conf=USE_CURRENT)
            #approach_conf = sub_inverse_kinematics(robot, arm_joints[0], arm_link, approach_pose, custom_limits=custom_limits)
        if (approach_conf is None) or collided(robot, addon_obstacles, articulated=False): ##
            if verbose:
                if approach_conf != None:
                    approach_conf = nice(approach_conf)
                print(f'{title}Approach IK failure', approach_conf)
                for b in addon_obstacles:
                    if pairwise_collision(robot, b):
                        print(f'                        robot at {nice(base_conf.values)} colliding with {b} at {nice(get_pose(b))}')
            # wait_if_gui()
            return None
        elif verbose:
            print(f'{title}Approach IK success | sub_inverse_kinematics({robot} at {nice(base_conf.values)}, '
                  f'{arm}, {nice(approach_pose[0])}) | pose = {pose}, grasp = {nice(grasp.approach)} -> {nice(approach_conf)}')

        # ## -------------------------------------------
        # arm_joints = get_arm_joints(robot, 'left')
        # aconf = Conf(robot, arm_joints, get_joint_positions(robot, arm_joints))
        # print(f'@ pr2_streams.get_ik_fn() -> aconf = {aconf} | bconf = {base_conf}')
        # ## -------------------------------------------

        set_joint_positions(robot, arm_joints, approach_conf)
        #approach_conf = get_joint_positions(robot, arm_joints)
        attachment = grasp.get_attachment(problem.robot, arm)
        attachments = {}  ## {attachment.child: attachment} TODO: problem with having (body, joint) tuple
        if teleport:
            path = [default_conf, approach_conf, grasp_conf]
        else:
            resolutions = DEFAULT_RESOLUTION * np.ones(len(arm_joints))
            grasp_path = plan_direct_joint_motion(robot, arm_joints, grasp_conf, attachments=attachments.values(),
                                                  obstacles=approach_obstacles, self_collisions=SELF_COLLISIONS,
                                                  use_aabb=True, cache=True,
                                                  custom_limits=custom_limits, resolutions=resolutions/2.)
            if grasp_path is None:
                if verbose: print(f'{title}Grasp path failure')
                return None
            set_joint_positions(robot, arm_joints, default_conf)
            approach_path = plan_joint_motion(robot, arm_joints, approach_conf, attachments=attachments.values(),
                                              obstacles=obstacles, self_collisions=SELF_COLLISIONS,
                                              custom_limits=custom_limits, resolutions=resolutions,
                                              use_aabb=True, cache=True,
                                              restarts=2, iterations=25, smooth=0) # smooth=25
            if approach_path is None:
                if verbose: print(f'{title}\tApproach path failure')
                return None
            path = approach_path + grasp_path
        mt = create_trajectory(robot, arm_joints, path)
        attachments = {attachment.child: attachment} ## TODO: problem with having (body, joint) tuple
        cmd = Commands(State(attachments=attachments), savers=[BodySaver(robot)], commands=[mt])

        set_joint_positions(robot, arm_joints, default_conf)  # default_conf | sample_fn()

        if ACONF:
            return (mt.path[-1], cmd)
        return (cmd,)
    return fn


##################################################



def get_ik_ir_wconf_gen(problem, max_attempts=25, learned=True, teleport=False,
                        verbose=False, visualize=False, **kwargs):
    """ given grasp of target object p, return base conf and arm traj """
    ir_max_attempts = 40
    ir_sampler = get_ir_sampler(problem, learned=learned, max_attempts=ir_max_attempts, verbose=verbose, **kwargs)
    ik_fn = get_ik_fn(problem, teleport=teleport, verbose=False, **kwargs)
    robot = problem.robot
    obstacles = problem.fixed
    heading = 'pr2_streams.get_ik_ir_wconf_gen | '

    def gen(*inputs):
        # set_renderer(enable=True)
        if visualize:
            #set_renderer(enable=True)
            samples = []

        a, o, p, g, w = inputs

        w.assign()
        # w.printout()

        """ check if hand pose is in collision """
        p.assign()
        if 'pstn' in str(p):
            pose_value = linkpose_from_position(p)
        else:
            pose_value = p.value
        gripper_grasp = robot.visualize_grasp(pose_value, g.value, a, body=g.body)
        if collided(gripper_grasp, obstacles):
            remove_body(gripper_grasp)
            # print(f'{heading} -------------- grasp {nice(g.value)} is in collision')
            return
        remove_body(gripper_grasp)

        ## solve IK for all 13 joints
        if robot.USE_TORSO and has_tracik():
            from pybullet_tools.tracik import IKSolver
            tool_from_root = get_tool_from_root(robot, a)
            tool_pose = robot.get_grasp_pose(pose_value, g.value, a, body=g.body)
            gripper_pose = multiply(tool_pose, invert(tool_from_root))

            custom_limits = robot.custom_limits
            joints = list(get_group_conf(robot, 'base-torso'))  ## [0, 1, 17, 2]
            joints.extend(get_arm_joints(robot, a))

            tool_link = PR2_TOOL_FRAMES[a]
            ik_solver = IKSolver(robot, tool_link=tool_link, first_joint=None,
                                 custom_limits=custom_limits)  ## using all 13 joints
            attempts = 0
            for i, conf in enumerate(ik_solver.generate(gripper_pose)):
                if max_attempts*2 <= attempts:
                    return

                bconf = list(conf[:2]) + list([conf[3], conf[2]])  ## ik solution is (x, y, theta, torso), switch last two
                base_joints = robot.get_base_joints()
                bq = Conf(robot, base_joints, bconf)
                bq.assign()
                attempts += 1
                if collided(robot, obstacles):
                    continue
                attempts += 1

                if visualize:
                    samples.append(visualize_bconf(bconf))
                    # set_renderer(True)
                    # Conf(robot, joints, conf).assign()
                    # wait_for_user()

                ir_outputs = (bq,)
                inputs = a, o, p, g
                ik_outputs = ik_fn(*(inputs + ir_outputs))
                if ik_outputs is None:
                    continue
                if verbose: print('succeed after TracIK solutions:', i)

                if visualize:
                    [remove_body(samp) for samp in samples]
                yield ir_outputs + ik_outputs
            return

        ## do ir sampling of x, y, theta, torso, then solve ik for arm
        else:
            inputs = a, o, p, g
            ir_generator = ir_sampler(*inputs)
            attempts = 0
            while True:
                if max_attempts <= attempts:
                    if not p.init:
                        return
                    attempts = 0
                    print(f'{heading} exceeding max_attempts = {max_attempts}')
                    yield None

                attempts += 1
                if verbose: print(f'   {attempts} | get_ik_ir_wconf_gen | inputs = {inputs}')

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

                ik_outputs = ik_fn(*(inputs + ir_outputs))
                if ik_outputs is None:
                    continue
                if verbose: print('succeed after IK attempts:', attempts)

                if visualize:
                    [remove_body(samp) for samp in samples]
                yield ir_outputs + ik_outputs
                return
                #if not p.init:
                #    return
    return gen

##################################################


def get_ik_ir_grasp_handle_gen(problem, max_attempts=40, learned=True, teleport=False,
                               verbose=False, ACONF=False, WCONF=False, **kwargs):
    # TODO: compose using general fn
    # ir_sampler = get_ir_sampler(problem, learned=learned, max_attempts=1, **kwargs)
    ir_sampler = get_ir_sampler(problem, learned=learned, max_attempts=40, **kwargs)
    ik_fn = get_ik_fn(problem, teleport=teleport, ACONF=ACONF, **kwargs)
    def gen(*inputs):
        #set_renderer(enable=verbose)
        if WCONF:
            a, o, p, g, w = inputs
            w.assign()
            inputs = a, o, p, g
        else:
            a, o, p, g = inputs

        ir_generator = ir_sampler(*inputs)
        attempts = 0
        while True:
            if max_attempts <= attempts:
                # if not p.init:
                #     return
                attempts = 0
                yield None
            attempts += 1
            if verbose: print(f'   {attempts} | get_ik_ir_gen | inputs = {inputs}')

            try:
                ir_outputs = next(ir_generator)
            except StopIteration:
                if verbose: print('    stopped ir_generator in', attempts, 'attempts')
                return

            if ir_outputs is None:
                continue
            inp = ir_generator.gi_frame.f_locals
            inp = [inp[k] for k in ['pose', 'grasp', 'custom_limits']]
            if verbose:
                print(f'           ir_generator  |  inputs = {inp}  |  ir_outputs = {ir_outputs}')
                samp = create_box(.1, .1, .1, mass=1, color=(1, 0, 1, 1))
                x,y,_ = ir_outputs[0].values
                set_point(samp, (x,y,0.2))

            ik_outputs = ik_fn(*(inputs + ir_outputs))
            if ik_outputs is None:
                continue
            # print('                         ik_outputs = ik_fn(*(inputs + ir_outputs)) =', ik_outputs, ' | commands =', ik_outputs[0].commands)
            if verbose: print('succeed after IK attempts:', attempts)
            yield ir_outputs + ik_outputs
            return
            #if not p.init:
            #    return
    return gen

##################################################

def get_arm_ik_fn(problem, custom_limits={}, collisions=True, teleport=False, verbose=False):
    robot = problem.robot
    obstacles = problem.fixed if collisions else []
    world = problem.world
    world_saver = WorldSaver()
    title = 'pr2_streams.get_arm_ik_fn:\t'

    def fn(arm, obj, pose, grasp, base_conf, grasp_conf, fluents=[]):
        if isinstance(obj, tuple): ## may be a (body, joint) or a body with a marker
            body = obj[0]
        else:
            body = obj
        if fluents:
            attachments = process_motion_fluents(fluents, robot) # TODO(caelan): use attachments
        else:
            world_saver.restore()

        if 'pstn' in str(pose): ## isinstance(pose, Position):
            pose_value = linkpose_from_position(pose)
        else:
            pose_value = pose.value

        addons = [body]
        # if world.BODY_TO_OBJECT[body].grasp_parent != None:
        #     addons.append(world.BODY_TO_OBJECT[body].grasp_parent)

        # approach_obstacles = {obst for obst in obstacles if not is_placement(obj, obst)}
        approach_obstacles = {o for o in obstacles if o not in addons}
        # approach_obstacles = problem.world.refine_marker_obstacles(obj, approach_obstacles)  ## for steerables

        # gripper_pose = multiply(pose_value, invert(grasp.value)) # w_f_g = w_f_o * (g_f_o)^-1
        # approach_pose = multiply(pose_value, invert(grasp.approach))

        tool_from_root = get_tool_from_root(robot, arm)
        gripper_pose = multiply(robot.get_grasp_pose(pose_value, grasp.value, body=obj), invert(tool_from_root))
        approach_pose = multiply(robot.get_grasp_pose(pose_value, grasp.approach, body=obj), invert(tool_from_root))

        arm_link = get_gripper_link(robot, arm)
        arm_joints = get_arm_joints(robot, arm)

        default_conf = arm_conf(arm, grasp.carry)
        pose.assign()
        base_conf.assign()
        open_arm(robot, arm)
        grasp_conf = grasp_conf.values
        set_joint_positions(robot, arm_joints, grasp_conf) # default_conf | sample_fn()
        # grasp_conf = pr2_inverse_kinematics(robot, arm, gripper_pose, custom_limits=custom_limits) #, upper_limits=USE_CURRENT)
        #                                     #nearby_conf=USE_CURRENT) # upper_limits=USE_CURRENT,
        if (grasp_conf is None) or collided(robot, obstacles, articulated=False): ## approach_obstacles): # [obj]
            if verbose:
                if grasp_conf != None:
                    grasp_conf = nice(grasp_conf)
                print(f'{title}Grasp IK failure | {grasp_conf} = pr2_inverse_kinematics({robot} at {nice(base_conf.values)}, '
                      f'{arm}, {nice(gripper_pose[0])}) | pose = {pose}, grasp = {grasp}')
                for b in obstacles:
                    if pairwise_collision(robot, b):
                        # set_renderer(True)
                        print(f'                        robot at {nice(base_conf.values)} colliding with {b} at {nice(get_pose(b))}')
            return None
        else:
            if verbose:
                print(f'{title}Grasp IK success | {nice(grasp_conf)} = pr2_inverse_kinematics({robot} at {nice(base_conf.values)}, '
                      f'{arm}, {nice(gripper_pose[0])}) | pose = {pose}, grasp = {grasp}')

        #approach_conf = sub_inverse_kinematics(robot, arm_joints[0], arm_link, approach_pose, custom_limits=custom_limits) ##, max_iterations=500
        approach_conf = solve_nearby_ik(robot, arm, approach_pose, custom_limits=custom_limits)
        if (approach_conf is None) or collided(robot, obstacles, articulated=False): ##
            if verbose:
                if approach_conf != None:
                    approach_conf = nice(approach_conf)
                print(f'{title}Approach IK failure | sub_inverse_kinematics({robot} at {nice(base_conf.values)}, '
                      f'{arm}, {nice(approach_pose[0])}) | pose = {pose}, grasp = {nice(grasp.approach)} -> {approach_conf}')
                for b in obstacles:
                    if pairwise_collision(robot, b):
                        print(f'                        robot at {nice(base_conf.values)} colliding with {b} at {nice(get_pose(b))}')
            #wait_if_gui()
            return None
        else:
            if verbose:
                print(f'{title}Approach IK success | sub_inverse_kinematics({robot} at {nice(base_conf.values)}, '
                      f'{arm}, {nice(approach_pose[0])}) | pose = {pose}, grasp = {nice(grasp.approach)} -> {nice(approach_conf)}')

        set_joint_positions(robot, arm_joints, approach_conf)
        #approach_conf = get_joint_positions(robot, arm_joints)
        attachment = grasp.get_attachment(problem.robot, arm)
        attachments = {}  ## {attachment.child: attachment} TODO: problem with having (body, joint) tuple
        if teleport:
            path = [default_conf, approach_conf, grasp_conf]
        else:
            resolutions = DEFAULT_RESOLUTION * np.ones(len(arm_joints))
            grasp_path = plan_direct_joint_motion(robot, arm_joints, grasp_conf, attachments=attachments.values(),
                                                  obstacles=approach_obstacles, self_collisions=SELF_COLLISIONS,
                                                  use_aabb=True, cache=True,
                                                  custom_limits=custom_limits, resolutions=resolutions/2.)
            if grasp_path is None:
                if verbose: print(f'{title}Grasp path failure')
                return None
            set_joint_positions(robot, arm_joints, default_conf)
            approach_path = plan_joint_motion(robot, arm_joints, approach_conf, attachments=attachments.values(),
                                              obstacles=obstacles, self_collisions=SELF_COLLISIONS,
                                              custom_limits=custom_limits, resolutions=resolutions,
                                              use_aabb=True, cache=True,
                                              restarts=2, iterations=25, smooth=0) # smooth=25
            if approach_path is None:
                if verbose: print(f'{title}Approach path failure')
                return None
            path = approach_path + grasp_path
        mt = create_trajectory(robot, arm_joints, path)
        attachments = {attachment.child: attachment} ## TODO: problem with having (body, joint) tuple
        cmd = Commands(State(attachments=attachments), savers=[BodySaver(robot)], commands=[mt])
        return (mt.path[-1], cmd)
    return fn

def get_ik_ungrasp_handle_gen(problem, max_attempts=25, teleport=False, WCONF=False, **kwargs):
    ik_fn = get_arm_ik_fn(problem, teleport=teleport, **kwargs)
    def gen(*inputs):
        if WCONF:
            a, o, p, g, q, aq1, w = inputs
            w.assign()
            inputs = a, o, p, g, q, aq1
        # return ik_fn(*(inputs))
        attempts = 0
        while True:
            if max_attempts <= attempts:
                return None
            yield ik_fn(*(inputs))
            return
    return gen

def get_ik_ungrasp_mark_gen(problem, max_attempts=25, teleport=False, **kwargs):
    ik_fn = get_ik_fn(problem, teleport=teleport, **kwargs)
    def gen(*inputs):
        return ik_fn(*(inputs))
        # attempts = 0
        # while True:
        #     if max_attempts <= attempts:
        #         return None
        #     yield ik_fn(*(inputs))
        #     return
    return gen

##################################################


def bconf_to_pose(bq):
    from pybullet_tools.utils import Pose
    if len(bq.values) == 3:
        x, y, yaw = bq.values
        z = 0
    elif len(bq.values) == 4:
        x, y, z, yaw = bq.values
    return Pose(point=Point(x,y,z), euler=Euler(yaw=yaw))


def pose_to_bconf(rpose, robot):
    (x, y, z), quant = rpose
    yaw = euler_from_quat(quant)[-1]
    if robot.USE_TORSO:
        return Conf(robot, get_group_joints(robot, 'base-torso'), (x, y, z, yaw))
    return Conf(robot, get_group_joints(robot, 'base'), (x, y, yaw))


def add_pose(p1, p2):
    point = np.asarray(p1[0]) + np.asarray(p2[0])
    euler = np.asarray(euler_from_quat(p1[1])) + np.asarray(euler_from_quat(p2[1]))
    return (tuple(point.tolist()), quat_from_euler(tuple(euler.tolist())))


def sample_new_bconf(bq1):
    limits = [0.05] * 3
    def rand(limit):
        return np.random.uniform(-limit, limit)
    values = (bq1.values[i] + rand(limits[i]) for i in range(len(limits)))
    return Conf(bq1.body, bq1.joints, values)


def get_pull_door_handle_motion_gen(problem, custom_limits={}, collisions=True, teleport=False,
                                    num_intervals=30, max_ir_trial=30, visualize=False, verbose=False):
    visualize &= has_gui()
    if teleport:
        num_intervals = 1
    robot = problem.robot
    world = problem.world
    saver = BodySaver(robot)
    world_saver = WorldSaver()
    obstacles = problem.fixed if collisions else []

    def fn(a, o, pst1, pst2, g, bq1, aq1, fluents=[]):
        if pst1.value == pst2.value:
            return None
        if fluents:
            process_motion_fluents(fluents, robot)
        else:
            world_saver.restore()

        saver.restore()
        pst1.assign()
        bq1.assign()
        aq1.assign()

        arm_joints = get_arm_joints(robot, a)
        resolutions = DEFAULT_RESOLUTION * np.ones(len(arm_joints))

        # BODY_TO_OBJECT = problem.world.BODY_TO_OBJECT
        # joint_object = BODY_TO_OBJECT[o]
        # old_pose = get_link_pose(joint_object.body, joint_object.handle_link)
        handle_link = get_handle_link(o)
        old_pose = get_link_pose(o[0], handle_link)
        tool_from_root = get_tool_from_root(robot, a)
        if visualize:
            #set_renderer(enable=True)
            gripper_before = robot.visualize_grasp(old_pose, g.value)

        # gripper_before = multiply(old_pose, invert(g.value))
        gripper_before = multiply(robot.get_grasp_pose(old_pose, g.value, body=o), invert(tool_from_root))
        world_from_base = bconf_to_pose(bq1)
        gripper_from_base = multiply(invert(gripper_before), world_from_base)
        # print('gripper_before', nice(gripper_before))
        # print('invert(gripper_before)', nice(invert(gripper_before)))

        ## saving the mapping between robot bconf to object pst for execution
        mapping = {}
        rpose_rounded = tuple([round(n, 3) for n in bq1.values])
        mapping[rpose_rounded] = pst1.value

        bpath = []
        apath = []
        bq_after = Conf(bq1.body, bq1.joints, bq1.values)
        aq_after = Conf(aq1.body, aq1.joints, aq1.values)
        for i in range(num_intervals):
            step_str = f"pr2_streams.get_pull_door_handle_motion_gen | step {i}/{num_intervals}\t"
            value = (i + 1) / num_intervals * (pst2.value - pst1.value) + pst1.value
            pst_after = Position((pst1.body, pst1.joint), value)
            pst_after.assign()
            new_pose = get_link_pose(o[0], handle_link)
            if visualize:
                gripper_after = robot.visualize_grasp(new_pose, g.value, color=BROWN)
                set_camera_target_body(gripper_after, dx=0.2, dy=0, dz=1) ## look top down
                remove_body(gripper_after)

            gripper_after = multiply(robot.get_grasp_pose(new_pose, g.value, body=o), invert(tool_from_root))
            # gripper_after = multiply(new_pose, invert(g.value))

            ## try to transform the base the same way as gripper to a cfree pose
            world_from_base = multiply(gripper_after, gripper_from_base)
            bq_after = pose_to_bconf(world_from_base, robot)

            bq_after.assign()
            if collided(robot, obstacles, articulated=False, world=world, verbose=True, min_num_pts=10):
                if len(bpath) > 1:
                    bpath[-1].assign()
            else:
                bpath.append(bq_after)
                apath.append(aq_after)
                if verbose: print(f'{step_str} : {nice(bq_after.values)}\t{nice(aq_after.values)}')

            ## save the joint positions as the base moves
            rpose_rounded = tuple([round(n, 3) for n in bq_after.values])
            mapping[rpose_rounded] = value

        if visualize:
            remove_body(gripper_before)

        if len(apath) < num_intervals: ## * 0.75:
            #wait_unlocked()
            return None

        body, joint = o
        if body not in LINK_POSE_TO_JOINT_POSITION:
            LINK_POSE_TO_JOINT_POSITION[body] = {}
        # mapping = sorted(mapping.items(), key=lambda kv: kv[1])
        LINK_POSE_TO_JOINT_POSITION[body][joint] = mapping
        # print(f'pr2_streams.get_pull_door_handle_motion_gen | last bconf = {rpose_rounded}, pstn value = {value}')

        # apath.append(apath[-1])
        # bpath.append(bpath[-1]) ## replicate the last one because somehow it's missing
        bt = Trajectory(bpath)
        at = Trajectory(apath) ## create_trajectory(robot, get_arm_joints(robot, a), apath)
        base_cmd = Commands(State(), savers=[BodySaver(robot)], commands=[bt])
        arm_cmd =  Commands(State(), savers=[BodySaver(robot)], commands=[at])
        bq2 = bt.path[-1]
        aq2 = at.path[-1]
        if aq2.values == aq1.values:
            aq2 = aq1
        step_str = f"pr2_streams.get_pull_door_handle_motion_gen | step {len(bpath)}/{num_intervals}\t"
        if not verbose: print(f'{step_str} : {nice(bq2.values)}\t{nice(aq2.values)}')

        pst1.assign()
        bq1.assign()
        aq1.assign()
        return (bq2, base_cmd, aq2, arm_cmd)

    return fn
#
# def get_turn_knob_handle_motion_gen(problem, custom_limits={}, collisions=True, teleport=False,
#                                     num_intervals=15, visualize=False, verbose=False):
#     if teleport:
#         num_intervals = 1
#     robot = problem.robot
#     world = problem.world
#     saver = BodySaver(robot)
#     obstacles = problem.fixed if collisions else []
#
#     def fn(a, o, pst1, pst2, g, bq1, aq1, fluents=[]):
#         if pst1.value == pst2.value:
#             return None
#
#         saver.restore()
#         pst1.assign()
#         bq1.assign()
#         aq1.assign()
#
#         arm_joints = get_arm_joints(robot, a)
#         resolutions = DEFAULT_RESOLUTION * np.ones(len(arm_joints))
#
#         BODY_TO_OBJECT = problem.world.BODY_TO_OBJECT
#         joint_object = BODY_TO_OBJECT[o]
#         old_pose = get_link_pose(joint_object.body, joint_object.handle_link)
#         if visualize:
#             set_renderer(enable=True)
#             gripper_before_body = robot.visualize_grasp(old_pose, g.value)
#         gripper_before = multiply(old_pose, invert(g.value))  ## multiply(, tool_from_root)
#
#         ## saving the mapping between robot bconf to object pst for execution
#         mapping = {}
#         rpose_rounded = tuple([round(n, 3) for n in aq1.values])
#         mapping[rpose_rounded] = pst1.value
#
#         apath = []
#         aq_after = Conf(aq1.body, aq1.joints, aq1.values)
#         for i in range(num_intervals):
#             step_str = f"pr2_streams.get_pull_door_handle_motion_gen | step {i}/{num_intervals}\t"
#             change = (i + 1) / num_intervals * (pst2.value - pst1.value)
#             value = change + pst1.value
#             pst_after = Position((pst1.body, pst1.joint), value)
#             pst_after.assign()
#             new_pose = get_link_pose(joint_object.body, joint_object.handle_link)
#             if visualize:
#                 if i == 0: remove_body(gripper_before_body)
#                 gripper_after = robot.visualize_grasp(new_pose, g.value, color=BROWN)
#                 set_camera_target_body(gripper_after, dx=0.2, dy=0, dz=1) ## look top down
#                 remove_body(gripper_after)
#             gripper_after = multiply(new_pose, invert(g.value))  ## multiply(, tool_from_root)
#             aconf_after = list(aq1.values)
#             aconf_after[-1] -= change
#             aq_after = Conf(aq1.body, aq1.joints, aconf_after)
#             aq_after.assign()
#
#             if any(pairwise_collision(robot, b) for b in obstacles):
#                 collided = []
#                 for b in obstacles:
#                     if pairwise_collision(robot, b):
#                         collided.append(b)
#                 collided = [world.BODY_TO_OBJECT[c].shorter_name for c in collided]
#                 print(f'{step_str} arm collide at {nice(aconf_after)} with {collided}')
#                 if len(apath) > 1:
#                     apath[-1].assign()
#                     break
#
#             apath.append(aq_after)
#             rpose_rounded = tuple([round(n, 3) for n in aq_after.values])
#             mapping[rpose_rounded] = value
#             if verbose: print(f'{step_str} : {nice(aq_after.values)}')
#
#         if len(apath) < num_intervals * 0.25:
#             return None
#
#         body, joint = o
#         if body not in LINK_POSE_TO_JOINT_POSITION:
#             LINK_POSE_TO_JOINT_POSITION[body] = {}
#         LINK_POSE_TO_JOINT_POSITION[body][joint] = mapping
#
#         at = Trajectory(apath) ## create_trajectory(robot, get_arm_joints(robot, a), apath)
#         arm_cmd =  Commands(State(), savers=[BodySaver(robot)], commands=[at])
#         aq2 = at.path[-1]
#         if aq2.values == aq1.values:
#             aq2 = aq1
#         step_str = f"pr2_streams.get_turn_knob_handle_motion_gen | step {len(apath)}/{num_intervals}\t"
#         if not verbose: print(f'{step_str} : {nice(aq2.values)}')
#         return (aq2, arm_cmd)
#
#     return fn

# def get_pull_drawer_handle_motion_gen(problem, custom_limits={}, collisions=True,
#                                       teleport=False, num_intervals=30, extent=None):
#     if teleport:
#         num_intervals = 1
#     robot = problem.robot
#     saver = BodySaver(robot)
#     obstacles = problem.fixed if collisions else []
#
#     def fn(a, o, pst1, pst2, g, bq1, fluents=[]):  ##
#         if extent == 'max':
#             pst1 = Position(o, 'min')
#             pst2 = Position(o, 'max')
#         elif extent == 'min':
#             pst1 = Position(o, 'max')
#             pst2 = Position(o, 'min')
#         else:
#             if pst1.value == pst2.value:
#                 return None
#         saver.restore()
#         pst1.assign()
#         bq1.assign()
#         BODY_TO_OBJECT = problem.world.BODY_TO_OBJECT
#         joint_object = BODY_TO_OBJECT[o]
#         tool_from_root = get_tool_from_root(robot, a)
#         old_pose = get_link_pose(joint_object.body, joint_object.handle_link)
#         gripper_before = multiply(multiply(old_pose, invert(g.value)), tool_from_root)
#         rpose = bconf_to_pose(bq1)
#
#         bpath = []
#         mapping = {}
#         rpose_rounded = tuple([round(n, 3) for n in bq1.values])
#         mapping[rpose_rounded] = pst1.value
#         for i in range(num_intervals):
#             value = (i+1)/num_intervals * (pst2.value-pst1.value) + pst1.value
#             pst_after = Position((pst1.body, pst1.joint), value)
#             pst_after.assign()
#             new_pose = get_link_pose(joint_object.body, joint_object.handle_link)
#             gripper_after = multiply(multiply(new_pose, invert(g.value)), tool_from_root)
#             transform = multiply(gripper_after, invert(gripper_before))
#
#             rpose_after = add_pose(rpose, transform)
#             bq_after = pose_to_bconf(rpose_after, robot)
#             bpath.append(bq_after)
#
#             rpose_rounded = tuple([round(n, 3) for n in bq_after.values])
#             mapping[rpose_rounded] = value
#
#         body, joint = o
#         if body not in LINK_POSE_TO_JOINT_POSITION:
#             LINK_POSE_TO_JOINT_POSITION[body] = {}
#         LINK_POSE_TO_JOINT_POSITION[body][joint] = mapping
#
#         bt = Trajectory(bpath)
#         cmd = Commands(State(), savers=[BodySaver(robot)], commands=[bt])
#         return (bq_after, cmd)
#     return fn

##################################################

# def get_marker_grasp_gen(problem, collisions=False, randomize=True, visualize=False):
#     collisions = True
#     obstacles = problem.fixed if collisions else []
#     world = problem.world
#     def fn(body):
#         grasps = []
#         markers = world.BODY_TO_OBJECT[body].grasp_markers
#         obs = copy.deepcopy(obstacles)
#         if body in obs: obs.remove(body) ## grasp can collide with the object
#         for marker in markers:
#             approach_vector = APPROACH_DISTANCE * get_unit_vector([1, 0, 0]) ##[2, 0, -1])
#             grasps.extend(HandleGrasp('side', marker, g, multiply((approach_vector, unit_quat()), g), SIDE_HOLDING_LEFT_ARM)
#                           for g in get_marker_grasps(marker, grasp_length=GRASP_LENGTH, robot=world.robot, obstacles=obs))  ## , body_pose=body_pose
#             for grasp in grasps:
#                 grasp.grasp_width = 1
#         return [(g,) for g in grasps]
#     return fn

def get_marker_grasp_gen(problem, collisions=False, randomize=True, visualize=False):
    collisions = True
    obstacles = problem.fixed if collisions else []
    world = problem.world
    def fn(marker):
        grasps = []
        obs = copy.deepcopy(obstacles)
        approach_vector = APPROACH_DISTANCE * get_unit_vector([1, 0, 0]) ##[2, 0, -1])
        grasps.extend(HandleGrasp('side', marker, g, multiply((approach_vector, unit_quat()), g), SIDE_HOLDING_LEFT_ARM)
                      for g in get_marker_grasps(marker, grasp_length=GRASP_LENGTH, robot=world.robot, obstacles=obs))  ## , body_pose=body_pose
        for grasp in grasps:
            grasp.grasp_width = 1
        return [(g,) for g in grasps]
    return fn

def get_marker_grasps(body, under=False, tool_pose=TOOL_POSE, body_pose=unit_pose(),
                    max_width=MAX_GRASP_WIDTH, grasp_length=GRASP_LENGTH, top_offset=SIDE_HEIGHT_OFFSET,
                    robot=None, obstacles=[]):

    from pybullet_tools.utils import Pose

    def check_cfree_gripper(grasp, visualize=False):
        gripper_grasp = robot.visualize_grasp(get_pose(body), grasp)
        if visualize:
            set_camera_target_body(gripper_grasp, dx=0, dy=-1, dz=1)

        for b in obstacles:
            if pairwise_collision(gripper_grasp, b):
                print('making marker grasp collide with', b)
                remove_body(gripper_grasp)
                return False
        remove_body(gripper_grasp)
        return True

    # TODO: compute bounding box width wrt tool frame
    center, (w, l, h) = approximate_as_prism(body, body_pose=body_pose)
    translate_center = Pose(point=point_from_pose(body_pose)-center)
    grasps = []
    x_offset = h/2 - top_offset
    under = True
    for j in range(1 + under):
        swap_xz = Pose(euler=[0, -math.pi / 2 + j * math.pi, 0])
        if l <= max_width:
            for gl in [grasp_length]:  ## [grasp_length/3, grasp_length/2, grasp_length]:
                translate_z = Pose(point=[x_offset, 0, w / 2 - gl])
                for i in range(1, 2):
                    rotate_z = Pose(euler=[0, 0, i * math.pi/2])
                    grasp = multiply(tool_pose, translate_z, rotate_z, swap_xz, translate_center, body_pose)
                    if check_cfree_gripper(grasp):
                        grasps += [grasp]  # , np.array([l])
    return grasps

# def visualize_grasp(robot, body_pose, grasp, arm='left', color=GREEN):
#     link_name = PR2_GRIPPER_ROOTS[arm]
#     links = get_link_subtree(robot, link_from_name(robot, link_name))
#     gripper_grasp = clone_body(robot, links=links, visual=True, collision=True)
#     open_cloned_gripper(gripper_grasp)
#
#     set_all_color(gripper_grasp, color)
#     tool_from_root = get_tool_from_root(robot, arm)
#     grasp_pose = multiply(multiply(body_pose, invert(grasp)), tool_from_root)
#     set_pose(gripper_grasp, grasp_pose)
#
#     direction = get_gripper_direction(grasp_pose)
#     print('\n', nice(grasp_pose), direction)
#     if direction == None:
#         print('new direction')
#         return gripper_grasp, True
#     if 'down' in direction:
#         print('pointing down')
#         return gripper_grasp, True
#     return gripper_grasp, False


##################################################

def sample_points_along_line(body, marker, num_intervals=None, bq=None,
                             limit=(2.5, 3), learned=False):

    x1, y1, z1 = get_pose(body)[0]
    (x2, y2, z2), quat = get_pose(marker)
    k = (y2 - y1) / (x2 - x1)

    def sample_point():
        lo, hi = limit
        dx = np.random.uniform(lo, hi)
        if learned: dx = lo
        return dx, (x2 + dx, y2 + k * dx)

    dx, (x, y) = sample_point()
    pose2 = ((x, y, z2), quat)

    if num_intervals != None and bq != None:
        rx, ry, ryaw = bq
        bqs = []  ## base config of robot
        for n in range(num_intervals):
            x = rx + dx * ((n+1)/num_intervals)
            y = ry + k * dx * ((n+1)/num_intervals)
            bqs.append((x, y, ryaw))
        return pose2, bqs

    return pose2

def get_bqs_given_p2(marker, parent, bq, pose2, num_intervals):
    x1, y1, z1 = get_pose(parent)[0]
    (x2, y2, z2), quat = get_pose(marker)
    k = (y2 - y1) / (x2 - x1)

    ## reverse engineer the dx
    ((x, y, z2), quat) = pose2
    dx = x - x2

    rx, ry, ryaw = bq
    bqs = []  ## base config of robot
    for n in range(num_intervals):
        x = rx + dx * ((n + 1) / num_intervals)
        y = ry + k * dx * ((n + 1) / num_intervals)
        bqs.append((x, y, ryaw))

    return bqs

def get_bqs_given_bq2(marker, parent, bq1, bq2, num_intervals):
    x1, y1, z1 = get_pose(parent)[0]
    (x2, y2, z2), quat = get_pose(marker)
    k = (y2 - y1) / (x2 - x1)

    ## reverse engineer the dx
    dx = bq2[0] - bq1[0]
    x = x2 + dx
    y = y2 + k * dx
    pose2 = ((x, y, z2), quat)

    rx, ry, ryaw = bq1
    bqs = []
    for n in range(num_intervals):
        x = rx + dx * ((n + 1) / num_intervals)
        y = ry + k * dx * ((n + 1) / num_intervals)
        bqs.append((x, y, ryaw))

    return pose2, bqs

def get_marker_pose_gen(problem, num_samples=70, collisions=False, visualize=False):
    from pybullet_tools.pr2_primitives import Pose
    collisions = True
    world = problem.world
    def fn(o, p1):
        poses = []
        parent = world.BODY_TO_OBJECT[o].grasp_parent
        p1.assign()

        for i in range(num_samples):
            pose2 = sample_points_along_line(parent, o)
            if visualize:
                visualize_point(pose2[0], world)
            p2 = Pose(o, pose2)
            poses.append((p2,))
        return poses
    return fn


def get_parent_new_pose(p1, p2, p3):
    x1, y1, _ = p1[0]
    x2, y2, _ = p2[0]
    (x3, y3, z3), quat = p3
    return ((x3+x2-x1, y3+y2-y1, z3), quat)


def get_pull_marker_random_motion_gen(problem, custom_limits={}, collisions=True, max_attempts=30,
                               teleport=False, num_intervals=30, learned=False):
    from pybullet_tools.pr2_primitives import Pose

    if teleport:
        num_intervals = 1
    robot = problem.robot
    saver = BodySaver(robot)
    obstacles = problem.fixed if collisions else []
    world = problem.world
    def fn(a, o, p1, g, bq1, o2, p3, fluents=[]):

        parent = world.BODY_TO_OBJECT[o].grasp_parent
        approach_obstacles = copy.deepcopy(obstacles)
        if parent in approach_obstacles: approach_obstacles.remove(parent)

        attempts = 0
        while attempts < max_attempts:
            saver.restore()
            p1.assign()
            bq1.assign()

            ## sample a p2 along the direction and deduce bconf
            pose2, bqs = sample_points_along_line(parent, o, num_intervals, bq1.values, learned=learned)
            p2 = Pose(o, pose2)
            p4 = Pose(o2, get_parent_new_pose(p1.value, pose2, p3.value))

            bpath = [Conf(robot, robot.get_base_joints(), bq) for bq in bqs]
            collided = False

            ## TODO: do collision checking with other streams
            for bq in bpath:
                bq.assign()
                if any(pairwise_collision(robot, b) or pairwise_collision(parent, b) for b in approach_obstacles):
                    collided = True
            if not collided:
                bt = Trajectory(bpath)
                cmd = Commands(State(), savers=[BodySaver(robot)], commands=[bt])
                yield (p2, bpath[-1], p4, cmd)
    return fn

def get_pull_marker_to_pose_motion_gen(problem, custom_limits={}, collisions=True,
                               teleport=False, num_intervals=30):
    from pybullet_tools.pr2_primitives import Pose

    if teleport:
        num_intervals = 1
    robot = problem.robot
    saver = BodySaver(robot)
    obstacles = problem.fixed if collisions else []
    world = problem.world
    def fn(a, o, p1, p2, g, bq1, o2, p3, fluents=[]):

        parent = world.BODY_TO_OBJECT[o].grasp_parent
        approach_obstacles = copy.deepcopy(obstacles)
        if parent in approach_obstacles: approach_obstacles.remove(parent)

        saver.restore()
        p1.assign()
        bq1.assign()

        ## get list of bconf sample along the direction given p2
        bqs = get_bqs_given_p2(o, parent, bq1.values, p2.value, num_intervals)
        p4 = Pose(o2, get_parent_new_pose(p1.value, p2.value, p3.value))

        bpath = [Conf(robot, robot.get_base_joints(), bq) for bq in bqs]
        collided = False

        ## TODO: do collision checking with other streams
        for bq in bpath:
            bq.assign()
            if any(pairwise_collision(robot, b) or pairwise_collision(parent, b) for b in approach_obstacles):
                collided = True
        if not collided:
            bt = Trajectory(bpath)
            cmd = Commands(State(), savers=[BodySaver(robot)], commands=[bt])
            return (bpath[-1], p4, cmd)
        return None

    return fn


def get_pull_marker_to_bconf_motion_gen(problem, custom_limits={}, collisions=True,
                               teleport=False, num_intervals=30):
    from pybullet_tools.pr2_primitives import Pose

    if teleport:
        num_intervals = 1
    robot = problem.robot
    saver = BodySaver(robot)
    obstacles = problem.fixed if collisions else []
    world = problem.world
    def fn(a, o, p1, g, bq1, bq2, o2, p3, fluents=[]):

        parent = world.BODY_TO_OBJECT[o].grasp_parent
        approach_obstacles = copy.deepcopy(obstacles)
        if parent in approach_obstacles: approach_obstacles.remove(parent)

        saver.restore()
        p1.assign()
        bq1.assign()

        ## get list of bconf sample along the direction given p2
        pose2, bqs = get_bqs_given_bq2(o, parent, bq1.values, bq2.values, num_intervals)
        p2 = Pose(o, pose2)
        p4 = Pose(o2, get_parent_new_pose(p1.value, pose2, p3.value))

        bpath = [Conf(robot, get_group_joints(robot, 'base'), bq) for bq in bqs]
        collided = False

        ## TODO: do collision checking with other streams
        for bq in bpath:
            bq.assign()
            if any(pairwise_collision(robot, b) or pairwise_collision(parent, b) for b in approach_obstacles):
                collided = True
        if not collided:
            bt = Trajectory(bpath)
            cmd = Commands(State(), savers=[BodySaver(robot)], commands=[bt])
            return (p2, p4, cmd)
        return None

    return fn



##################################################

def get_pose_in_region_test():
    def test(o, p, r):
        p.assign()
        obj_aabb = aabb2d_from_aabb(get_aabb(o))
        region_aabb = aabb2d_from_aabb(get_aabb(r))
        # return aabb_overlap(obj_aabb, region_aabb)

        obj_center = get_aabb_center(obj_aabb)
        return aabb_contains_point(obj_aabb, region_aabb)
    return test

def get_bconf_in_region_test(robot):
    rob = robot
    def test(bq, r):
        bq.assign()
        ## needs to be only rob base because arm may stick over in the air
        rob_aabb = aabb2d_from_aabb(get_aabb(rob, link_from_name(rob, "base_link")))
        region_aabb = aabb2d_from_aabb(get_aabb(r))
        # return aabb_overlap(rob_aabb, region_aabb)

        rob_center = get_aabb_center(rob_aabb)
        return aabb_contains_point(rob_center, region_aabb)
    return test


def get_bconf_in_region_gen(problem, collisions=True, max_attempts=10, verbose=False, visualize=False):
    obstacles = problem.fixed if collisions else []
    robot = problem.robot
    yaw = 0

    def gen(region):
        lower, upper = get_aabb(region)
        attempts = 0
        while attempts < max_attempts:
            attempts += 1
            x = np.random.uniform(lower[0], upper[0])
            y = np.random.uniform(lower[1], upper[1])
            bq = Conf(robot, robot.get_base_joints(), (x, y, yaw))
            bq.assign()
            if not any(pairwise_collision(robot, obst) for obst in obstacles):
                if visualize:
                    rbb = create_pr2()
                    set_group_conf(rbb, 'base', bq.values)
                yield (bq,)

    def list_fn(region):
        lower, upper = get_aabb(region)
        attempts = 0
        bqs = []
        while attempts < max_attempts:
            attempts += 1
            x = np.random.uniform(lower[0], upper[0])
            y = np.random.uniform(lower[1], upper[1])
            bq = Conf(robot, robot.get_base_joints(), (x, y, yaw))
            bq.assign()
            if not any(pairwise_collision(robot, obst) for obst in obstacles):
                if visualize:
                    rbb = create_pr2()
                    set_group_conf(rbb, robot.base_group, bq.values)
                bqs.append((bq,))
        return bqs
    return list_fn ## gen

def get_pose_in_region_gen(problem, collisions=True, max_attempts=40, verbose=False, visualize=False):
    from pybullet_tools.pr2_primitives import Pose
    obstacles = problem.fixed if collisions else []
    robot = problem.robot
    def gen(o, r):
        ((_, _, z), quat) = get_pose(o)
        lower, upper = get_aabb(r)
        attempts = 0
        while attempts < max_attempts:
            attempts += 1
            x = np.random.uniform(lower[0], upper[0])
            y = np.random.uniform(lower[1], upper[1])
            pose = (x, y, z), quat
            # x, y, z, yaw = sample_obj_in_body_link_space(robot, region,
            #     PLACEMENT_ONLY=True, XY_ONLY=True, verbose=verbose, **kwargs)
            p = Pose(o, pose)
            p.assign()
            if not any(pairwise_collision(o, obst) for obst in obstacles):
                yield (p,)
    # return gen

    def list_fn(o, r):
        ((_, _, z), quat) = get_pose(o)
        lower, upper = get_aabb(r)
        attempts = 0
        poses = []
        while attempts < max_attempts:
            attempts += 1
            x = np.random.uniform(lower[0], upper[0])
            y = np.random.uniform(lower[1], upper[1])
            pose = (x, y, z), quat
            # x, y, z, yaw = sample_obj_in_body_link_space(robot, region,
            #     PLACEMENT_ONLY=True, XY_ONLY=True, verbose=verbose, **kwargs)
            p = Pose(o, pose)
            p.assign()
            if not any(pairwise_collision(o, obst) for obst in obstacles):
                poses.append((p,))
        return poses
    return list_fn ## gen

##################################################

def process_motion_fluents(fluents, robot, verbose=True):
    if verbose:
        print('Fluents:', fluents)
    attachments = []
    for atom in fluents:
        predicate, args = atom[0], atom[1:]
        if predicate == 'atpose':
            o, p = args
            p.assign()
        elif predicate == 'atgrasp':
            a, o, g = args
            attachments.append(g.get_attachment(robot, a))
        elif predicate == 'atposition':
            o, p = args
            p.assign()
        elif predicate == 'ataconf': # TODO: the arm conf isn't being set pre/post moves correctly
            # a, q = args
            # q.assign()
            pass
        else:
            raise NotImplementedError(atom)
    return attachments

def get_motion_wconf_gen(problem, custom_limits={}, collisions=True, teleport=False, debug=False):
    robot = problem.robot
    saver = BodySaver(robot)
    obstacles = problem.fixed if collisions else []
    def fn(bq1, bq2, w=None, fluents=[], context=None):
        #print(context) # NOTE(caelan): should be None
        saver.restore()
        ## the only thing added from get_motion_wconf
        if w is not None:
            w.assign()
        attachments = process_motion_fluents(fluents, robot) # TODO(caelan): use attachments
        bq1.assign()
        # TODO: did base motion planning fail?
        # TODO: add objects to obstacles

        bconf = get_joint_positions(robot, robot.get_base_joints())
        aconf = get_joint_positions(robot, get_arm_joints(robot, 'left'))

        # arm_joints = get_arm_joints(robot, arm) # TODO(caelan): should set the arms conf
        # default_conf = arm_conf(arm, grasp.carry)
        # set_joint_positions(robot, arm_joints, default_conf)

        params = [(6, 75)] ## (4, 50), (10, 100)
        num_trials = len(params)  ## sometimes it can't find a path to get around the open door
        while num_trials > 0:
            param = params[-num_trials]
            raw_path = plan_joint_motion(robot, bq2.joints, bq2.values, attachments=[],
                                         obstacles=obstacles, self_collisions=SELF_COLLISIONS,
                                         custom_limits=custom_limits, resolutions=None, # TODO: base resolutions
                                         use_aabb=True, cache=True,
                                         restarts=param[0], iterations=param[1], smooth=0) # smooth=50
                                         # restarts=4, iterations=50, smooth=50)
            # break
            num_trials -= 1
            if raw_path != None:
                break
            bq1.assign()

        # print(f'pr2_streams.get_motion_wconf_gen\t under {w.printout(obstacles)}, '
        #       f'from bconf = {nice(bconf)}, aconf = {nice(aconf)}, num_trials = {2-num_trials}')
        if raw_path is None:
            print('Failed motion plan (with world config)!', obstacles)
            if debug:
                for i, bq in enumerate([bq1, bq2]):
                    bq.assign()
                    print('{}) Base conf: {}'.format(i, bq))
                    wait_unlocked()
            return None
        path = [Conf(robot, bq2.joints, q) for q in raw_path]
        bt = Trajectory(path)
        cmd = Commands(State(), savers=[BodySaver(robot)], commands=[bt])
        return (cmd,)
    return fn

##################################################

def get_cfree_btraj_pose_test(robot, collisions=True, verbose=True):
    def test(c, b2, p2):
        # TODO: infer robot from c
        if not collisions:
            return True
        state = c.assign()
        if b2 in state.attachments:
            return True
        p2.assign()

        if verbose:
            robot_pose = robot.get_pose()
            print('    get_cfree_btraj_pose_test   \    pose of robot', nice(robot_pose))
        for _ in c.apply(state):
            state.assign()
            for b1 in state.attachments:
                if pairwise_collision(b1, b2):
                    if verbose:
                        print(f'      collision with {b1}, {b2}')
                        print(f'         pose of {b1}', nice(get_pose(b1)))
                        print(f'         pose of {b2}', nice(get_pose(b2)))
                    #wait_for_user()
                    return False
            if pairwise_collision(robot, b2):
                if verbose:
                    print(f'      collision {robot}, {b2}')
                    print(f'         pose of robot', nice(robot.get_pose()))
                    print(f'         pose of {b2}', nice(get_pose(b2)))
                return False
        # TODO: just check collisions with moving links
        return True
    return test

# def get_motion_list_gen(problem, custom_limits={}, num_attempts=1, collisions=True, teleport=False):
#     robot = problem.robot
#     saver = BodySaver(robot)
#     obstacles = problem.fixed if collisions else []
#     def list_fn(bq1, bq2):
#         paths = []
#         # print('pr2_streams.get_motion_gen\tobstacles:', obstacles)
#         for i in range(num_attempts):
#             saver.restore()
#             bq1.assign()
#             raw_path = plan_joint_motion(robot, bq2.joints, bq2.values, attachments=[],
#                                          obstacles=obstacles, custom_limits=custom_limits, self_collisions=SELF_COLLISIONS,
#                                          restarts=4, max_iterations=50, smooth=50)
#             if raw_path != None:
#                 path = [Conf(robot, bq2.joints, q) for q in raw_path]
#                 bt = Trajectory(path)
#                 cmd = Commands(State(), savers=[BodySaver(robot)], commands=[bt])
#                 paths.append((cmd,))
#         if len(paths) == 0:
#             return None
#         return paths
#     return list_fn

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
        if stream.name in ['inverse-kinematics-wconf', 'sample-pose', 'sample-pose-inside', 'sample-grasp',
                           'test-cfree-pose-pose', 'test-cfree-approach-pose', 'plan-base-motion-wconf']:
            pass
        elif stream.name == 'test-cfree-traj-pose':
            _, o, p = inputs
            p.assign()
        elif stream.name == 'test-cfree-traj-position':
            _, o, p = inputs
            p.assign()
        else:
            raise ValueError(stream.name)

def get_ik_gen(problem, max_attempts=100, collisions=True, learned=True, teleport=False, ir_only=False,
               soft_failures=False, verbose=False, visualize=False, ACONF=False, WCONF=True, **kwargs):
    """ given grasp of target object p, return base conf and arm traj """
    ir_max_attempts = 40
    ir_sampler = get_ir_sampler(problem, collisions=collisions, learned=learned,
                                max_attempts=ir_max_attempts, verbose=verbose, **kwargs)
    ik_fn = get_ik_fn(problem, collisions=collisions, teleport=teleport, verbose=False, ACONF=ACONF, **kwargs)
    robot = problem.robot
    obstacles = problem.fixed if collisions else []
    heading = 'pr2_streams.get_ik_ir_wconf_gen | '

    def gen(a, o, p, g, w=None, context=None):
        #set_renderer(enable=False)
        if visualize:
            #set_renderer(enable=True)
            samples = []

        process_ik_context(context)
        if WCONF and (w is not None):
            w.assign() # TODO(caelan): skip moving bodies/links
            # w.printout()

        """ check if hand pose is in collision """
        p.assign()
        if 'pstn' in str(p):
            pose_value = linkpose_from_position(p)
        else:
            pose_value = p.value
        open_arm(robot, a)
        context_saver = WorldSaver(bodies=[robot, o])

        #gripper_grasp = robot.visualize_grasp(pose_value, g.value, arm=a, body=g.body)
        gripper_grasp = robot.set_gripper_pose(pose_value, g.value, arm=a, body=g.body)
        if collided(gripper_grasp, obstacles, articulated=True): # w is not None
            #wait_unlocked()
            #robot.remove_gripper(gripper_grasp)
            print(f'{heading} -------------- grasp {nice(g.value)} is in collision')
            return
        #robot.remove_gripper(gripper_grasp)
        # Fix grasps
        # Fix negation
        # Open gripper

        arm_joints = get_arm_joints(robot, a)
        default_conf = arm_conf(a, g.carry)

        ## solve IK for all 13 joints
        if robot.USE_TORSO and has_tracik():
            from pybullet_tools.tracik import IKSolver
            tool_from_root = robot.get_tool_from_root(a)
            tool_pose = robot.get_grasp_pose(pose_value, g.value, a, body=g.body)
            gripper_pose = multiply(tool_pose, invert(tool_from_root))

            tool_link = robot.get_tool_link(a)
            ik_solver = IKSolver(robot, tool_link=tool_link, first_joint=None,
                                 custom_limits=robot.custom_limits)  ## using all 13 joints

            attempts = 0
            for conf in ik_solver.generate(gripper_pose): # TODO: islice
                joint_state = dict(zip(ik_solver.joints, conf))
                if max_attempts <= attempts:
                    print(f'{get_ik_gen.__name__} timed out after {attempts} attempts!')
                    #wait_unlocked()
                    if soft_failures:
                        attempts = 0
                        yield None
                        context_saver.restore()
                        continue
                    else:
                        break
                attempts += 1

                base_joints = robot.get_base_joints()
                bconf = list(map(joint_state.get, base_joints))
                bq = Conf(robot, base_joints, bconf, joint_state=joint_state)
                bq.assign()

                set_joint_positions(robot, arm_joints, default_conf)
                if collided(robot, obstacles, articulated=True):
                    # wait_unlocked()
                    continue

                ik_solver.set_conf(conf)
                if collided(robot, obstacles, articulated=True):
                    # wait_unlocked()
                    continue

                if visualize:
                    samples.append(visualize_bconf(bconf))
                    # set_renderer(True)
                    # Conf(robot, joints, conf).assign()
                    # wait_for_user()

                ir_outputs = (bq,)
                if ir_only:
                    yield ir_outputs
                    continue

                inputs = a, o, p, g
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
            inputs = a, o, p, g
            ir_generator = ir_sampler(*inputs)
            attempts = 0
            while True:
                if max_attempts <= attempts:
                    # print(f'{heading} exceeding max_attempts = {max_attempts}')
                    yield None
                    # break # TODO(caelan): probably should be break/return

                attempts += 1
                if verbose: print(f'   {attempts} | get_ik_ir_wconf_gen | inputs = {inputs}')

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

                ik_outputs = ik_fn(*(inputs + ir_outputs))
                if ik_outputs is None:
                    continue
                if verbose: print('succeed after IK attempts:', attempts)

                if visualize:
                    [remove_body(samp) for samp in samples]
                yield ir_outputs + ik_outputs
                return
                #if not p.init:
                #    return
    return gen