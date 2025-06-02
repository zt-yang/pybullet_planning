from os.path import dirname, join
from pprint import pprint
import random

from pybullet_tools.utils import wait_unlocked, get_movable_joints, get_joint_name, quat_from_euler, \
    get_joint_positions, set_joint_positions, set_pose, joint_from_name, get_joint_position, \
    get_pose, unit_quat, draw_pose, unit_pose, multiply, draw_aabb, get_aabb, YELLOW, BLUE, RED, \
    link_from_name, invert
from pybullet_tools.bullet_utils import load_robot_urdf, get_nice_pose, get_nice_joint_positions
from pybullet_tools.logging_utils import print_yellow
from pybullet_tools.grasp_utils import enumerate_rotational_matrices
from pybullet_tools.camera_utils import set_camera_target_body

from tutorials.test_utils import get_test_world

SHADOWHAND_URDF_PATH = join(dirname(__file__), 'assets/shadowhand_description')
SHADOWHAND_URDFS = {
    'right': join(SHADOWHAND_URDF_PATH, 'shadowhand.urdf'),
    'left': join(SHADOWHAND_URDF_PATH, 'shadowhand_left.urdf'),
}

SHADOWHAND_JOINTS = [
    # 'WRJ2', 'WRJ1',  ## skipping for now
    'FFJ4', 'FFJ3', 'FFJ2', 'FFJ1',
    'MFJ4', 'MFJ3', 'MFJ2', 'MFJ1',
    'RFJ4', 'RFJ3', 'RFJ2', 'RFJ1',
    'LFJ5', 'LFJ4', 'LFJ3', 'LFJ2', 'LFJ1',
    'THJ5', 'THJ4', 'THJ3', 'THJ2', 'THJ1'
]

SHADOWHAND_GRIPPER_ROOT = {
    'right': 'wrist', 'left': 'wrist_left'
}


def translate_to_xml_joint_name(k):
    if '_' in k:
        k, suffix = k.split('_')
    return f'robot0:{k[:-1]}{eval(k[-1])-1}'


def load_shadowhand_urdf(side='right'):
    robot = load_robot_urdf(SHADOWHAND_URDFS[side])
    return robot


def get_shadowhand_joints(side='right'):
    joints = SHADOWHAND_JOINTS
    if side == 'left':
        joints = [j + '_left' for j in joints]
    return joints


def mirror_grasp_pose(pose):
    ## (1,-1,1) (-x,y,-z, w)
    pose = [list(pose[0]), list(pose[1])]
    pose[0][1] = -pose[0][1]
    pose[1][0] = -pose[1][0]
    pose[1][2] = -pose[1][2]
    return pose


def convert_shadowhand_pose_conf(grasp_sample, side='right', verbose=False):
    from dexgraspnet_tools.dexgraspnet_utils import get_grasp_pose
    qpos = grasp_sample['qpos']

    trans, euler = get_grasp_pose(qpos)
    pose = [trans, list(quat_from_euler(euler))]
    if side == 'left':
        pose = mirror_grasp_pose(pose)
    if verbose: pprint(qpos)

    joint_names = get_shadowhand_joints(side)

    # conf = [0, 0] + [qpos[translate_to_xml_joint_name(k)] for k in joints[2:]]
    conf = [qpos[translate_to_xml_joint_name(k)] for k in joint_names]
    if verbose: pprint(dict(zip(joint_names, conf)))
    return pose, conf


def test_load_object_grasp(grasp_object='sem-Bottle-af3dda1cfe61d0fc9403b0d0536a04af',
                           random_sample=False, verbose=True):
    from dexgraspnet_tools.dexgraspnet_utils import load_grasp_data, load_object_in_pybullet
    grasp_data = load_grasp_data(grasp_object, filtered=True)
    if random_sample:
        grasp_sample = random.choice(grasp_data)
    else:
        grasp_sample = grasp_data[0]
    body = load_object_in_pybullet(grasp_object, float(grasp_sample["scale"]), verbose=verbose)
    return body, grasp_sample


def load_test_object_and_grasp_pose(grasp_object='sem-Bottle-af3dda1cfe61d0fc9403b0d0536a04af',
                                    side='right', random_sample=False, transform=None, **kwargs):
    body, grasp_sample = test_load_object_grasp(grasp_object, random_sample, **kwargs)
    pose, conf = convert_shadowhand_pose_conf(grasp_sample, side=side)
    if transform is not None:
        pose = multiply(pose, transform)
    return body, pose, conf


## --------------------------------------------------------------------------


def test_object_grasp_scale_options(grasp_object='sem-Bottle-af3dda1cfe61d0fc9403b0d0536a04af', verbose=True):
    from dexgraspnet_tools.dexgraspnet_utils import load_grasp_data, load_object_in_pybullet
    grasp_data = load_grasp_data(grasp_object, filtered=True)
    grasps_by_scale = {}
    for grasp_sample in grasp_data:
        if
    if random_sample:
        grasp_sample = random.choice(grasp_data)
    else:
        grasp_sample = grasp_data[0]
    body = load_object_in_pybullet(grasp_object, float(grasp_sample["scale"]), verbose=verbose)
    return body, grasp_sample


def load_test_object_and_grasp_sampler(grasp_object='sem-Bottle-af3dda1cfe61d0fc9403b0d0536a04af',
                                       transform=None, **kwargs):
    body, grasp_sample = test_load_object_grasp(grasp_object, **kwargs)
    pose, conf = convert_shadowhand_pose_conf(grasp_sample, side=side)
    if transform is not None:
        pose = multiply(pose, transform)
    return body, pose, conf


## --------------------------------------------------------------------------


def set_shadowhand_pose_conf(robot, grasp_sample, side='right', **kwargs):
    pose, conf = convert_shadowhand_pose_conf(grasp_sample, side=side, **kwargs)

    joint_names = get_shadowhand_joints(side)
    joints = [joint_from_name(robot, k) for k in joint_names]
    print(f"before:\t {get_nice_pose(robot)} {get_nice_joint_positions(robot, joints)}")
    set_pose(robot, pose)
    set_joint_positions(robot, joints, conf)
    print(f"after:\t {get_nice_pose(robot)} {get_nice_joint_positions(robot, joints)}")


def test_load_robot_object_grasp(grasp_object='sem-Bottle-af3dda1cfe61d0fc9403b0d0536a04af'):
    world = get_test_world(robot=None)
    robot = load_shadowhand_urdf()

    body, grasp_sample = test_load_object_grasp(grasp_object=grasp_object)
    set_shadowhand_pose_conf(robot, grasp_sample)  ## convert for .urdf file
    wait_unlocked()


def test_load_two_hands(pause=True):
    world = get_test_world(robot=None)
    draw_pose(unit_pose())

    right = load_shadowhand_urdf('right')
    set_pose(right, ((0, 0.15, 0), unit_quat()))

    left = load_shadowhand_urdf('left')
    set_pose(left, ((0, -0.15, 0), unit_quat()))

    if pause:
        wait_unlocked()
    return right, left


def test_set_grasps_of_two_hands(grasp_object='sem-Bottle-af3dda1cfe61d0fc9403b0d0536a04af'):
    right, left = test_load_two_hands(pause=False)

    body, grasp_sample = test_load_object_grasp(grasp_object=grasp_object)
    set_camera_target_body(body, dx=0.5, dy=0, dz=0.5)

    set_shadowhand_pose_conf(right, grasp_sample)
    set_shadowhand_pose_conf(left, grasp_sample, side='left')

    wait_unlocked()


if __name__ == '__main__':
    """
    pip install transforms3d
    """
    # test_load_robot_object_grasp()
    # test_load_two_hands()
    test_set_grasps_of_two_hands()
