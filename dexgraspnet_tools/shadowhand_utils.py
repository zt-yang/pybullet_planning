from os.path import dirname, join
from pprint import pprint

from pybullet_tools.utils import wait_unlocked, get_movable_joints, get_joint_name, quat_from_euler, \
    get_joint_positions, set_joint_positions, set_pose, joint_from_name, get_joint_position
from tutorials.test_utils import get_test_world
from pybullet_tools.bullet_utils import load_robot_urdf

from dexgraspnet_tools.dexgraspnet_utils import load_grasp_data, get_grasp_pose

SHADOWHAND_URDF = join(dirname(__file__), 'assets/shadowhand_description/shadowhand.urdf')
SHADOWHAND_JOINTS = ['WRJ2', 'WRJ1', 'FFJ4', 'FFJ3', 'FFJ2', 'FFJ1', 'MFJ4', 'MFJ3', 'MFJ2', 'MFJ1', 'RFJ4', 'RFJ3', 'RFJ2', 'RFJ1', 'LFJ5', 'LFJ4', 'LFJ3', 'LFJ2', 'LFJ1', 'THJ5', 'THJ4', 'THJ3', 'THJ2', 'THJ1']


def test_shadowhand_urdf():
    world = get_test_world(robot=None)
    robot = load_robot_urdf(SHADOWHAND_URDF)
    ## ['WRJ2', 'WRJ1', 'FFJ4', 'FFJ3', 'FFJ2', 'FFJ1', 'MFJ4', 'MFJ3', 'MFJ2', 'MFJ1', 'RFJ4', 'RFJ3', 'RFJ2', 'RFJ1', 'LFJ5', 'LFJ4', 'LFJ3', 'LFJ2', 'LFJ1', 'THJ5', 'THJ4', 'THJ3', 'THJ2', 'THJ1']
    # print([get_joint_name(robot, j) for j in get_movable_joints(robot)])
    return robot


def set_shadowhand_pose_conf(robot, grasp_sample):
    qpos = grasp_sample['qpos']
    trans, euler = get_grasp_pose(qpos)
    pose = trans, quat_from_euler(euler)
    conf = [0, 0] + [qpos[f'robot0:{k}'] if f'robot0:{k}' in qpos else 0 for k in SHADOWHAND_JOINTS[2:]]
    set_pose(robot, pose)
    print(get_joint_position(robot, joint_from_name(robot, 'WRJ2')))
    joints = [joint_from_name(robot, k) for k in SHADOWHAND_JOINTS]
    set_joint_positions(robot, joints, conf)


def load_dexgraspnet_grasps(robot):
    grasp_data = load_grasp_data('sem-Bottle-a86d587f38569fdf394a7890920ef7fd', filtered=True)
    grasp_sample = grasp_data[0]
    # pprint(grasp_sample)
    # print(get_hand_pose(grasp_sample['qpos']))  ## convert for .mjcf file
    pprint(grasp_sample['qpos'])
    set_shadowhand_pose_conf(robot, grasp_sample)  ## convert for .urdf file


if __name__ == '__main__':
    robot = test_shadowhand_urdf()
    load_dexgraspnet_grasps(robot)
    wait_unlocked()