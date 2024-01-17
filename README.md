# Toolbox: pybullet_planning

A fork of Caelan Garrett's [pybullet_planning](https://github.com/caelan/pybullet-planning) utility functions for robotic motion planning, manipulation planning, and task and motion planning (TAMP).

## Setup

1. Clone and grab the submodules, may take a while

```shell
git clone --recurse-submodules git@github.com:zt-yang/pybullet_planning.git
cd pybullet_planning
```

2. Install dependencies

```shell
conda create -n pybullet python==3.8
pip install -r requirements.txt
conda activate pybullet
```

3. Build IK solvers

IKFast solver for PR2 arm planning (see [troubleshooting notes](pybullet_tools/ikfast/troubleshooting.md) if encountered error):

```shell
## sudo apt-get install python-dev
(cd bullet/pybullet_planning/pybullet_tools/ikfast/pr2; python setup.py)
```

TracIK for PR2 base, torso, and arm planning:

```shell
sudo apt-get install libeigen3-dev liborocos-kdl-dev libkdl-parser-dev liburdfdom-dev libnlopt-dev libnlopt-cxx-dev swig
pip install git+https://github.com/mjd3/tracikpy.git
```

---

## Tutorials (to be updated)

Run a flying panda gripper (feg) in kitchen simulation:
```shell
python tutorials/test_floating_gripper.py -t test_feg_pick
python tutorials/test_data_generation.py -c kitchen_full_feg.yaml
```
