# Use Isaac Gym to render generated trajectories

Here are tools for rendering generated scenes and trajectories in Isaac Gym (public). You can also use any rendering platform, e.g. Blender, ThreeJS, Isaac Sim.

## Installation

### Step 1. Download Isaac Gym
1. download isaac gym from https://developer.nvidia.com/isaac-gym/download, 
2. follow instruction in docs/install.html to install isaacgym 
    `cd ~/Documents/isaacgym/python; pip install -e .`
3. test that default installation works
    `cd examples; python joint_monkey.py`

### Step 2. Other prerequisites 

```shell
## packages for replay rendering purposes
pip install setuptools_scm trimesh h5py opencv-python

## otherwise Python 3.8 failed to load shared libraries "libpython3.8.so.1.0" (F4-210)
export LD_LIBRARY_PATH=/home/yang/miniconda3/envs/kitchen/lib

```

### Step 3. Test rendering works 

```shell
python pybullet_planning/tutorials/test_replay.py
```

---

## Common issues

### Issue: Segmentation fault (core dumped)
Problem is `vulkaninfo` [doesn't work](https://forums.developer.nvidia.com/t/cannot-run-the-examples/165180/2)
Install [vulkan](https://www.reddit.com/r/linux4noobs/comments/g9ru6n/vulkan_not_working_ubuntu_2004/)

```shell
# sudo apt install libvulkan1
sudo apt install mesa-vulkan-drivers vulkan-tools
# export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json
```

### Issue: Please don't do the following way, didn't reboot successfully and TIG helped solved the driver problem

[Uninstall current NVIDIA driver](https://askubuntu.com/questions/206283/how-can-i-uninstall-a-nvidia-driver-completely)
```shell
sudo apt-get remove --purge '^nvidia-.*'
sudo apt-get install ubuntu-desktop
sudo rm /etc/X11/xorg.conf
echo 'nouveau' | sudo tee -a /etc/modules
````

2. [Install driver again](https://ubuntu.com/server/docs/nvidia-drivers-installation)

```shell
sudo ubuntu-drivers install --gpgpu nvidia:550-server
sudo apt install nvidia-utils-550-server
```

---

## Optional: If you are part of NVIDIA Seattle Robotics Lab (SRL internal use)
(if you are part of SRL, you may use full version of Isaac Gym / Sim tools ) add python path to `srl_stream`
```
## install the repo
git clone git@gitlab.com:nvidia_srl/caelan/srl_stream.git

## or whatever the parent repo you're using, so that pybullet_planning references in srl_stream can be found
export PYTHONPATH=~/Documents/nvidia/vlm-tamp:$PYTHONPATH  
```



in python code, replace where `create_single_world` is used in `isaac_tools/gym_utils.py`
```
## at beginning of file
import platform
if platform.node() == 'meraki':
    sys.path.append('/home/yang/Documents/playground/srl_stream/src')
else:
    sys.path.append('/home/zhutiany/Documents/playground/srl_stream/src')

## where the function is used
from isaac_tools.gym_world import create_single_world
## from isaac_tools.isaac_gym_world import create_single_world  ## default one with minimum usage
```