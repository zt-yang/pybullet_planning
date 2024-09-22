## Note to myself and for other NVIDIA SRL collaborators

1. clone git@gitlab.com:nvidia_srl/caelan/srl_stream.git
2. download isaac gym from https://developer.nvidia.com/isaac-gym/download, 
    a. follow instruction in docs/install.html to install isaacgym 
        `(cd ~/Documents/isaacgym/python; pip install -e .)`
3. add python path to srl_stream (srl_stream/src), then

```shell
pip install setuptools_scm trimesh h5py opencv-python

## otherwise Python 3.8 failed to load shared libraries "libpython3.8.so.1.0" (F4-210)
export LD_LIBRARY_PATH=/home/yang/miniconda3/envs/fastamp_kitchen/lib

## import pybullet_planning in srl_stream
export PYTHONPATH=~/Documents/nvidia/vlm-tamp:$PYTHONPATH

cd tutorial; python test_gym.py
```

## Issue: Segmentation fault (core dumped)
Problem is `vulkaninfo` [doesn't work](https://forums.developer.nvidia.com/t/cannot-run-the-examples/165180/2)
Install [vulkan](https://www.reddit.com/r/linux4noobs/comments/g9ru6n/vulkan_not_working_ubuntu_2004/)

```shell
# sudo apt install libvulkan1
sudo apt install mesa-vulkan-drivers vulkan-tools
# export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json
```

## Issue: Please don't do the following way, didn't reboot successfully and TIG helped solved the driver problem

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