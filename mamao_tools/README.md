# Set up FASTAMP

```shell
conda env create -f environment_kitchen.yml
conda activate fastamp_kitchen
pip install git+https://github.com/openai/CLIP.git
pip install pybullet==3.2.5 lark
```

install TracIK for PR2 base, torso, and arm planning:

```shell
cd ~/Documents/
sudo apt-get install ffmpeg
sudo apt-get install libeigen3-dev liborocos-kdl-dev libkdl-parser-dev liburdfdom-dev libnlopt-dev libnlopt-cxx-dev swig
git clone https://github.com/mjd3/tracikpy.git
pip install tracikpy/
```
