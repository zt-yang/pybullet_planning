# Set up FASTAMP inside kitchen-world

first change branch 

```shell
cd kitchen-world
git checkout fastamp

cd pybullet_planning
git pull --recurse-submodules
```
    
then run the following command

```shell
cd fastamp
conda env create -f environment_kitchen.yml
conda activate fastamp_kitchen
pip install git+https://github.com/openai/CLIP.git
pip install pybullet==3.2.5 lark
```

install TracIK for PR2 base, torso, and arm planning:

```shell
sudo apt-get install libeigen3-dev liborocos-kdl-dev libkdl-parser-dev liburdfdom-dev libnlopt-dev libnlopt-cxx-dev swig
cd ~/Documents/
git clone https://github.com/mjd3/tracikpy.git
pip install tracikpy/
```

download the weights and config files for using the models:

```shell

```

(optional) other packages for kitchen-worlds 

```shell
sudo apt-get install ffmpeg
pip install pandas
```