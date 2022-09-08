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
1. download [models.zip](https://drive.google.com/file/d/1bfwjqha-M_xP-a98fyB4E2UrPdhsXXEk/view?usp=sharing)
2. unzip to extract `models` and `wandb` folders; put them inside `fastamp` folder 

(optional) other packages for kitchen-worlds 

```shell
sudo apt-get install ffmpeg
pip install pandas
```

# Planner performance experiments

```shell
(fastamp_kitchen) ➜  kitchen-worlds git:(fastamp) $ (cd tests; python test_rerun.py)

```

# Visualize planner performance

you can also run the script inside `(fastamp_kitchen)` conda env, but the figure generated is somehow smaller.

```shell
(base) ➜  pybullet_planning git:(fastamp) $ conda activate kitchen
(kitchen) ➜  pybullet_planning git:(fastamp) $ (cd mamao_tools; python plotting.py)
```