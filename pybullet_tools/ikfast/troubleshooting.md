# Troubleshooting with IKFast Compilation

We recommend using [IKFast](http://openrave.org/docs/0.8.2/openravepy/ikfast/), an analytical inverse kinematics solver, instead of PyBullet's damped least squares solver.
IKFast bindings are included for the following robots:
* Franka Panda - `pybullet-planning$ (cd pybullet_tools/ikfast/franka_panda; python setup.py)`
* MOVO - `pybullet-planning$ (cd pybullet_tools/ikfast/movo; python setup.py)`
* PR2 - `pybullet-planning$ (cd pybullet_tools/ikfast/pr2; python setup.py)`

### Issue 1: `error: command 'x86_64-linux-gnu-gcc' failed with exit status 1`

```
sudo apt-get install build-essential libssl-dev libffi-dev python-dev
```

### Issue 2: compiled but no built console output, and cannot import

Make sure the python version in compiled output `ikLeft.cpython-38-x86_64-linux-gnu` is the same as your venv python version (3.8), by running 
`pybullet-planning$ (cd pybullet_tools/ikfast/pr2; python3.8 setup.py)`

