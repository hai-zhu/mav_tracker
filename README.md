# mav_tracker
Trajectory tracking controllers for micro aerial vehicles 

```cmd
cd mav_tracker
cd external/acados
git submodule update --recursive --init
mkdir -p build
cd build
cmake -DACADOS_WITH_QPOASES=ON ..
make install
``` 

```cmd
cd mav_tracker
python3 -m venv .env
source .env/bin/activate
pip install -e external/acados/interfaces/acados_template/
pip install pyyaml rospkg
source env_set.sh
```

```cmd
catkin build
source devel/setup.bash
roslaunch mav_nmpc_tracker mav_nmpc_tracker.launch
```
