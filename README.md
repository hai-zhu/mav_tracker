# mav_tracker
Trajectory tracking controllers for micro aerial vehicles 

# Create workspace
Create a catkin workspace:
```cmd
source /opt/ros/noetic/setup.bash
mkdir -p ~/catkin_ws/src
cd ~/catkin_ws
catkin_make
```

Install dependencies:
```cmd
cd ~/catkin_ws/src
git clone https://github.com/ethz-asl/mav_comm.git
cd ..
catkin_make
```

Clone the repository:
```cmd
cd ~/catkin_ws/src
git clone https://github.com/hai-zhu/mav_tracker.git
```

## Install acados
```cmd
cd ~/catkin_ws/src/mav_tracker/external
git clone https://github.com/acados/acados.git
cd acados
git checkout 16f677a716ea4abc06b88b1cad7bb433512d66db
git submodule update --recursive --init
mkdir -p build
cd build
cmake -DACADOS_WITH_QPOASES=ON ..
make install
``` 

## Python environment
Create a python virtual environment and build the package:
```cmd
cd ~/catkin_ws/src/mav_tracker
python3 -m venv .env
source .env/bin/activate
source env_set.sh
pip install -e external/acados/interfaces/acados_template/
pip install pyyaml rospkg
cd ~/catkin_ws
catkin_make -DPYTHON_EXECUTABLE=/usr/bin/python3
catkin_make 
source devel/setup.bash
```

To use the tracker
```cmd
cd ~/catkin_ws/src/mav_tracker
source .env/bin/activate
source env_set.sh
cd ~/catkin_ws
source devel/setup.bash
roslaunch mav_nmpc_tracker mav_nmpc_tracker.launch tracking_mode:='track'
```
