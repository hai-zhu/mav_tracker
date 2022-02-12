cmake_minimum_required(VERSION 3.0.2)
project(mav_fake_simulator)

## Compile as C++11, supported in ROS Kinetic and newer
# add_compile_options(-std=c++11)


## Set cmake type
# set(CMAKE_BUILD_TYPE Release)


## Find catkin macros and libraries
find_package(catkin REQUIRED COMPONENTS
        roscpp
        rospy
        std_msgs
        geometry_msgs
        nav_msgs
        mav_msgs
        tf
        )

catkin_python_setup()

include_directories(
    ${catkin_INCLUDE_DIRS}
)

## For debugging
# set (CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS}  -g ")
# set (CMAKE_VERBOSE_MAKEFILE ON)

