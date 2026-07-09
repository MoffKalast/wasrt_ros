cmake_minimum_required(VERSION 3.0.2)
project(wasrt_ros)

find_package(catkin REQUIRED COMPONENTS
    sensor_msgs
    rospy
    cv_bridge
)

catkin_package(
    CATKIN_DEPENDS rospy sensor_msgs cv_bridge
)

include_directories(
    ${catkin_INCLUDE_DIRS}
)

file(
    GLOB PYTHON_SCRIPTS_DIR "scripts/*.py"
)
catkin_install_python(
    PROGRAMS ${PYTHON_SCRIPTS_DIR}
    DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(
    DIRECTORY public launch
    DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)