cmake_minimum_required(VERSION 3.0.2)
project(wasrt_ros)

add_compile_options(-std=c++17 -O3)

find_package(catkin REQUIRED COMPONENTS
    roscpp
    rospy
    std_msgs
    sensor_msgs
    geometry_msgs
    cv_bridge
    tf2_ros
    message_filters
)
find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)

catkin_package(
    CATKIN_DEPENDS roscpp rospy std_msgs sensor_msgs geometry_msgs cv_bridge tf2_ros message_filters
)

include_directories(
    include
    ${catkin_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
    ${EIGEN3_INCLUDE_DIRS}
)

add_executable(camera_preprocessor_node src/camera_preprocessor_node.cpp)
target_link_libraries(camera_preprocessor_node ${catkin_LIBRARIES} ${OpenCV_LIBRARIES})

add_executable(lidar_verifier_node src/lidar_verifier_node.cpp)
target_link_libraries(lidar_verifier_node ${catkin_LIBRARIES} ${OpenCV_LIBRARIES})

add_executable(ground_projector_node src/ground_projector_node.cpp)
target_link_libraries(ground_projector_node ${catkin_LIBRARIES} ${OpenCV_LIBRARIES})

install(
    TARGETS camera_preprocessor_node lidar_verifier_node ground_projector_node
    RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
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
