cmake_minimum_required(VERSION 3.0.2)
project(wasrt_ros)

add_compile_options(-std=c++17 -O3)

find_package(catkin REQUIRED COMPONENTS
    roscpp
    rospy
    std_msgs
    sensor_msgs
    geometry_msgs
    tf2_ros
    message_filters
)
# cv_bridge is deliberately absent: it links the distro OpenCV 4.5 while these nodes build against
# JetPack's 4.8, and two OpenCVs in one process bind to whichever the loader saw first and crash.
# Image conversion lives in include/wasrt_ros/image_msg.h instead. The python nodes still use the
# python cv_bridge, which resolves through PYTHONPATH at runtime and never touches this link line.
find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs)
find_package(Eigen3 REQUIRED)

# tripwire: if a catkin component ever drags OpenCV in again, say so here rather than at runtime
string(REGEX MATCHALL "libopencv_[a-z_0-9]+" _catkin_opencv "${catkin_LIBRARIES}")
if(_catkin_opencv)
    message(WARNING "catkin_LIBRARIES pulls in OpenCV (${_catkin_opencv}); a second OpenCV will be loaded at runtime")
endif()

catkin_package(
    CATKIN_DEPENDS roscpp rospy std_msgs sensor_msgs geometry_msgs tf2_ros message_filters
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
