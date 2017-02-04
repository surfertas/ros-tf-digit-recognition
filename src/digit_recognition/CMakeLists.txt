cmake_minimum_required(VERSION 2.8.3)
project(digit_recognition)

find_package(catkin REQUIRED COMPONENTS
    cv_bridge
    rospy
    std_msgs
    sensor_msgs
    message_generation
)

catkin_python_setup()

add_service_files(
    FILES
    DigitCheck.srv
)

generate_messages(
    DEPENDENCIES
    std_msgs
    sensor_msgs
)

catkin_package(
    CATKIN_DEPENDS cv_bridge rospy std_msgs sensor_msgs message_runtime
)

include_directories(
    ${catkin_INCLUDE_DIRS}
)
