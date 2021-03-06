cmake_minimum_required(VERSION 2.8.3)
project(jsk_fetch_startup)
if($ENV{ROS_DISTRO} STREQUAL "hydro")
  message(STATUS "fetch is not supported on hydro")
  find_package(catkin)
  catkin_package()
  return()
endif()

## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_package(catkin)

###################################
## catkin specific configuration ##
###################################
catkin_package()

catkin_add_env_hooks(99.jsk_fetch_startup SHELLS bash zsh sh
  DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/env-hooks)

#############
## Install ##
#############
install(DIRECTORY config launch scripts data
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION})

#############
## Testing ##
#############
if(CATKIN_ENABLE_TESTING)
  find_package(catkin REQUIRED COMPONENTS rostest roslaunch)
  # https://github.com/ros/ros_comm/pull/730
  # https://github.com/ros/ros_comm/pull/998
  set(roslaunch_check_script ${PROJECT_SOURCE_DIR}/scripts/roslaunch-check)
  # https://github.com/ros/ros_comm/issues/953 could not load launch file with args directory
  #roslaunch_add_file_check(launch/fetch_bringup.launch launch_teleop:=false)
  roslaunch_add_file_check(test/roslaunch-check-fetch_bringup.xml)
  roslaunch_add_file_check(launch/rviz.launch)
  roslaunch_add_file_check(launch/fetch_gazebo_bringup.launch)
  set(roslaunch_check_script ${roslaunch_DIR}/../scripts/roslaunch-check)
endif()

