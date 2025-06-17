cmake_minimum_required(VERSION 3.22.1)

#
# Project definition
#
project(panda_deburring LANGUAGES CXX)

#
# Options
#
option(BUILD_SHARED_LIBS "Build using shared libraries" ON)

#
# Handle dependencies by reading the package.xml
#
find_package(ament_cmake_auto REQUIRED)
find_package(pinocchio REQUIRED)

ament_auto_find_build_dependencies()

generate_parameter_module(trajectory_publisher_parameters
                          ${PROJECT_NAME}/trajectory_publisher_parameters.yaml)

ament_python_install_package(${PROJECT_NAME} SCRIPTS_DESTINATION
                             lib/${PROJECT_NAME})

generate_parameter_library(ft_calibration_filer_parameters
                           src/ft_calibration_filer_parameters.yaml)
set(${PROJECT_NAME}_headers
    include/${PROJECT_NAME}/visibility.hpp
    include/${PROJECT_NAME}/butterworth_filter.hpp
    include/${PROJECT_NAME}/ft_calibration_filer.hpp)

set(${PROJECT_NAME}_sources src/ft_calibration_filer.cpp)

ament_auto_add_library(${PROJECT_NAME} ${${PROJECT_NAME}_sources}
                       ${${PROJECT_NAME}_headers})

set_target_properties(${PROJECT_NAME} PROPERTIES CXX_VISIBILITY_PRESET hidden)

target_compile_definitions(
  ${PROJECT_NAME}
  PUBLIC -DPANDA_DEBURRING_IS_SHARED
  PRIVATE -DPANDA_DEBURRING_DO_EXPORT)

target_link_libraries(${PROJECT_NAME} pinocchio::pinocchio)
ament_target_dependencies(${PROJECT_NAME} Eigen3)
target_link_libraries(${PROJECT_NAME} ft_calibration_filer_parameters)

#
# Export plugins
#
pluginlib_export_plugin_description_file(controller_interface
                                         controller_plugins.xml)

#
# Installation
#
# install(DIRECTORY config DESTINATION share/${PROJECT_NAME})
install(DIRECTORY config launch rviz DESTINATION share/${PROJECT_NAME})

ament_export_libraries(${PROJECT_NAME})
ament_auto_package()
