# TrackingExpert+ cmake file. 
#
# Main file for the TrackingExpert+ project. 
#
#
#
# Rafael Radkowski
# Iowa State University
# Virtual Reality Applications Center
# rafael@iastate.eduy
# Sep 22, 2019
# rafael@iastate.edu
#
# MIT License
#---------------------------------------------------------------------
#
# Last edits:
#
# Oct 22, 2019, RR
# - Added the CMAKE_TRY_COMPILE_TARGET_TYPE compiler option. 
# 


cmake_minimum_required(VERSION 2.6)

# Note that this line must be added before project. 
# The cuda compiler will not work properly otherwise. 
set(CMAKE_TRY_COMPILE_TARGET_TYPE "STATIC_LIBRARY")


set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)



# The project name
project(TrackingExpertPlus CUDA CXX)

# set policies
cmake_policy(SET CMP0015 NEW)
cmake_policy(SET CMP0074 NEW)
cmake_policy(SET CMP0028 NEW)

# All cmake find files are managed locally. 
set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)


# Set the project architecture for this example
set( PROJECT_ARCH "x64" )


# set the output paths for the content
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY_DEBUG ${CMAKE_SOURCE_DIR}/lib/)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY_DEBUG ${CMAKE_SOURCE_DIR}/lib/)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_DEBUG ${CMAKE_SOURCE_DIR}/bin/)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY_RELEASE ${CMAKE_SOURCE_DIR}/lib/)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY_RELEASE ${CMAKE_SOURCE_DIR}/lib/)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE ${CMAKE_SOURCE_DIR}/bin/)





################################################################
add_subdirectory(src)
add_subdirectory(tests)
add_subdirectory(external)
