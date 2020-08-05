#  cmake file 
#  09_Multiple_Lights
#  
#  Multiple per-fragment-light example
#  This cmake file is part of CS/ME?CPRE 557
# 
#	Rafael Radkowski
#	Iowa State University
#	1620 Howe Hall
#	Ames, IA 50011
#	rafael@iastate.edu
#----------------------------------------------------------------
# Last edits
#


cmake_minimum_required(VERSION 2.6)

# policy for relative subfolders
cmake_policy(SET CMP0015 NEW)
cmake_policy(SET CMP0074 NEW)


# enable visual studio folders
set_property(GLOBAL PROPERTY USE_FOLDERS ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)


# build the gearbox demo
#add_subdirectory(gear_demo)
add_subdirectory(trackingexpertdemo)
