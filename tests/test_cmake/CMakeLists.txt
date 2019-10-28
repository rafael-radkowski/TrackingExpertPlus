# TrackingExpert+ cmake file. 
# /tests-files. 
#
# source and project build file for TrackingExpert+
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
# 
cmake_minimum_required(VERSION 2.6)

# cmake modules
set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

# set policies
cmake_policy(SET CMP0074 NEW)
cmake_policy(SET CMP0015 NEW)
cmake_policy(SET CMP0010 NEW)

project(cmake_test)

# cmake modules
set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/../../cmake)

message(${PROJECT_SOURCE_DIR})
message(${CMAKE_MODULE_PATH})

#----------------------------------------------------------------------
# Compiler standards

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)



# Required packages

FIND_PACKAGE(Structure REQUIRED)



################################################################
