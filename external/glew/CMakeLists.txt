# Main cmake file 
cmake_minimum_required(VERSION 2.6)

project(glew)

cmake_policy(SET CMP0015 NEW)
cmake_policy(SET CMP0074 NEW)


# enable visual studio folders
set_property(GLOBAL PROPERTY USE_FOLDERS ON)

add_subdirectory(src)