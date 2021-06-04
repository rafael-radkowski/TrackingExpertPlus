# TrackingExpert+ cmake file. 
# /tests-files. 
#
# source and project build file for TrackingExpert+
#  Gear box demo
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
cmake_minimum_required(VERSION 3.1)


project(GearTransmissionDemo)
set(ProjectName GearTransmissionDemo)


# cmake modules
set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/../../../cmake)

# set policies
cmake_policy(SET CMP0074 NEW)


#----------------------------------------------------------------------
# Compiler standards

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)


# Required packages
find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(TBB REQUIRED)
find_package(glm REQUIRED)
find_package(GLEW REQUIRED)
find_package(GLFW3 REQUIRED)
FIND_PACKAGE(CUDA REQUIRED)
FIND_PACKAGE(CUB REQUIRED)
FIND_PACKAGE(OpenGL REQUIRED)
FIND_PACKAGE(Structure REQUIRED)


#include dir
include_directories(${OpenCV_INCLUDE_DIR})
include_directories(${EIGEN3_INCLUDE_DIR})
include_directories(${TBB_INCLUDE_DIR})
include_directories(${GLM_INCLUDE_DIR})
include_directories(${GLFW3_INCLUDE_DIR})
include_directories(${GLEW_INCLUDE_DIR})
include_directories(${STRUCTURE_INCLUDE_DIR})
include_directories(${AZUREKINECT_INCLUDE_DIR})


# local 
include_directories(${TRACKINGX_ROOT_DIR}/include)
include_directories(${TRACKINGX_ROOT_DIR}/include/camera)
include_directories(${TRACKINGX_ROOT_DIR}/include/detection)
include_directories(${TRACKINGX_ROOT_DIR}/include/kdtree)
include_directories(${TRACKINGX_ROOT_DIR}/include/loader)
include_directories(${TRACKINGX_ROOT_DIR}/include/nearest_neighbors)
include_directories(${TRACKINGX_ROOT_DIR}/include/pointcloud)
include_directories(${TRACKINGX_ROOT_DIR}/include/utils)
include_directories(${TRACKINGX_ROOT_DIR}/external/gl_support/include)
include_directories(${TRACKINGX_ROOT_DIR}/external/gl_ext)
include_directories(${TRACKINGX_ROOT_DIR}/external)
include_directories(${TRACKINGX_ROOT_DIR}/examples/demo_common)

# All output files are copied to bin
#set(EXECUTABLE_OUTPUT_PATH ${CMAKE_SOURCE_DIR}/../../../bin)
#set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_DEBUG ${CMAKE_SOURCE_DIR}/../../../bin)
#set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE ${CMAKE_SOURCE_DIR}/../../../bin)



#--------------------------------------------
# Source code


set(Demo_SRC
	demo_main.cpp
	DemoScene.h
	DemoScene.cpp
	GearBoxRenderer.cpp
	GearBoxRenderer.h
	../../demo_common/AssemblySequence.cpp
	../../demo_common/PartDatabase.cpp
	../../demo_common/AssemblySequence.h
	../../demo_common/ProcedureLoader.h
	../../demo_common/ProcedureLoader.cpp
	../../demo_common/ProcedureRenderer.h
	../../demo_common/ProcedureRenderer.cpp
	../../demo_common/PartDatabase.h
	../../demo_common/Sequence.h
)



#-----------------------------------------------------------------
#  SRC Groups, organize the tree

source_group(src FILES ${Demo_SRC})


#----------------------------------------------------------------------
# Compiler standards

add_compile_definitions(GLM_ENABLE_EXPERIMENTAL)



# Create the executable
add_executable(${ProjectName}
	${Demo_SRC}
)


set_target_properties (${ProjectName} PROPERTIES
    FOLDER Example
)

# dependencies
add_dependencies(${ProjectName} trackingx)
add_dependencies(${ProjectName} GLUtils)

#message(${TRACKINGX_ROOT_DIR})
# preporcessor properties
target_link_libraries(${ProjectName}  ${OpenCV_LIBS})
target_link_libraries(${ProjectName}  ${AZUREKINECT_LIBRARY})
target_link_libraries(${ProjectName}  ${TBB_LIBS})
target_link_libraries(${ProjectName}  ${GLEW_LIBS})
target_link_libraries(${ProjectName}  ${GLFW3_LIBS})
target_link_libraries(${ProjectName} optimized ${TRACKINGX_ROOT_DIR}/lib/trackingx.lib)
target_link_libraries(${ProjectName} debug ${TRACKINGX_ROOT_DIR}/lib/trackingxd.lib)
target_link_libraries(${ProjectName} debug  ${TRACKINGX_ROOT_DIR}/lib/GLUtilsd.lib )
target_link_libraries(${ProjectName} optimized  ${TRACKINGX_ROOT_DIR}/lib/GLUtils.lib )
target_link_libraries(${ProjectName} optimized  cudart.lib )
target_link_libraries(${ProjectName} debug  cudart.lib )
target_link_libraries(${ProjectName} ${GLEW_LIBS} ${GLEW_LIBS} ${GLFW3_LIBS} ${OPENGL_LIBS} ${OPENGL_LIBRARIES} )
target_link_libraries(${ProjectName} ${STRUCTURE_LIBRARY} )

#----------------------------------------------------------------------
# Pre-processor definitions
add_compile_definitions( _SILENCE_EXPERIMENTAL_FILESYSTEM_DEPRECATION_WARNING )

# add a "d" to all debug libraries
SET_TARGET_PROPERTIES(${ProjectName} PROPERTIES  DEBUG_POSTFIX "d")
SET_TARGET_PROPERTIES(${ProjectName} PROPERTIES LINK_FLAGS_RELEASE " /FORCE:MULTIPLE")
SET_TARGET_PROPERTIES(${ProjectName} PROPERTIES LINK_FLAGS_DEBUG "/FORCE:MULTIPLE ")
SET_TARGET_PROPERTIES(${ProjectName} PROPERTIES LINK_FLAGS "/FORCE:MULTIPLE")





