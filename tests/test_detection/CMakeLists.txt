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


#----------------------------------------------------------------------
# Compiler standards

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

# Check for CUDA support
include(CheckLanguage)
check_language(CUDA)
find_package(Cuda REQUIRED)



# Make CUDA optional, even if supported on host
if (CMAKE_CUDA_COMPILER OR CUDA_NVCC_EXECUTABLE)
	option(ENABLE_CUDA "Enable CUDA support" ON)
else()
	message(STATUS "CUDA compiler not found")
endif()
option(ENABLE_CUDA "Enable CUDA support" ON)

# Enable CUDA if selected
if(ENABLE_CUDA)
	enable_language(CUDA)
	set(CMAKE_CUDA_STANDARD 14)
	set(CMAKE_CUDA_STANDARD_REQUIRED ON)
	find_package(CUB REQUIRED)
endif()


# Required packages
find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(TBB REQUIRED)
find_package(GLM REQUIRED)
find_package(GLEW REQUIRED)
find_package(GLFW3 REQUIRED)
FIND_PACKAGE(Cuda REQUIRED)
FIND_PACKAGE(Cub REQUIRED)
FIND_PACKAGE(OpenGL REQUIRED)

#include dir
include_directories(${OpenCV_INCLUDE_DIR})
include_directories(${EIGEN3_INCLUDE_DIR})
include_directories(${TBB_INCLUDE_DIR})
include_directories(${GLM_INCLUDE_DIR})
include_directories(${GLFW3_INCLUDE_DIR})
include_directories(${GLEW_INCLUDE_DIR})

# local 
include_directories(${PROJECT_SOURCE_DIR}/include/camera)
include_directories(${PROJECT_SOURCE_DIR}/include/detection)
include_directories(${PROJECT_SOURCE_DIR}/include/kdtree)
include_directories(${PROJECT_SOURCE_DIR}/include/loader)
include_directories(${PROJECT_SOURCE_DIR}/include/nearest_neighbors)
include_directories(${PROJECT_SOURCE_DIR}/include/pointcloud)
include_directories(${PROJECT_SOURCE_DIR}/include/utils)
include_directories(${PROJECT_SOURCE_DIR}/external/gl_support/include)
include_directories(${PROJECT_SOURCE_DIR}/external/gl_ext)


# All output files are copied to bin
set(EXECUTABLE_OUTPUT_PATH ${CMAKE_SOURCE_DIR}/bin)
set("CMAKE_RUNTIME_OUTPUT_DIRECTORY_DEBUG" "${CMAKE_SOURCE_DIR}/bin")
set("CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE" "${CMAKE_SOURCE_DIR}/bin")



#--------------------------------------------
# Source code


set(test_SRC
	registration_test.cpp
	registration_analysis.h
	registration_analysis.cpp
)



#-----------------------------------------------------------------
#  SRC Groups, organize the tree

source_group(main_tests FILES ${test_SRC})


#----------------------------------------------------------------------
# Compiler standards

add_compile_definitions(GLM_ENABLE_EXPERIMENTAL)


# Create the tracking expert library

add_executable(test_trackingx_detect
	${test_SRC}
)


set_target_properties (test_trackingx_detect PROPERTIES
    FOLDER Tests
)


# preporcessor properties

target_link_libraries(test_trackingx_detect  ${OpenCV_LIBS})
target_link_libraries(test_trackingx_detect  ${TBB_LIBS})
target_link_libraries(test_trackingx_detect  ${GLEW_LIBS})
target_link_libraries(test_trackingx_detect  ${GLFW3_LIBS})
target_link_libraries(test_trackingx_detect  ${PROJECT_SOURCE_DIR}/lib/trackingx.lib)
target_link_libraries(test_trackingx_detect debug  ${PROJECT_SOURCE_DIR}/lib/GLUtilsd.lib )
target_link_libraries(test_trackingx_detect optimized  ${PROJECT_SOURCE_DIR}/lib/GLUtils.lib )
target_link_libraries(test_trackingx_detect optimized  cudart.lib )
target_link_libraries(test_trackingx_detect ${GLEW_LIBS} ${GLEW_LIBS} ${GLFW3_LIBS} ${OPENGL_LIBS} ${OPENGL_LIBRARIES} )

#----------------------------------------------------------------------
# Pre-processor definitions

# add a "d" to all debug libraries
SET_TARGET_PROPERTIES(test_trackingx_detect PROPERTIES  DEBUG_POSTFIX "d")
SET_TARGET_PROPERTIES(test_trackingx_detect PROPERTIES LINK_FLAGS_RELEASE " /FORCE:MULTIPLE")
SET_TARGET_PROPERTIES(test_trackingx_detect PROPERTIES LINK_FLAGS_DEBUG "/FORCE:MULTIPLE ")
SET_TARGET_PROPERTIES(test_trackingx_detect PROPERTIES LINK_FLAGS "/FORCE:MULTIPLE")



#----------------------------------------------------------------------
# Cuda standards
if(ENABLE_CUDA)

	target_link_libraries(test_trackingx_detect
		CUB::CUB
		 ${PROJECT_SOURCE_DIR}/lib/trackingx.lib
	)
	set_target_properties(test_trackingx_detect PROPERTIES
		CUDA_SEPARABLE_COMPILATION ON
	)
	# POSITION_INDEPENDENT_CODE needs to be set to link as a library
	set_target_properties(test_trackingx_detect PROPERTIES
		POSITION_INDEPENDENT_CODE ON
	)


	# Need to set this property so CUDA functions can be linked to targets that link afrl library
	set_property(TARGET test_trackingx_detect PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)

	# Target compute capability 5.0
	target_compile_options(test_trackingx_detect PUBLIC $<$<COMPILE_LANGUAGE:CUDA>:-gencode arch=compute_50,code=sm_50>)

	# Device debug info in debug mode
	set(CMAKE_CUDA_FLAGS_DEBUG "${CMAKE_CUDA_FLAGS_DEBUG} -g -G")
	set(CMAKE_CUDA_FLAGS_RELWITHDEBINFO "${CMAKE_CUDA_FLAGS_RELWITHDEBINFO} --generate-line-info")

endif()






################################################################
