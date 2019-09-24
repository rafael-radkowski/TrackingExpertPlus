# TrackingExpert+ cmake file. 
# /src-files. 
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


#if(WIN32)
#	if (NOT CMAKE_CUDA_COMPILER)
#		set(CMAKE_CUDA_COMPILER "$ENV{PROGRAMFILES}/NVIDIA GPU Computing Toolkit/CUDA/v9.0/bin/nvcc.exe")
#	endif()
#endif()


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

# local 
include_directories(./loader)
include_directories(./detection)
include_directories(./kdtree)
include_directories(./cuda)
include_directories(./camera)
include_directories(./graphics)
include_directories(./utils)
include_directories(./point_cloud)


# All output files are copied to bin
set(EXECUTABLE_OUTPUT_PATH ${CMAKE_SOURCE_DIR}/bin)
set("CMAKE_RUNTIME_OUTPUT_DIRECTORY_DEBUG" "${CMAKE_SOURCE_DIR}/bin")
set("CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE" "${CMAKE_SOURCE_DIR}/bin")



#--------------------------------------------
# Source code


set(Detection_SRC
	detection/FDTypes.cpp
	detection/FDTypes.h
	detection/FDTools.cpp
	detection/FDTools.h
	detection/FDMatching.h
	detection/FDMatching.cpp
	detection/FDClustering.h
	detection/FDClustering.cpp
	detection/nurmur.h
)

set(Cuda_kdtree_SRC
	kdtree/sort.h
	kdtree/sort.cu
	kdtree/dequeue.h
	kdtree/dequeue.cu
	kdtree/CudaErrorCheck.cu
	kdtree/Cuda_KdTree.h
	kdtree/Cuda_KdTree.cu
	kdtree/Cuda_Helpers.h
	kdtree/Cuda_Helpers.cpp
	kdtree/Cuda_Common.h
)

set(Loader_SRC
	loader/Types.h
	loader/SamplingTypes.h
	loader/Sampling.h
	loader/Sampling.cpp
	loader/PointCloudProducerTypes.h
	loader/NoiseFilter.h
	loader/NoiseFilter.cpp
	loader/LoaderOBJ.h
	loader/LoaderOBJ.cpp
	loader/FileUtils.h
	loader/FileUtils.cpp
)


set(NN_SRC
	nearest_neighbors/NearestNeighbors.h
	nearest_neighbors/NearestNeighbors.cpp
	nearest_neighbors/ICPTransform.h
	nearest_neighbors/ICPTransform.cpp
)

set( PointCloud_SRC
	point_cloud/PointCloudUtils.h
	point_cloud/PointCloudUtils.cpp
	point_cloud/PointCloudTransform.h
	point_cloud/PointCloudTransform.cpp
	point_cloud/PointCloudEval.h
	point_cloud/PointCloudEval.cpp
	point_cloud/MatrixUtils.h
	point_cloud/MatrixUtils.cpp
	point_cloud/Utils.h
)


set( Shader_SRC
	shaders/simple_point_renderer.vs
	shaders/simple_point_renderer.fs
	shaders/lit_scene.vs
	shaders/lit_scene.fs
)

set( Utils_SRC
	utils/TimeUtils.h
	utils/TimeUtils.cpp
	utils/RandomGenerator.h
	utils/RandomGenerator.cpp
	utils/LogTypes.h
	utils/LogReaderWriter.h
	utils/LogReaderWriter.cpp
	utils/ArgParser.h
	utils/ArgParser.cpp
)



#-----------------------------------------------------------------
#  SRC Groups, organize the tree

source_group(Detection FILES ${Detection_SRC})
source_group(Cuda kd-tree FILES ${Cuda_kdtree_SRC})
source_group(Loader_SRC FILES ${Loader_SRC})
source_group(NN_SRC FILES ${NN_SRC})
source_group(PointCloud_SRC FILES ${PointCloud_SRC})
source_group(Shader_SRC FILES ${Shader_SRC})
source_group(Utils_SRC FILES ${Utils_SRC})


#----------------------------------------------------------------------
# Compiler standards

add_compile_definitions(GLM_ENABLE_EXPERIMENTAL)


# Create the tracking expert library

add_library(trackingx
	${Detection_SRC}
	${Cuda_kdtree_SRC}
	${Loader_SRC}
	${NN_SRC}
	${PointCloud_SRC}
	${Shader_SRC}
	${Utils_SRC}
)


# preporcessor properties

target_link_libraries(trackingx  ${OpenCV_LIBS})
target_link_libraries(trackingx  ${TBB_LIBS})
target_link_libraries(trackingx  ${GLEW_LIBS})
target_link_libraries(trackingx  ${GLFW3_LIBS})


#----------------------------------------------------------------------
# Pre-processor definitions

# add a "d" to all debug libraries
SET_TARGET_PROPERTIES(trackingx PROPERTIES  DEBUG_POSTFIX "d")
SET_TARGET_PROPERTIES(trackingx PROPERTIES LINK_FLAGS_RELEASE " /FORCE:MULTIPLE")
SET_TARGET_PROPERTIES(trackingx PROPERTIES LINK_FLAGS_DEBUG "/FORCE:MULTIPLE ")
SET_TARGET_PROPERTIES(trackingx PROPERTIES LINK_FLAGS "/FORCE:MULTIPLE")



#----------------------------------------------------------------------
# Cuda standards
if(ENABLE_CUDA)

	target_link_libraries(trackingx
		CUB::CUB
	)
	set_target_properties(trackingx PROPERTIES
		CUDA_SEPARABLE_COMPILATION ON
	)
	# POSITION_INDEPENDENT_CODE needs to be set to link as a library
	set_target_properties(trackingx PROPERTIES
		POSITION_INDEPENDENT_CODE ON
	)


	# Need to set this property so CUDA functions can be linked to targets that link afrl library
	set_property(TARGET trackingx PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)

	# Target compute capability 5.0
	target_compile_options(trackingx PUBLIC $<$<COMPILE_LANGUAGE:CUDA>:-gencode arch=compute_50,code=sm_50>)

	# Device debug info in debug mode
	set(CMAKE_CUDA_FLAGS_DEBUG "${CMAKE_CUDA_FLAGS_DEBUG} -g -G")
	set(CMAKE_CUDA_FLAGS_RELWITHDEBINFO "${CMAKE_CUDA_FLAGS_RELWITHDEBINFO} --generate-line-info")

endif()






################################################################
