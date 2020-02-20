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
cmake_policy(SET CMP0015 NEW)
cmake_policy(SET CMP0074 NEW)
cmake_policy(SET CMP0010 NEW)



#----------------------------------------------------------------------
# Compiler standards

set(CMAKE_TRY_COMPILE_TARGET_TYPE "STATIC_LIBRARY")



# Check for CUDA support
include(CheckLanguage)
check_language(CUDA)
find_package(Cuda REQUIRED)


#if(WIN32)
#	if (NOT CMAKE_CUDA_COMPILER)
#		set(CMAKE_CUDA_COMPILER "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v9.0/bin/nvcc.exe")
#	endif()
#endif()

# Make CUDA optional, even if supported on host
if (CMAKE_CUDA_COMPILER OR CUDA_NVCC_EXECUTABLE)
	option(ENABLE_CUDA "Enable CUDA support" ON)
else()
	message(STATUS "CUDA compiler not found")
endif()

option(ENABLE_CUDA "Enable CUDA support" ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)


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
FIND_PACKAGE(Structure REQUIRED)

#include dir
include_directories(${OpenCV_INCLUDE_DIR})
include_directories(${EIGEN3_INCLUDE_DIR})
include_directories(${TBB_INCLUDE_DIR})
include_directories(${GLM_INCLUDE_DIR})
include_directories(${GLFW3_INCLUDE_DIR})
include_directories(${STRUCTURE_INCLUDE_DIR})

# local 
include_directories(${PROJECT_SOURCE_DIR}/include/camera)
include_directories(${PROJECT_SOURCE_DIR}/include/camera/cuda)
include_directories(${PROJECT_SOURCE_DIR}/include/detection)
include_directories(${PROJECT_SOURCE_DIR}/include/kdtree)
include_directories(${PROJECT_SOURCE_DIR}/include/loader)
include_directories(${PROJECT_SOURCE_DIR}/include/nearest_neighbors)
include_directories(${PROJECT_SOURCE_DIR}/include/pointcloud)
include_directories(${PROJECT_SOURCE_DIR}/include/utils)
include_directories(C:/SDK/StructureSDK/Libraries/Structure/Headers)




# All output files are copied to bin
set(EXECUTABLE_OUTPUT_PATH ${CMAKE_SOURCE_DIR}/bin)
set("CMAKE_RUNTIME_OUTPUT_DIRECTORY_DEBUG" "${CMAKE_SOURCE_DIR}/bin")
set("CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE" "${CMAKE_SOURCE_DIR}/bin")


#--------------------------------------------
# Camera options
option(ENABLE_REAL_SENSE "Enable Intel Real sense support" OFF)
option(ENABLE_STRUCTURE_CORE "Enable Structure Core support" ON)


#--------------------------------------------
# Source code


set(Header_SRC
	${PROJECT_SOURCE_DIR}/include/detection/FDTypes.h
	${PROJECT_SOURCE_DIR}/include/detection/FDTools.h
	${PROJECT_SOURCE_DIR}/include/detection/FDMatching.h
	${PROJECT_SOURCE_DIR}/include/detection/FDClustering.h
	${PROJECT_SOURCE_DIR}/include/detection/PCRegistration.h
	${PROJECT_SOURCE_DIR}/include/detection/nurmur.h
	${PROJECT_SOURCE_DIR}/include/kdtree/sort.h
	${PROJECT_SOURCE_DIR}/include/kdtree/dequeue.h
	${PROJECT_SOURCE_DIR}/include/kdtree/Cuda_KdTree.h
	${PROJECT_SOURCE_DIR}/include/kdtree/Cuda_Helpers.h
	${PROJECT_SOURCE_DIR}/include/kdtree/Cuda_Common.h
	${PROJECT_SOURCE_DIR}/include/kdtree/Cuda_Types.h
	${PROJECT_SOURCE_DIR}/include/loader/Types.h
	${PROJECT_SOURCE_DIR}/include/loader/SamplingTypes.h
	${PROJECT_SOURCE_DIR}/include/loader/Sampling.h
	${PROJECT_SOURCE_DIR}/include/loader/PointCloudProducerTypes.h
	${PROJECT_SOURCE_DIR}/include/loader/NoiseFilter.h
	${PROJECT_SOURCE_DIR}/include/loader/LoaderOBJ.h
	${PROJECT_SOURCE_DIR}/include/loader/FileUtilsExt.h
	${PROJECT_SOURCE_DIR}/include/nearest_neighbors/ICPKNN.h
	${PROJECT_SOURCE_DIR}/include/nearest_neighbors/ICPTransform.h
	${PROJECT_SOURCE_DIR}/include/nearest_neighbors/ICP.h
	${PROJECT_SOURCE_DIR}/include/pointcloud/PointCloudUtils.h
	${PROJECT_SOURCE_DIR}/include/pointcloud/PointCloudTransform.h
	${PROJECT_SOURCE_DIR}/include/pointcloud/PointCloudEval.h
	${PROJECT_SOURCE_DIR}/include/pointcloud/MatrixUtils.h
	${PROJECT_SOURCE_DIR}/include/pointcloud/Utils.h
	${PROJECT_SOURCE_DIR}/include/utils/TimeUtils.h
	${PROJECT_SOURCE_DIR}/include/utils/RandomGenerator.h
	${PROJECT_SOURCE_DIR}/include/utils/LogTypes.h
	${PROJECT_SOURCE_DIR}/include/utils/LogReaderWriter.h
	${PROJECT_SOURCE_DIR}/include/utils/ArgParser.h
	${PROJECT_SOURCE_DIR}/include/utils/FileUtilsX.h
	${PROJECT_SOURCE_DIR}/include/utils/MSVerCheck.h
	${PROJECT_SOURCE_DIR}/include/loader/ReaderWriter.h
	${PROJECT_SOURCE_DIR}/include/loader/ReaderWriterOBJ.h
	${PROJECT_SOURCE_DIR}/include/loader/ReaderWriterPLY.h
	${PROJECT_SOURCE_DIR}/include/loader/ImgReaderWriter.h
	${PROJECT_SOURCE_DIR}/include/loader/ImgReaderWriterPNG.h
	${PROJECT_SOURCE_DIR}/include/loader/DataReaderWriter.h
	



#if( ENABLE_STRUCTURE_CORE)
	${PROJECT_SOURCE_DIR}/include/camera/StructureCoreCaptureDevice.h
	${PROJECT_SOURCE_DIR}/include/camera/ICaptureDevice.h
	${PROJECT_SOURCE_DIR}/include/camera/ICaptureDeviceTypes.h
	${PROJECT_SOURCE_DIR}/include/camera/PointCloudProducer.h
	${PROJECT_SOURCE_DIR}/include/camera/CameraParameters.h

	
#endif()
#if( ENABLE_REAL_SENSE)
#	${PROJECT_SOURCE_DIR}/include/camera/SensorRealSense.h
#endif()
)


set( PointCloud_cuda_INC
	#${PROJECT_SOURCE_DIR}/include/camera/cuda/cuDeviceMemory.h
	#${PROJECT_SOURCE_DIR}/include/camera/cuda/cuPCU.h
	#${PROJECT_SOURCE_DIR}/include/camera/cuda/cuPCUImpl.h
	${PROJECT_SOURCE_DIR}/include/camera/cuda/cutil_math.h
	${PROJECT_SOURCE_DIR}/include/camera/cuda/cuDeviceMemory3f.h
	${PROJECT_SOURCE_DIR}/include/camera/cuda/cuPCU3f.h
	${PROJECT_SOURCE_DIR}/include/camera/cuda/cuPCUImpl3f.h
)


set(Detection_SRC
	detection/FDTypes.cpp
	detection/FDTools.cpp
	detection/FDMatching.cpp
	detection/FDClustering.cpp
	detection/PCRegistration.cpp
)

set(Cuda_kdtree_SRC
	kdtree/sort.cu
	kdtree/dequeue.cu
	kdtree/CudaErrorCheck.cu
	kdtree/Cuda_KdTree.cu
	kdtree/Cuda_Helpers.cpp
)

set(Loader_SRC
	loader/Sampling.cpp
	loader/NoiseFilter.cpp
	loader/LoaderOBJ.cpp
	loader/FileUtilsExt.cpp
	loader/ReaderWriterOBJ.cpp
	loader/ReaderWriterPLY.cpp
	loader/ImgReaderWriterPNG.cpp
	loader/DataReaderWriter.cpp
)


set(NN_SRC
	nearest_neighbors/ICPKNN.cpp
	nearest_neighbors/ICPTransform.cpp
	nearest_neighbors/ICP.cpp
)

set( PointCloud_SRC
	point_cloud/PointCloudUtils.cpp
	point_cloud/PointCloudTransform.cpp
	point_cloud/PointCloudEval.cpp
	point_cloud/MatrixUtils.cpp
#if( ENABLE_STRUCTURE_CORE)
	cam/StructureCoreCaptureDevice.cpp
	cam/PointCloudProducer.cpp
	cam/CameraParameters.cpp
#endif()
#if( ENABLE_REAL_SENSE)
#	cam/SensorRealSense.cpp
#endif()
)

set( PointCloud_cuda_SRC
	#cam/cuda/cuDeviceMemory.cpp
	#cam/cuda/cuPCU.cu
	#cam/cuda/cuPCUImpl.cu
	cam/cuda/cuDeviceMemory3f.cpp
	cam/cuda/cuPCU3f.cu
	cam/cuda/cuPCUImpl3f.cu
)


set( Shader_SRC
	shaders/simple_point_renderer.vs
	shaders/simple_point_renderer.fs
	shaders/lit_scene.vs
	shaders/lit_scene.fs
)

set( Utils_SRC
	utils/TimeUtils.cpp
	utils/RandomGenerator.cpp
	utils/LogReaderWriter.cpp
	utils/ArgParser.cpp
	utils/FileUtilsX.cpp
)



#-----------------------------------------------------------------
#  SRC Groups, organize the tree

source_group(Detection FILES ${Detection_SRC})
source_group(Cuda kd-tree FILES ${Cuda_kdtree_SRC})
source_group(Loader FILES ${Loader_SRC})
source_group(NN FILES ${NN_SRC})
source_group(PointCloud FILES ${PointCloud_SRC})
source_group(Shader FILES ${Shader_SRC})
source_group(Utils FILES ${Utils_SRC})
source_group("PointCloud/Cuda" FILES ${PointCloud_cuda_SRC} )
source_group(Cuda FILES ${PointCloud_cuda_INC})
#----------------------------------------------------------------------
# Compiler standards

add_compile_definitions(GLM_ENABLE_EXPERIMENTAL)


# Create the tracking expert library

add_library(trackingx
	${Header_SRC}
	${Detection_SRC}
	${Cuda_kdtree_SRC}
	${Loader_SRC}
	${NN_SRC}
	${PointCloud_SRC}
	${Shader_SRC}
	${Utils_SRC}
	${PointCloud_cuda_SRC}
	${PointCloud_cuda_INC}
)

# preporcessor properties

target_link_libraries(trackingx  ${OpenCV_LIBS})
target_link_libraries(trackingx  ${TBB_LIBS})
target_link_libraries(trackingx  ${GLEW_LIBS})
target_link_libraries(trackingx  ${GLFW3_LIBS})

target_link_libraries(trackingx  "C:/SDK/StructureSDK/Libraries/Structure/Windows/x86_64/Structure.lib")


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
