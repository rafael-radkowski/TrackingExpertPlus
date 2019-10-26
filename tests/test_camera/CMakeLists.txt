# TrackingExpert+ cmake file. 
# /camera test cmake file. 
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
include_directories("C:/SDK/StructureSDK/Libraries/Structure/Headers")


# All output files are copied to bin
set(EXECUTABLE_OUTPUT_PATH ${CMAKE_SOURCE_DIR}/bin)
set("CMAKE_RUNTIME_OUTPUT_DIRECTORY_DEBUG" "${CMAKE_SOURCE_DIR}/bin")
set("CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE" "${CMAKE_SOURCE_DIR}/bin")



#--------------------------------------------
# Source code


set(test_SRC
	main_camera_test.cpp
)



#-----------------------------------------------------------------
#  SRC Groups, organize the tree

source_group(src FILES ${test_SRC})


#----------------------------------------------------------------------
# Compiler standards

add_compile_definitions(GLM_ENABLE_EXPERIMENTAL)


# Create the tracking expert library
set( ProjectName test_camera)
add_executable(${ProjectName}
	${test_SRC}
)


set_target_properties (${ProjectName} PROPERTIES
    FOLDER Tests
)


# preporcessor properties

target_link_libraries(${ProjectName}  ${OpenCV_LIBS})
target_link_libraries(${ProjectName}  ${TBB_LIBS})
target_link_libraries(${ProjectName}  ${GLEW_LIBS})
target_link_libraries(${ProjectName}  ${GLFW3_LIBS})
target_link_libraries(${ProjectName}  optimized ${PROJECT_SOURCE_DIR}/lib/trackingx.lib)
target_link_libraries(${ProjectName}  debug ${PROJECT_SOURCE_DIR}/lib/trackingxd.lib)
target_link_libraries(${ProjectName} debug  ${PROJECT_SOURCE_DIR}/lib/GLUtilsd.lib )
target_link_libraries(${ProjectName} optimized  ${PROJECT_SOURCE_DIR}/lib/GLUtils.lib )
target_link_libraries(${ProjectName} optimized  cudart.lib )
target_link_libraries(${ProjectName} debug  cudart.lib )
target_link_libraries(${ProjectName} ${GLEW_LIBS} ${GLEW_LIBS} ${GLFW3_LIBS} ${OPENGL_LIBS} ${OPENGL_LIBRARIES} )
target_link_libraries(${ProjectName} "C:/SDK/StructureSDK/Libraries/Structure/Windows/x86_64/Structure.lib" )


#----------------------------------------------------------------------
# Pre-processor definitions

# add a "d" to all debug libraries
SET_TARGET_PROPERTIES(${ProjectName} PROPERTIES  DEBUG_POSTFIX "d")
SET_TARGET_PROPERTIES(${ProjectName} PROPERTIES LINK_FLAGS_RELEASE " /FORCE:MULTIPLE")
SET_TARGET_PROPERTIES(${ProjectName} PROPERTIES LINK_FLAGS_DEBUG "/FORCE:MULTIPLE ")
SET_TARGET_PROPERTIES(${ProjectName} PROPERTIES LINK_FLAGS "/FORCE:MULTIPLE")



#----------------------------------------------------------------------
# Cuda standards
if(ENABLE_CUDA)

	target_link_libraries(${ProjectName}
		CUB::CUB
		 ${PROJECT_SOURCE_DIR}/lib/trackingx.lib
	)
	set_target_properties(${ProjectName} PROPERTIES
		CUDA_SEPARABLE_COMPILATION ON
	)
	# POSITION_INDEPENDENT_CODE needs to be set to link as a library
	set_target_properties(${ProjectName} PROPERTIES
		POSITION_INDEPENDENT_CODE ON
	)


	# Need to set this property so CUDA functions can be linked to targets that link afrl library
	set_property(TARGET ${ProjectName} PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)

	# Target compute capability 5.0
	target_compile_options(${ProjectName} PUBLIC $<$<COMPILE_LANGUAGE:CUDA>:-gencode arch=compute_50,code=sm_50>)

	# Device debug info in debug mode
	set(CMAKE_CUDA_FLAGS_DEBUG "${CMAKE_CUDA_FLAGS_DEBUG} -g -G")
	set(CMAKE_CUDA_FLAGS_RELWITHDEBINFO "${CMAKE_CUDA_FLAGS_RELWITHDEBINFO} --generate-line-info")

endif()






################################################################
