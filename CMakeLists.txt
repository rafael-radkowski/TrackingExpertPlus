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
# Feb 5, 2020, RR
# - Added submodule support for glGLutils4
# 08/02/21, 
# - RR: Added feature toggle ENABLE_CPF_GPU to add/remove detection GPU support. Runs for a specific cuda version only. 
# 


cmake_minimum_required(VERSION 2.6)

# Note that this line must be added before project. 
# The cuda compiler will not work properly otherwise. 
set(CMAKE_TRY_COMPILE_TARGET_TYPE "STATIC_LIBRARY")


# enable visual studio folders
set_property(GLOBAL PROPERTY USE_FOLDERS ON)


set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)



# The project name
project(TrackingExpertPlus CUDA CXX)

# set policies
cmake_policy(SET CMP0015 NEW)
cmake_policy(SET CMP0074 NEW)


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



# GLUtils Submodule
# create the directory for all dependencies and download the dependencies.
find_path( GLSUPPORT_DIR NAMES external/gl_support/README.md)

if( (NOT GLSUPPORT_DIR) OR (NOT EXISTS ${GLSUPPORT_DIR}))
	file(MAKE_DIRECTORY "./external/gl_support")

	execute_process(COMMAND git submodule update --init 
					WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})

	set(GLSUPPORT_DIR ${CMAKE_CURRENT_SOURCE_DIR}/external/gl_support
					CACHE PATH "gl_support dir" FORCE)


		
	
else()
  # update the submodule to the latest version.
  #execute_process(COMMAND git submodule update --remote --merge
  #	  				WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})

endif()


# GLM Submodule
# create the directory for all dependencies and download the dependencies.
find_path( GLM_DIR NAMES external/glm/readme.md)

if( (NOT GLM_DIR) OR (NOT EXISTS ${GLM_DIR}))
	file(MAKE_DIRECTORY "./external/glm")

	execute_process(COMMAND git submodule update --init 
					WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})

	set(GLM_DIR ${CMAKE_CURRENT_SOURCE_DIR}/external/glm
					CACHE PATH "glm dir" FORCE)


	set( ENV{GLM_DIR_LOCAL} ${CMAKE_CURRENT_SOURCE_DIR}/external/glm )	
	
else()
  # update the submodule to the latest version.
  #execute_process(COMMAND git submodule update --remote --merge
  #	  				WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})

endif()



#---------------------------------------------------------------------------------
# Build local glfw version
option(TRAKINGX_BUILD_GLFW "TrackingExpert: Build glfw library" ON)
if(TRAKINGX_BUILD_GLFW)
	set( ENV{GLFW_DIR_LOCAL} ${CMAKE_CURRENT_SOURCE_DIR}/external/glfw )	
endif()

#---------------------------------------------------------------------------------
# Build local glew version
option(TRAKINGX_BUILD_GLEW "TrackingExpert: Build glew library" ON)
if(TRAKINGX_BUILD_GLEW)
	set( ENV{GLEW_DIR_LOCAL} ${CMAKE_CURRENT_SOURCE_DIR}/external/glew )	
endif()


#---------------------------------------------------------------------------------
# Build with demo tool
option(TRAKINGX_BUILD_DEMO "TrackingExpert: Build with examples" ON)


#message(${CMAKE_CURRENT_SOURCE_DIR})
set( TRACKINGX_ROOT_DIR ${CMAKE_CURRENT_SOURCE_DIR}) 


file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/tests/Test.cmd
     DESTINATION ${CMAKE_CURRENT_SOURCE_DIR}/bin/)
	 

#--------------------------------------------
# Camera options
option(ENABLE_REAL_SENSE "Enable Intel Real sense support" OFF)
option(ENABLE_STRUCTURE_CORE "Enable Structure Core support" OFF)
option(ENABLE_AZURE_KINECT "Enable Structure Core support" OFF)

#--------------------------------------------
# Features toggles
option(ENABLE_CPF_GPU "Enable CPF GPU support" OFF)

################################################################
add_subdirectory(src)
add_subdirectory(tests)
add_subdirectory(external)

if(TRAKINGX_BUILD_DEMO)
	add_subdirectory(examples)
endif()
