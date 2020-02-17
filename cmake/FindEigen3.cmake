# @file  FindEigen.cmake
# @brief Cmake script to find the Eigen3 library (http://eigen.tuxfamily.org/index.php)
#
# The module finds the Eigen 3 library by reading an Eigen 3 environment variable and 
# looking whether or not the Eigen 3 library is installed at this location. 
#
# Use 
#	find_package(Eigen3)
# in a cmake file to find it. 
# Note that this file was tested with Eigen 3.3.7
#
# 1. Setup
# Set the environment variable Eigen3_DIR pointing to the main Eigen 3 folder, e.g., C:\SDK\Eigen3-3.3.7
#
# 2. Variables
# The following variables are produced and usable in a CMakeList.txt file after cmake is done: 
#
#  EIGEN3_FOUND - system has eigen lib with correct version
#  EIGEN3_INCLUDE_DIR - the eigen include directory
#  EIGEN3_VERSION - eigen version
#
#
# #  # tested with:
# - Eigen 3.3.7:  MSVC 15,2017
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
# Feb 14, 2020, RR
# - Changed Eigen_include_dir search path
# 




##----------------------------------------------------------
## Init the search
set(EIGEN3_FOUND FALSE)


# 1. Read the environment variable
if (NOT EIGEN3_DIR)
  if (DEFINED ENV{Eigen3_DIR})
    set (EIGEN3_DIR "$ENV{Eigen3_DIR}" CACHE PATH "Installation prefix of OpenCV Library." FORCE)
  elseif (DEFINED ENV{Eigen3_ROOT})
    set (EIGEN3_DIR "$ENV{Eigen3_ROOT}" CACHE PATH "Installation prefix of OpenCV Library." FORCE)
  elseif (DEFINED ENV{EIGEN3_ROOT})
    set (EIGEN3_DIR "$ENV{EIGEN3_ROOT}" CACHE PATH "Installation prefix of OpenCV Library." FORCE)
  elseif (DEFINED ENV{EIGEN3_DIR})
    set (EIGEN3_DIR "$ENV{EIGEN3_DIR}" CACHE PATH "Installation prefix of OpenCV Library." FORCE)
  endif ()
endif ()


if(NOT EIGEN3_INCLUDE_DIR)


# 2. Find the root folder
# check the path 
find_file(_find_eigen_ 
	NAMES Eigen 
	PATHS
	${EIGEN3_DIR}
	PATH_SUFFIXES eigen3 eigen
)

# ----------------------------------------------------------------------------
# 3. Find the include directory
set(__INCLUDE_DIRS 
	${EIGEN3_DIR}/Eigen
	${EIGEN3_DIR}/eigen
	${EIGEN3_DIR}
)



if(_find_eigen_)
	find_path(
		EIGEN3_INCLUDE_DIR Eigen/Eigen
		PATHS ${__INCLUDE_DIRS}
	)
endif(_find_eigen_)
unset(_find_eigen_ CACHE)

endif() # if(NOT EIGEN3_INCLUDE_DIR)



# ----------------------------------------------------------------------------
# 4. Find the version

if(EIGEN3_DIR)

find_file(__find_version "Macros.h"  PATHS  "${EIGEN3_DIR}/Eigen/src/Core/util"  )
if(__find_version)
	SET(EIGEN_VERSION_FILE "${EIGEN3_DIR}/Eigen/src/Core/util/Macros.h")
	file(STRINGS "${EIGEN_VERSION_FILE}" EIGEN_VERSION_PARTS REGEX "#define EIGEN_[A-Z]+_+VERSION+[ ]+" )
	string(REGEX REPLACE ".+EIGEN_WORLD_VERSION[ ]+([0-9]+).*" "\\1" EIGEN_VERSION_MAJOR "${EIGEN_VERSION_PARTS}")
	string(REGEX REPLACE ".+EIGEN_MAJOR_VERSION[ ]+([0-9]+).*" "\\1" EIGEN_VERSION_MINOR "${EIGEN_VERSION_PARTS}")
	string(REGEX REPLACE ".+EIGEN_MINOR_VERSION[ ]+([0-9]+).*" "\\1" EIGEN_VERSION_PATCH "${EIGEN_VERSION_PARTS}")
	set(EIGEN_VERSION_PLAIN "${EIGEN_VERSION_MAJOR}${EIGEN_VERSION_MINOR}${EIGEN_VERSION_PATCH}")
	set(EIGEN_VERSION "${EIGEN_VERSION_MAJOR}.${EIGEN_VERSION_MINOR}.${EIGEN_VERSION_PATCH}" CACHE PATH "Installed version")
	mark_as_advanced(EIGEN_VERSION)
endif()
unset(__find_version CACHE)

endif()



if(EIGEN3_INCLUDE_DIR)
	message(STATUS "[FindEigen3] - Found Eigen3 in " ${EIGEN3_DIR} " Version " ${EIGEN_VERSION})
	set(EIGEN3_FOUND TRUE CACHE PATH "found eigen3")
	mark_as_advanced(EIGEN3_FOUND)
else(EIGEN3_INCLUDE_DIR)
	message( "[FindEigen3] - ERROR did not find the Eigen3 include dir")
endif(EIGEN3_INCLUDE_DIR)



