# @file  FindStructure.cmake
# @brief Cmake script to find the Structure libraries, https://structure.io/developers
#
# The module finds the Structure library Structure.lib for Windows using the environment variable
# Strucure_ROOT to find it. 
#
# Use 
#	find_package(Structure)
# in a cmake file to find it. 
# Note that this file was tested with Eigen 3.3.7
#
# 1. Setup
# Set the environment variable STRUCTURE_DIR pointing to the main folder containing the structure SDK 3 folder, e.g., C:\SDK\StructureSDK
# The folder must contain the subfolder Libraries
#
# 2. Variables
# The following variables are produced and usable in a CMakeList.txt file after cmake is done: 
#
#  STRUCTURE_FOUND - system has eigen lib with correct version
#  STRUCTURE_INCLUDE_DIR - the structure lib include directory
#  STRUCTURE_LIBRARY - macro pointing to the structure library
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
# Oct 22, 2019
# rafael@iastate.edu
#
# MIT License
#---------------------------------------------------------------------
#
# Last edits:
#
# 




##----------------------------------------------------------
## Init the search
set(STRUCTURE_FOUND FALSE)


# 1. Read the environment variable
if (NOT STRUCTURE_DIR)
  if (DEFINED ENV{Structure_DIR})
    set (STRUCTURE_DIR "$ENV{Structure_DIR}" CACHE PATH "Installation prefix of the Structure Library." FORCE)
  elseif (DEFINED ENV{Structure_ROOT})
    set (STRUCTURE_DIR "$ENV{Structure_ROOT}" CACHE PATH "Installation prefix of the Structure Library." FORCE)
  elseif (DEFINED ENV{STRUCTURE_DIR})
    set (STRUCTURE_DIR "$ENV{STRUCTURE_DIR}" CACHE PATH "Installation prefix of the Structure Library." FORCE)
  elseif (DEFINED ENV{STRUCTURE_ROOT})
    set (STRUCTURE_DIR "$ENV{STRUCTURE_ROOT}" CACHE PATH "Installation prefix of the Structure Library." FORCE)
  endif ()
endif ()




# 2.  Find the include directory
if(NOT STRUCTURE_INCLUDE_DIR AND STRUCTURE_DIR)
find_path(STRUCTURE_INCLUDE_DIR 
	NAMES ST/CaptureSession.h 
	HINTS
	${STRUCTURE_DIR}/Libraries/Structure/Headers/
	PATH_SUFFIXES ST 
)
endif() # if(NOT STRUCTURE_INCLUDE_DIR)

# 3. Find the library path
if(NOT STRUCTURE_LIBRARY AND STRUCTURE_DIR)

find_file(STRUCTURE_LIBRARY
	NAMES Structure.lib
	PATHS 
	${STRUCTURE_DIR}/Libraries/Structure/Windows/x86_64
	)
	
endif()
	
if(STRUCTURE_LIBRARY)
	set(STRUCTURE_FOUND TRUE  CACHE STRING "Found strcture")
	message(STATUS "[Structure] Found structure sdk at ${STRUCTURE_DIR}" )
endif()
