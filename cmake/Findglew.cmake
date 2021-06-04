# Locate the glew files 
#
# 1. Setup
# The script requires the environment variables GLEW_DIR to point to the path [your path]/opencv
# So the the GLEW_DIR environment variable, e.g., C:\SDK\glew-2.1.0
#
# 2. Variables
# The following variables are produced and usable in a CMakeList.txt file after cmake is done
# GLEW_INCLUDE_DIR  the location of the gl/glew.h header file. 
# GLEW_FOUND        True, if glew can be found. 
# GLEW_LIBs		    the names of the glew32.lib and glew32d.lib libraries
#
# To help locate the library and include file, you can define a 
# variable called GLEW_ROOT which points to the root of the glfw library 
# installation.
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
# Feb 5, 2020, RR
# - Adapted the definitions so that the file works with a local glew copy
# 

set(GLEW_FOUND FALSE)


# default search dirs
set( _glew_SEARCH_DIRS 
  "/usr/include"
  "/usr/local/include"
  "$ENV{PROGRAMFILES}/glew/include" 
  "C:/SDK/glew/include"
  )
  

set (GLEW_DIR_LOCAL_ FALSE)
 

# 1. Read the environment variable
if (NOT _glew_DIR)
   if (DEFINED ENV{GLEW_DIR_LOCAL})
    set (_glew_DIR "$ENV{GLEW_DIR_LOCAL}" CACHE PATH "Installation prefix of Glew Library." FORCE)
	set (GLEW_DIR_LOCAL_ TRUE)
  elseif (DEFINED ENV{GLEW_DIR})
    set (_glew_DIR "$ENV{GLEW_DIR}" CACHE PATH "Installation prefix of Glew Library." FORCE)
  elseif (DEFINED ENV{Glew_DIR})
    set (_glew_DIR "$ENV{Glew_DIR}" CACHE PATH "Installation prefix of Glew Library." FORCE)
  elseif (DEFINED ENV{GLEW_ROOT})
    set (_glew_DIR "$ENV{GLEW_ROOT}" CACHE PATH "Installation prefix of Glew Library." FORCE)
  elseif (DEFINED ENV{Glew_ROOT})
    set (_glew_DIR "$ENV{Glew_ROOT}" CACHE PATH "Installation prefix of Glew Library." FORCE)
  endif ()
endif ()

if(_glew_DIR  )
	set(GLEW_DIR ${_glew_DIR} CACHE PATH "GLEW installatin path")
endif()

# 2. If environment variable is nto defined, search at typical locations
if (NOT _glew_DIR)
	FIND_PATH(GLEW_DIR "gl/glew.h"
		PATHS ${_glew_SEARCH_DIRS} )
endif()
unset(_glew_DIR CACHE)


# 3. Find the include directory
find_file(__find_glew "gl/glew.h" ${GLEW_DIR}/include)
if(__find_glew)
	find_path(GLEW_INCLUDE_DIR "gl/glew.h"
	PATHS ${GLEW_DIR}/include)
	#message(${GLEW_INCLUDE_DIR})
	
	if(NOT GLEW_INCLUDE_DIR)
		message("[FindGLEW] - ERROR: did not find the GLEW include dir")
	endif()
	
endif()
unset(__find_glew CACHE)

# 4. Find the library directory
set( _glew_LIB_SEARCH_DIRS 
  "${GLEW_DIR}/lib/Release/x64"
  "${GLEW_DIR}/lib"
  "${GLEW_DIR}/lib/Release" 
  "${GLEW_DIR}/lib/Debug/x64"
  "${GLEW_DIR}/lib/Debug" 
  )
  
if(GLEW_DIR AND GLEW_DIR_LOCAL_ MATCHES FALSE)
	set(GLEW_LIBS_LIST )

	find_file(__find_glew_lib "glew32.lib" PATHS ${_glew_LIB_SEARCH_DIRS})
	if(__find_glew_lib)
		find_library(GLEW_LIBRARY_RELEASE "glew32.lib"  PATHS ${_glew_LIB_SEARCH_DIRS})
		if(GLEW_LIBRARY_RELEASE)
			set(GLEW_LIBRARY_RELEASE_CONF "optimized" ${GLEW_LIBRARY_RELEASE})
			list(APPEND GLEW_LIBS_LIST ${GLEW_LIBRARY_RELEASE_CONF})
		endif()
	else(__find_glew_lib)
		message("[FindGLEW] - ERROR: cannot find glew32.lib")
	endif(__find_glew_lib)
	unset(__find_glew_lib CACHE)

	find_file(__find_glew_lib "glew32d.lib" PATHS ${_glew_LIB_SEARCH_DIRS})
	if(__find_glew_lib)
		find_library(GLEW_LIBRARY_DEBUG "glew32d.lib"  PATHS ${_glew_LIB_SEARCH_DIRS})
		if(GLEW_LIBRARY_DEBUG)
			set(GLEW_LIBRARY_DEBUG "debug" ${GLEW_LIBRARY_DEBUG})
			list( APPEND GLEW_LIBS_LIST ${GLEW_LIBRARY_DEBUG})
		endif()
	else(__find_glew_lib)
		message( STATUS "[FindGLEW] - WARNING: cannot find glew32d.lib. Use glew32.lib instead.")
		## add the release library to the debug config so that the debug config has an entry. 
		set(GLEW_LIBRARY_DEBUG "debug" ${GLEW_LIBRARY_RELEASE})
		list( APPEND GLEW_LIBS_LIST ${GLEW_LIBRARY_DEBUG})
	endif(__find_glew_lib)
	unset(__find_glew_lib CACHE)
	
	set(GLEW_LIBS ${GLEW_LIBS_LIST} CACHE STRING "Glew libs")
else()
	set(GLEW_LIBS ${GLEW_DIR}/lib/glew.lib CACHE STRING "Glew libs")
endif ()


# 5. Read the version number
find_file(__find_version "version"  PATHS  "${GLEW_DIR}/config"  )
if(__find_version)
	SET(GLEW_VERSION_FILE "${GLEW_DIR}/config/version")
	file(STRINGS "${GLEW_VERSION_FILE}" GLEW_VERSION_PARTS REGEX "GLEW_+[A-Z]+[ ]+" )
	string(REGEX REPLACE "GLEW_MAJOR = +([0-9])+.*" "\\1" GLEW_VERSION_MAJOR "${GLEW_VERSION_PARTS}")
	string(REGEX REPLACE ".+GLEW_MINOR = +([0-9])+.*" "\\1" GLEW_VERSION_MINOR "${GLEW_VERSION_PARTS}")
	string(REGEX REPLACE ".+GLEW_MICRO = +([0-9]+).*" "\\1" GLEW_VERSION_PATCH "${GLEW_VERSION_PARTS}")
	set(GLEW_VERSION_PLAIN "${GLEW_VERSION_MAJOR}${GLEW_VERSION_MINOR}${GLEW_VERSION_PATCH}")
	set(GLEW_VERSION "${GLEW_VERSION_MAJOR}.${GLEW_VERSION_MINOR}.${GLEW_VERSION_PATCH}" CACHE PATH "Installed GL0EW version")
	mark_as_advanced(GLEW_VERSION)
endif()
unset(__find_version CACHE)

if(GLEW_DIR AND GLEW_LIBS ) 
	set(GLEW_FOUND TRUE CACHE STRING "Found glew")
	message(STATUS "[FindGLEW] - Found GLEW at " ${GLEW_DIR} ", Version " ${GLEW_VERSION})
endif()


