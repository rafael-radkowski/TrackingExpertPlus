# Locate the glfw3 library
#
# This module defines the following variables:
#
# GLFW3_LIB	      	 the name of the library;
# GLFW3_INCLUDE_DIR  where to find glfw include files.
# GLFW3_FOUND        true if both the GLFW3_LIBRARY and GLFW3_INCLUDE_DIR have been found.
#
# To help locate the library and include file, you can define a 
# variable called GLFW3_ROOT which points to the root of the glfw library 
# installation.
#
# C:\SDK\glfw-3.2.1\src\Debug\glfw3.lib
# C:\SDK\glfw-3.2.1\src\Release\glfw3.lib

# default search dirs
set(GLFW3_FOUND FALSE)


# default search dirs
set( _glfw3_SEARCH_DIRS 
  "/usr/include"
  "/usr/local/include"
  "$ENV{PROGRAMFILES}/glfw/include" 
  "C:/SDK/glfw/include"
  )
  

# 1. Read the environment variable
if (NOT _glfw_DIR)
  if (DEFINED ENV{GLFW_DIR})
    set (_glfw_DIR "$ENV{GLFW_DIR}" CACHE PATH "Installation prefix of glfw Library." FORCE)
  elseif (DEFINED ENV{GLFW3_DIR})
    set (_glfw_DIR "$ENV{GLFW3_DIR}" CACHE PATH "Installation prefix of glfw Library." FORCE)
  elseif (DEFINED ENV{Glfw3_DIR})
    set (_glfw_DIR "$ENV{Glfw3_DIR}" CACHE PATH "Installation prefix of glfw Library." FORCE)
 elseif (DEFINED ENV{Glfw_DIR})
    set (_glfw_DIR "$ENV{Glfw_DIR}" CACHE PATH "Installation prefix of glfw Library." FORCE)
  elseif (DEFINED ENV{GLFW_ROOT})
    set (_glfw_DIR "$ENV{GLFW_ROOT}" CACHE PATH "Installation prefix of glfw Library." FORCE)
  elseif (DEFINED ENV{GLFW3_ROOT})
    set (_glfw_DIR "$ENV{GLFW3_ROOT}" CACHE PATH "Installation prefix of glfw Library." FORCE)
  elseif (DEFINED ENV{Glfw3_ROOT})
    set (_glfw_DIR "$ENV{Glfw3_ROOT}" CACHE PATH "Installation prefix of glfw Library." FORCE)
 elseif (DEFINED ENV{Glfw_ROOT})
    set (_glfw_DIR "$ENV{Glfw_ROOT}" CACHE PATH "Installation prefix of glfw Library." FORCE)
  endif ()
endif ()

if(_glfw_DIR  )
	set(GLFW3_DIR ${_glfw_DIR} CACHE PATH "GLFW installatin path")
endif()

# 2. If environment variable is nto defined, search at typical locations
if (NOT _glfw_DIR)
	FIND_PATH(GLFW3_DIR "glfw/glfw3.h"
		PATHS ${_glfw3_SEARCH_DIRS} )
endif()
unset(_glfw_DIR CACHE)


# 3. Find the include directory
find_file(__find_glfw "glfw/glfw3.h" ${GLFW3_DIR}/include)
if(__find_glfw)
	find_path(GLFW3_INCLUDE_DIR "glfw/glfw3.h"
	PATHS ${GLFW3_DIR}/include)
	
	if(NOT GLFW3_INCLUDE_DIR)
		message("[FindGLFW3] - ERROR: did not find the GLFW3 include dir")
	endif()
else()	
   message("[FindGLFW3] - ERROR: did not find the GLFW3 include dir")
endif()
unset(__find_glfw CACHE)

# 4. Find the library directory
set( _glfw_LIB_SEARCH_DIRS 
  "${GLFW3_DIR}/lib/Release/x64"
  "${GLFW3_DIR}/lib"
  "${GLFW3_DIR}/lib/Release"  
  "${GLFW3_DIR}/src/Release" 
  "${GLFW3_DIR}/src/Release/x64" 
  )
  
 set( _glew_DEBUG_LIB_SEARCH_DIRS 
  "${GLFW3_DIR}/lib/Debug/x64"
  "${GLFW3_DIR}/lib/Debug" 
  "${GLFW3_DIR}/src/Debug" 
  "${GLFW3_DIR}/src/Debug/x64"  
  )
  
if(GLFW3_DIR)
	set(GLFW3_LIBS_LIST )

	find_file(__find_glfw_lib "glfw3.lib" PATHS ${_glfw_LIB_SEARCH_DIRS})
	if(__find_glfw_lib)
		find_library(GLFW3_LIBRARY_RELEASE "glfw3.lib"  PATHS ${_glfw_LIB_SEARCH_DIRS})
		if(GLFW3_LIBRARY_RELEASE)
			set(GLFW3_LIBRARY_RELEASE_CONF "optimized" ${GLFW3_LIBRARY_RELEASE})
			list(APPEND GLFW3_LIBS_LIST ${GLFW3_LIBRARY_RELEASE_CONF})
		endif()
	else(__find_glfw_lib)
		message("[FindGLFW3] - ERROR: cannot find glfw3.lib")
	endif(__find_glfw_lib)
	unset(__find_glfw_lib CACHE)

	find_file(__find_glfw_lib "glfw3.lib" PATHS ${_glew_DEBUG_LIB_SEARCH_DIRS})
	if(__find_glfw_lib)
		find_library(GLFW3_LIBRARY_DEBUG "glfw3.lib"  PATHS ${_glew_DEBUG_LIB_SEARCH_DIRS})
		if(GLFW3_LIBRARY_DEBUG)
			set(GLFW3_LIBRARY_DEBUG "debug" ${GLFW3_LIBRARY_DEBUG})
			list( APPEND GLFW3_LIBS_LIST ${GLFW3_LIBRARY_DEBUG})
		endif()
	else(__find_glfw_lib)
		message("[FindGLFW3] - ERROR: cannot find glfw3.lib")
		## add the release library to the debug config so that the debug config has an entry. 
		set(GLFW3_LIBRARY_DEBUG "debug" ${GLFW3_LIBRARY_RELEASE})
		list( APPEND GLFW3_LIBS_LIST ${GLFW3_LIBRARY_DEBUG})
	endif(__find_glfw_lib)
	unset(__find_glfw_lib CACHE)
	
	set(GLFW3_LIBS ${GLFW3_LIBS_LIST} CACHE DOC "Glfw3 libs")
endif (GLFW3_DIR)



if(GLFW3_DIR AND GLFW3_LIBS ) 
	set(GLFW3_FOUND TRUE CACHE DOC "Found glfw3")
	message(STATUS "[FindGLFW3] - Found GLFW3 at " ${GLFW3_DIR} )
endif()

