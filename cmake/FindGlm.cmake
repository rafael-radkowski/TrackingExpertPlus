# Locate the glm files 
#
# This module defines the following variables:
#
# GLM_INCLUDE_DIR   where to find glm include files.
# GLM_FOUND         true if both the GLM_INCLUDE_DIR and GLFW3_INCLUDE_DIR have been found.
# GLM_VERSION		the version number of the installed glm version
#
# To help locate the library and include file, you can define a 
# variable called GLFW3_ROOT which points to the root of the glfw library 
# installation.
#

set(GLM_FOUND FALSE)

# default search dirs
set( _glm_HEADER_SEARCH_DIRS 
  "/usr/include"
  "/usr/local/include"
  "$ENV{PROGRAMFILES}/glm " 
  "C:/SDK/glm"
  "C:/SDK/glm-0.9.9.3"
  )
  
  
# 1. Read the environment variable
if (NOT _glm_DIR)
  if (DEFINED ENV{GLM_DIR})
    set (_glm_DIR "$ENV{GLM_DIR}" CACHE PATH "Installation prefix of OpenCV Library." FORCE)
  elseif (DEFINED ENV{glm_ROOT})
    set (_glm_DIR "$ENV{glm_ROOT}" CACHE PATH "Installation prefix of OpenCV Library." FORCE)
  elseif (DEFINED ENV{Glm_ROOT})
    set (_glm_DIR "$ENV{Glm_ROOT}" CACHE PATH "Installation prefix of OpenCV Library." FORCE)
  elseif (DEFINED ENV{Glm_DIR})
    set (_glm_DIR "$ENV{Glm_DIR}" CACHE PATH "Installation prefix of OpenCV Library." FORCE)
 elseif (DEFINED ENV{Glm_DIR})
    set (_glm_DIR "$ENV{Glm_DIR}" CACHE PATH "Installation prefix of OpenCV Library." FORCE)
  endif ()
endif ()

if(_glm_DIR  )
	set(GLM_DIR ${_glm_DIR} CACHE PATH "GLM installatin path")
endif()

# 2. If environment variable is nto defined, search at typical locations
if (NOT _glm_DIR)
	FIND_PATH(GLM_DIR "glm/glm.hpp"
		PATHS ${_glm_HEADER_SEARCH_DIRS} )
endif()



if (GLM_DIR)

# 3. Search for the header 
# Search for the header 
FIND_PATH(GLM_INCLUDE_DIR "glm/glm.hpp"
	PATHS ${GLM_DIR} )
	
# 4. Read the version number
find_file(__find_version "setup.hpp"  PATHS  "${GLM_DIR}/glm/detail"  )
if(__find_version)
	SET(GLM_VERSION_FILE "${GLM_DIR}/glm/detail/setup.hpp")
	file(STRINGS "${GLM_VERSION_FILE}" GLM_VERSION_PARTS REGEX "#define GLM_VERSION_[A-Z]+\t+" )
	#message(${GLM_VERSION_PARTS})
	string(REGEX REPLACE ".+GLM_VERSION_MAJOR[\t]+([0-9]+).*" "\\1" GLM_VERSION_MAJOR "${GLM_VERSION_PARTS}")
	string(REGEX REPLACE ".+GLM_VERSION_MINOR[\t]+([0-9]+).*" "\\1" GLM_VERSION_MINOR "${GLM_VERSION_PARTS}")
	string(REGEX REPLACE ".+GLM_VERSION_PATCH[\t]+([0-9]+).*" "\\1" GLM_VERSION_PATCH "${GLM_VERSION_PARTS}")
	string(REGEX REPLACE ".+GLM_VERSION_REVISION[\t]+([0-9]+).*" "\\1" GLM_VERSION_REVISION "${GLM_VERSION_PARTS}")
	set(GLM_VERSION_PLAIN "${GLM_VERSION_MAJOR}${GLM_VERSION_MINOR}${GLM_VERSION_PATCH}${GLM_VERSION_REVISION}")
	set(GLM_VERSION "${GLM_VERSION_MAJOR}.${GLM_VERSION_MINOR}.${GLM_VERSION_PATCH}.${GLM_VERSION_REVISION}" CACHE PATH "Installed GLM version")
	mark_as_advanced(GLM_VERSION)
endif()
unset(__find_version CACHE)

message(STATUS "[FindGLM] - Found glm at " ${GLM_DIR} ", Version " ${GLM_VERSION} )


set(GLM_FOUND TRUE CACHE PATH "Found glm")

endif()
	
