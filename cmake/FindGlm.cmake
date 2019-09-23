# Locate the glm files 
#
# This module defines the following variables:
#
# GLM_INCLUDE_DIR  where to find glm include files.
# GLM_FOUND        true if both the GLM_INCLUDE_DIR and GLFW3_INCLUDE_DIR have been found.
#
# To help locate the library and include file, you can define a 
# variable called GLFW3_ROOT which points to the root of the glfw library 
# installation.
#

# default search dirs
set( _glm_HEADER_SEARCH_DIRS 
  "/usr/include"
  "/usr/local/include"
  "C:/Program Files (x86)/glm/include" 
  "../../SDK/include"
  "../SDK/include"
  "C:/SDK/glm-0.9.9.3"
  )
  

# Check environment for root search directory
set( _glm_ENV_ROOT $ENV{GLM_ROOT} )
if( NOT GLM_ROOT AND _glm_ENV_ROOT )
	set(GLM_ROOT ${_glm_ENV_ROOT} )
endif()

# Put user specified location at beginning of search
if( GLM_ROOT )
	list( INSERT _glm_HEADER_SEARCH_DIRS 0 "${GLM_ROOT}/include" )
endif()

# Search for the header 
FIND_PATH(GLM_INCLUDE_DIR "glm/glm.hpp"
	PATHS ${_glm_HEADER_SEARCH_DIRS} )

	

INCLUDE(FindPackageHandleStandardArgs)
FIND_PACKAGE_HANDLE_STANDARD_ARGS(GLM DEFAULT_MSG
	GLM_INCLUDE_DIR)