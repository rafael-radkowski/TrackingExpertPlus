# - Try to find Eigen3 lib
#
# This module supports requiring a minimum version, e.g. you can do
#   find_package(Eigen3 3.1.2)
# to require version 3.1.2 or newer of Eigen3.
#
# Once done this will define
#
#  EIGEN3_FOUND - system has eigen lib with correct version
#  EIGEN3_INCLUDE_DIR - the eigen include directory
#  EIGEN3_VERSION - eigen version
#
# This module reads hints about search locations from 
# the following enviroment variables:
#
# EIGEN3_ROOT
# EIGEN3_ROOT_DIR

# Copyright (c) 2006, 2007 Montel Laurent, <montel@kde.org>
# Copyright (c) 2008, 2009 Gael Guennebaud, <g.gael@free.fr>
# Copyright (c) 2009 Benoit Jacob <jacob.benoit.1@gmail.com>
# Redistribution and use is allowed according to the terms of the 2-clause BSD license.



find_path(EIGEN3_ROOT 
	NAMES signature_of_eigen3_matrix_library 
	PATHS
    	${CMAKE_INSTALL_PREFIX}/include
    	${KDE4_INCLUDE_DIR}
	C:/SDK/Eigen3
	C:/SDK/Eigen
	D:/SDK/Eigen3
	C:/SDK/Eigen3-3.3.7
	PATH_SUFFIXES eigen3 eigen
)


find_path(EIGEN3_INCLUDE_DIR NAMES signature_of_eigen3_matrix_library
    PATHS
    ${CMAKE_INSTALL_PREFIX}/include
    ${KDE4_INCLUDE_DIR}
	C:/SDK/Eigen3
	C:/SDK/Eigen
	D:/SDK/Eigen3
	C:/SDK/Eigen3-3.3.7
    PATH_SUFFIXES eigen3 eigen
	
)



find_path(EIGEN3_DIR NAMES signature_of_eigen3_matrix_library
    PATHS
    ${CMAKE_INSTALL_PREFIX}/include
    ${KDE4_INCLUDE_DIR}
	C:/SDK/Eigen3
	C:/SDK/Eigen
	D:/SDK/Eigen3
	C:/SDK/Eigen3-3.3.7
    PATH_SUFFIXES eigen3 eigen
	
)


 if(${EIGEN3_INCLUDE_DIR} STREQUAL "" )
    set(EIGEN3_VERSION_OK FALSE)
 else(${EIGEN3_INCLUDE_DIR} STREQUAL "")
    set(EIGEN3_VERSION_OK TRUE)
endif(${EIGEN3_INCLUDE_DIR} STREQUAL "")


set(EIGEN3_FOUND ${EIGEN3_VERSION_OK})
mark_as_advanced(EIGEN3_FOUND)