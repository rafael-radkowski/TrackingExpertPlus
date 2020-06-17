# Downloaded from:
# Locate Intel Threading Building Blocks include paths and libraries
# FindTBB.cmake can be found at https://code.google.com/p/findtbb/ (MIT License)
#
# 1. Setup
# Set the environment variable TBB_DIR pointing to the main Eigen 3 folder, e.g., C:\SDK\tbb-2019
#
# 2. Variables
# The following variables are produced and usable in a CMakeList.txt file after cmake is done: 
#
#  TBB_INCLUDE_DIR - the eigen include directory
#  TBB_LIBS - the tbb libraries, it distinguishes debug vs. release libraries. 
#
# Original file extended/modified by:
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
# - Changed TBB_Libs cache name; removed space. 
# 



if (WIN32)
 
    set(_TBB_DEFAULT_INSTALL_DIR "C:/SDK/tbb2017_20170226oss" 
				"C:/Program Files/Intel/TBB" 
				"C:/Program Files (x86)/Intel/TBB" 
				"$ENV{TBB_ROOT}"
				"$ENV{TBB_DIR}"
				"$ENV{tbb_DIR}"
				"$ENV{Tbb_DIR}"
				"$ENV{PROGRAMFILES}/Intel/TBB" 
		)
    

set(_TBB_LIB_NAME "tbb")
    set(_TBB_LIB_MALLOC_NAME "${_TBB_LIB_NAME}malloc")
    set(_TBB_LIB_DEBUG_NAME "${_TBB_LIB_NAME}_debug")
    set(_TBB_LIB_MALLOC_DEBUG_NAME "${_TBB_LIB_MALLOC_NAME}_debug")

    if (MSVC71)
        set (_TBB_COMPILER "vc7.1")
    endif(MSVC71)
    if (MSVC80)
        set(_TBB_COMPILER "vc8")
    endif(MSVC80)
    if (MSVC90)
        set(_TBB_COMPILER "vc9")
    endif(MSVC90)
    if(MSVC10)
        set(_TBB_COMPILER "vc10")
    endif(MSVC10)
	if(MSVC12)
        set(_TBB_COMPILER "vc12")
    endif(MSVC12)
	if(MSVC14)
        set(_TBB_COMPILER "vc14")
    endif(MSVC14)
	if(MSVC15)
        set(_TBB_COMPILER "vc15")
    endif(MSVC15)
	if(MSVC17)
        set(_TBB_COMPILER "vc17")
    endif(MSVC17)
	if(MSVC19)
        set(_TBB_COMPILER "vc19")
    endif(MSVC19)

    set(_TBB_ARCHITECTURE ${TBB_ARCHITECTURE})
endif (WIN32)


#-- Clear the public variables
set (TBB_FOUND "NO")


find_path(TBB_DIR 
	NAMES bin/intel64/${_TBB_COMPILER}/tbb.dll
	PATHS ${_TBB_DEFAULT_INSTALL_DIR}
	PATH_SUFFIXES bin
)



# look for tbb in typical include dirs
find_path (TBB_INCLUDE_DIR NAMES tbb/tbb.h
		PATHS ${_TBB_DEFAULT_INSTALL_DIR} ${TBB_DIR}/include
		PATH_SUFFIXES include )

if (NOT TBB_INCLUDE_DIR) 
	 message("ERROR: Intel TBB NOT found!")
endif(NOT TBB_INCLUDE_DIR) 

if ( TBB_INCLUDE_DIR) 
	 message(STATUS "[FindTBB] - Found Intel TBB at " ${TBB_DIR})
endif( TBB_INCLUDE_DIR) 


find_path (TBB_LIBRARY_DIR NAMES intel64/${_TBB_COMPILER}/tbb.lib
		PATHS ${TBB_DIR}/lib
		PATH_SUFFIXES include )


find_library(TBB_MALLOC_LIBRARY_DEBUG ${_TBB_LIB_MALLOC_DEBUG_NAME} 
		PATHS ${TBB_DIR}/lib/intel64/${_TBB_COMPILER}
)


find_library(TBB_MALLOC_LIBRARY ${_TBB_LIB_MALLOC_NAME} 
		PATHS ${TBB_DIR}/lib/intel64/${_TBB_COMPILER}
)


find_library(TBB_LIBRARY_DEBUG ${_TBB_LIB_DEBUG_NAME} 
		PATHS ${TBB_DIR}/lib/intel64/${_TBB_COMPILER}
)

find_library(TBB_LIBRARY ${_TBB_LIB_NAME} 
		PATHS ${TBB_DIR}/lib/intel64/${_TBB_COMPILER}
)




set(TBB_LIBS 
	optimized ${TBB_MALLOC_LIBRARY} 
    optimized ${TBB_LIBRARY} 
	debug ${TBB_MALLOC_LIBRARY_DEBUG} 
	debug ${TBB_LIBRARY_DEBUG}
	CACHE PATH "TBB_libraries")
	
mark_as_advanced(${TBB_LIBS})

unset(TBB_MALLOC_LIBRARY_DEBUG CACHE)
unset(TBB_MALLOC_LIBRARY CACHE)
unset(TBB_LIBRARY_DEBUG CACHE)
unset(TBB_LIBRARY CACHE)



