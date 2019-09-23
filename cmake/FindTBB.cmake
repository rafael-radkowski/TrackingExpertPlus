# Locate Intel Threading Building Blocks include paths and libraries
# FindTBB.cmake can be found at https://code.google.com/p/findtbb/


if (WIN32)
 
    set(_TBB_DEFAULT_INSTALL_DIR "C:/SDK/tbb2017_20170226oss" 
				"C:/Program Files/Intel/TBB" 
				"C:/Program Files (x86)/Intel/TBB" 
				"${TBB_ROOT}"
				"C:/SDK/tbb2017_20170118oss/include"
				"C:/SDK/tbb/include"
				"C:/SDK/tbb2017_20170226oss/include"
				"C:/SDK/tbb2017_20170118oss"
				"C:/SDK/tbb"
				"D:/SDK/tbb2017_20170226oss"
				"C:/SDK/tbb2017_20170226oss" )
    

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

    set(_TBB_ARCHITECTURE ${TBB_ARCHITECTURE})
endif (WIN32)


#-- Clear the public variables
set (TBB_FOUND "NO")


find_path(TBB_ROOT 
	NAMES bin/intel64/vc14/tbb.dll
	PATHS ${_TBB_DEFAULT_INSTALL_DIR}
	PATH_SUFFIXES bin
)



# look for tbb in typical include dirs
find_path (TBB_INCLUDE_DIR NAMES tbb/tbb.h
		PATHS ${_TBB_DEFAULT_INSTALL_DIR} ${TBB_ROOT}/include
		PATH_SUFFIXES include )

if (NOT TBB_INCLUDE_DIR) 
	 message("ERROR: Intel TBB NOT found!")
endif(NOT TBB_INCLUDE_DIR) 

if ( TBB_INCLUDE_DIR) 
	 message("Found Intel TBB")
endif( TBB_INCLUDE_DIR) 


find_path (TBB_LIBRARY_DIR NAMES intel64/${_TBB_COMPILER}/tbb.lib
		PATHS ${TBB_ROOT}/lib
		PATH_SUFFIXES include )


find_library(TBB_MALLOC_LIBRARY_DEBUG ${_TBB_LIB_MALLOC_DEBUG_NAME} 
		PATHS ${TBB_ROOT}/lib/intel64/${_TBB_COMPILER}
)

find_library(TBB_MALLOC_LIBRARY ${_TBB_LIB_MALLOC_NAME} 
		PATHS ${TBB_ROOT}/lib/intel64/${_TBB_COMPILER}
)


find_library(TBB_LIBRARY_DEBUG ${_TBB_LIB_DEBUG_NAME} 
		PATHS ${TBB_ROOT}/lib/intel64/${_TBB_COMPILER}
)

find_library(TBB_LIBRARY ${_TBB_LIB_NAME} 
		PATHS ${TBB_ROOT}/lib/intel64/${_TBB_COMPILER}
)

set(TBB_LIBS_DEBUG ${TBB_MALLOC_LIBRARY_DEBUG}  ${TBB_LIBRARY_DEBUG})
set(TBB_LIBS ${TBB_MALLOC_LIBRARY}  ${TBB_LIBRARY})
mark_as_advanced(${TBB_LIBS_DEBUG} ${TBB_LIBS})





