# Locate OpenSceneGraph 
# Rafael Radkowski


set(_OPEN_THREADS_NAME OpenThreads )
set(_OPEN_THREADS_DEBUG_NAME OpenThreadsd )
set(_OSG_LIB_NAME osg)
set(_OSG_LIB_DEBUG_NAME osgd)
set(_OSG_ANIMATION_LIB_NAME osgAnimation)
set(_OSG_ANIMATION_LIB_DEBUG_NAME osgAnimationd)
set(_OSG_DB_LIB_NAME osgDB)
set(_OSG_DB_LIB_DEBUG_NAME osgDBd)
set(_OSG_GA_LIB_NAME osgGA)
set(_OSG_GA_LIB_DEBUG_NAME osgGAd)
set(_OSG_MANIPULATOR_LIB_NAME osgManipulator)
set(_OSG_MANIPULATOR_LIB_DEBUG_NAME osgManipulatord)
set(_OSG_TEXT_LIB_NAME osgText)
set(_OSG_TEXT_LIB_DEBUG_NAME osgTextd)
set(_OSG_UTIL_LIB_NAME osgUtil)
set(_OSG_UTIL_LIB_DEBUG_NAME osgUtild)
set(_OSG_VIEWER_LIB_NAME osgViewer)
set(_OSG_VIEWER_LIB_DEBUG_NAME osgViewerd)


set(_OSG_DEFAULT_INSTALL_DIR 
				"${OSG_ROOT}"
				"C:/SDK/OpenSceneGraph-3.4.0" 
				"C:/SDK/OpenSceneGraph-3.2.4" 
				"D:/SDK/OpenSceneGraph-3.2.3"
				 )
    



#-- Clear the public variables
set (OSG_FOUND "NO")


find_path(OSG_ROOT 
	NAMES /src/osg/Node.cpp
	PATHS ${_OSG_DEFAULT_INSTALL_DIR}
)



# look for tbb in typical include dirs
find_path (OSG_INCLUDE_DIR NAMES osg/Node
		PATHS ${OSG_ROOT}/include
		PATH_SUFFIXES include )


# verify that config is in the right path
find_file(OSG_CONFIG NAMES osg/config
	PATHS ${OSG_ROOT}/include)

if(NOT OSG_CONFIG)
	message("ERROR: OSG CONFIGS IS IN THE WRONG PATH")
endif(NOT OSG_CONFIG)
mark_as_advanced(${OSG_CONFIG})

find_path (OPENTHREADS_INCLUDE_DIR NAMES OpenThreads/Thread
		PATHS ${OSG_ROOT}/include
		PATH_SUFFIXES include )


find_path (OSG_LIBRARY_DIR NAMES osg.lib
		PATHS ${OSG_ROOT}/lib ${OSG_ROOT}/build/lib
		PATH_SUFFIXES lib)


# look for libraries

find_library(OPENTHREADS_LIB ${_OPEN_THREADS_NAME} 
		PATHS ${OSG_ROOT}/lib ${OSG_ROOT}/build/lib
)

if(NOT OPENTHREADS_LIB)
	message("ERROR: Did not find OpenThreads!")
endif(NOT OPENTHREADS_LIB)

find_library(OPENTHREADS_DEBUG_LIB ${_OPEN_THREADS_DEBUG_NAME} 
		PATHS ${OSG_ROOT}/lib ${OSG_ROOT}/build/lib
)

find_library(OPENTHREADS_LIBRARY ${_OPEN_THREADS_NAME} 
		PATHS ${OSG_ROOT}/lib ${OSG_ROOT}/build/lib
)
find_library(OPENTHREADS_LIBRARY_DEBUG ${_OPEN_THREADS_DEBUG_NAME} 
		PATHS ${OSG_ROOT}/lib ${OSG_ROOT}/build/lib
)

find_library(OSG_LIB ${_OSG_LIB_NAME} 
		PATHS ${OSG_ROOT}/lib ${OSG_ROOT}/build/lib
)

if(NOT OSG_LIB)
	message("ERROR: Did not find osg.lib")
endif(NOT OSG_LIB)

find_library(OSG_DEBUG_LIB ${_OSG_LIB_DEBUG_NAME} 
		PATHS ${OSG_ROOT}/lib ${OSG_ROOT}/build/lib
)

find_library(OSG_ANIMATION_LIB ${_OSG_ANIMATION_LIB_NAME} 
		PATHS ${OSG_ROOT}/lib ${OSG_ROOT}/build/lib
)
find_library(OSG_ANIMATION_DEBUG_LIB ${_OSG_ANIMATION_LIB_DEBUG_NAME} 
		PATHS ${OSG_ROOT}/lib ${OSG_ROOT}/build/lib
)

find_library(OSG_DB_LIB ${_OSG_DB_LIB_NAME} 
		PATHS ${OSG_ROOT}/lib ${OSG_ROOT}/build/lib
)
find_library(OSG_DB_DEBUG_LIB ${_OSG_DB_LIB_DEBUG_NAME} 
		PATHS ${OSG_ROOT}/lib ${OSG_ROOT}/build/lib
)


find_library(OSG_GA_LIB ${_OSG_GA_LIB_NAME} 
		PATHS ${OSG_ROOT}/lib ${OSG_ROOT}/build/lib
)
find_library(OSG_GA_DEBUG_LIB ${_OSG_GA_LIB_DEBUG_NAME} 
		PATHS ${OSG_ROOT}/lib ${OSG_ROOT}/build/lib
)

find_library(OSG_MANIPULATOR_LIB ${_OSG_MANIPULATOR_LIB_NAME} 
		PATHS ${OSG_ROOT}/lib ${OSG_ROOT}/build/lib
)
find_library(OSG_MANIPULATOR_DEBUG_LIB ${_OSG_MANIPULATOR_LIB_DEBUG_NAME} 
		PATHS ${OSG_ROOT}/lib ${OSG_ROOT}/build/lib
)

find_library(OSG_TEXT_LIB ${_OSG_TEXT_LIB_NAME} 
		PATHS ${OSG_ROOT}/lib ${OSG_ROOT}/build/lib
)
find_library(OSG_TEXT_DEBUG_LIB ${_OSG_TEXT_LIB_DEBUG_NAME} 
		PATHS ${OSG_ROOT}/lib ${OSG_ROOT}/build/lib
)

find_library(OSG_UTIL_LIB ${_OSG_UTIL_LIB_NAME} 
		PATHS ${OSG_ROOT}/lib ${OSG_ROOT}/build/lib
)
find_library(OSG_UTIL_DEBUG_LIB ${_OSG_UTIL_LIB_DEBUG_NAME} 
		PATHS ${OSG_ROOT}/lib ${OSG_ROOT}/build/lib
)

find_library(OSG_VIEWER_LIB ${_OSG_VIEWER_LIB_NAME} 
		PATHS ${OSG_ROOT}/lib ${OSG_ROOT}/build/lib
)
find_library(OSG_VIEWER_DEBUG_LIB ${_OSG_VIEWER_LIB_DEBUG_NAME} 
		PATHS ${OSG_ROOT}/lib ${OSG_ROOT}/build/lib
)



set(OSG_LIBRARIES_DEBUG 
	${OPENTHREADS_DEBUG_LIB}
	${OSG_DEBUG_LIB}
	${OSG_ANIMATION_DEBUG_LIB}
	${OSG_DB_DEBUG_LIB}
	${OSG_GA_DEBUG_LIB}
	${OSG_MANIPULATOR_DEBUG_LIB}
	${OSG_TEXT_DEBUG_LIB}
	${OSG_UTIL_DEBUG_LIB}
	${OSG_VIEWER_DEBUG_LIB})
	
set(OSG_LIBRARIES 
	${OPENTHREADS_LIB}
	${OSG_LIB}
	${OSG_ANIMATION_LIB}
	${OSG_DB_LIB}
	${OSG_GA_LIB}
	${OSG_MANIPULATOR_LIB}
	${OSG_TEXT_LIB}
	${OSG_UTIL_LIB}
	${OSG_VIEWER_LIB})

if(NOT OSG_LIBRARIES )
	message("ERROR: OSG incomplete!")
endif(NOT OSG_LIBRARIES )
if(NOT OSG_LIBRARIES_DEBUG )
	message("ERROR: OSG incomplete!")
endif(NOT OSG_LIBRARIES_DEBUG )

if( OSG_LIBRARIES )
	message("Found OSG!")
endif( OSG_LIBRARIES )
if(OSG_LIBRARIES_DEBUG )
	message("Found OSG debug")
endif( OSG_LIBRARIES_DEBUG )


mark_as_advanced(${OSG_LIBRARIES_DEBUG} ${OSG_LIBRARIES} )