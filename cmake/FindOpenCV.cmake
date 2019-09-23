##############################################################################
# @file  FindOpenCV.cmake
# @brief Find OpenCV Library (http://sourceforge.net/projects/opencvlibrary/)
#
# The script supports OpenCV 3 and OpenCV 2 
# It was tested with 2.4.13 and 3.4.5 under Windows 10
#
# 1. Setup
# The script requires the environment variables OpenCV_DIR to point to the path [your path]/opencv
# So the the OpenCV_DIR environment variable, e.g., C:\SDK\opencv-3.4.5
#
# 2. Variables
#
# The following variables are produced and usable in a CMakeList.txt file after cmake is done: 
# - OpenCV_FOUND
# - OpenCV_LIBS 
# - OpenCV_LIBRARY_DIR, path to the folder with release files
# - OpenCV_INCLUDE_DIR
# - OpenCV_VERSION (OpenCV_VERSION_MAJOR, OpenCV_VERSION_MINOR, OpenCV_VERSION_PATCH)
#
#
#  # tested with:
# - OpenCV 2.4.13:  MSVC 15,2017
# - OpenCV 3.4.5:  MSVC 15,2017
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
#
# 



# ----------------------------------------------------------------------------
# start search
set (OpenCV_FOUND FALSE)
set (OpenCV_Version)
set (OpenCV_LIBRARY_DIR_FOUND FALSE)


# Default OpenCV folders 
set(_OPENCV_DEFAULT_INSTALL_DIR 
	C:/SDK/OpenCV
)


# 1. Start to determine OpenCV path from the environment variable.
if (NOT OpenCV_DIR)
  if (DEFINED ENV{OpenCV_DIR})
    set (OpenCV_DIR "$ENV{OpenCV_DIR}" CACHE PATH "Installation prefix of OpenCV Library." FORCE)
  elseif (DEFINED ENV{OPENCV_DIR})
    set (OpenCV_DIR "$ENV{OPENCV_DIR}" CACHE PATH "Installation prefix of OpenCV Library." FORCE)
  elseif (DEFINED ENV{OPENCV_ROOT})
    set (OpenCV_DIR "$ENV{OPENCV_ROOT}" CACHE PATH "Installation prefix of OpenCV Library." FORCE)
  elseif (DEFINED ENV{OpenCV_ROOT})
    set (OpenCV_DIR "$ENV{OpenCV_ROOT}" CACHE PATH "Installation prefix of OpenCV Library." FORCE)
  endif ()
endif ()

# 2. Look at typical install pathes for OpenCV
#if (NOT OpenCV_DIR)
#  find_path ( OpenCV_DIR 
#	NAMES /include/opencv2/opencv.hpp
#	PATHS ${_OPENCV_DEFAULT_INSTALL_DIR}
#   PATH_SUFFIXES "include" "include/open
#    DOC "Directory of cv.h header file."
#  )
#endif ()

    

# ----------------------------------------------------------------------------
# Extract the version from a version file. 

# OpenCV Version 3 support
find_file(__find_version "version.hpp"  PATHS  "${OpenCV_DIR}/modules/core/include/opencv2/core/"  )
if(__find_version)
	SET(OPENCV_VERSION_FILE "${OpenCV_DIR}/modules/core/include/opencv2/core/version.hpp")
	file(STRINGS "${OPENCV_VERSION_FILE}" OPENCV_VERSION_PARTS REGEX "#define CV_VERSION_[A-Z]+[ ]+" )

	string(REGEX REPLACE ".+CV_VERSION_MAJOR[ ]+([0-9]+).*" "\\1" OPENCV_VERSION_MAJOR "${OPENCV_VERSION_PARTS}")
	string(REGEX REPLACE ".+CV_VERSION_MINOR[ ]+([0-9]+).*" "\\1" OPENCV_VERSION_MINOR "${OPENCV_VERSION_PARTS}")
	string(REGEX REPLACE ".+CV_VERSION_REVISION[ ]+([0-9]+).*" "\\1" OPENCV_VERSION_PATCH "${OPENCV_VERSION_PARTS}")
	string(REGEX REPLACE ".+CV_VERSION_STATUS[ ]+\"([^\"]*)\".*" "\\1" OPENCV_VERSION_STATUS "${OPENCV_VERSION_PARTS}")

	set(OPENCV_VERSION_PLAIN "${OPENCV_VERSION_MAJOR}.${OPENCV_VERSION_MINOR}.${OPENCV_VERSION_PATCH}")
	set(OPENCV_VERSION "${OPENCV_VERSION_PLAIN}${OPENCV_VERSION_STATUS}")
	set(OPENCV_SOVERSION "${OPENCV_VERSION_MAJOR}.${OPENCV_VERSION_MINOR}")
	set(OPENCV_LIBVERSION "${OPENCV_VERSION_MAJOR}.${OPENCV_VERSION_MINOR}.${OPENCV_VERSION_PATCH}")
	set(OpenCV_VERSION "${OPENCV_VERSION_MAJOR}.${OPENCV_VERSION_MINOR}.${OPENCV_VERSION_PATCH}" CACHE PATH "Install version")
	set(OpenCV_LIBVERSION "${OPENCV_VERSION_MAJOR}${OPENCV_VERSION_MINOR}${OPENCV_VERSION_PATCH}")

	mark_as_advanced(OpenCV_VERSION)
	message("[FindOpenCV] - Found OpenCV at " ${OpenCV_DIR} ", Version "  ${OpenCV_VERSION})
endif()
unset(__find_version CACHE)

# OpenCV Version 2 support
find_file(__find_version "version.hpp"  PATHS  "${OpenCV_DIR}/sources/modules/core/include/opencv2/core/"  )
if(__find_version)
	SET(OPENCV_VERSION_FILE "${OpenCV_DIR}/sources/modules/core/include/opencv2/core/version.hpp")
	file(STRINGS "${OPENCV_VERSION_FILE}" OPENCV_VERSION_PARTS REGEX "#define CV_VERSION_[A-Z]+[ ]+" )

	string(REGEX REPLACE ".+CV_VERSION_EPOCH[ ]+([0-9]+).*" "\\1" OPENCV_VERSION_MAJOR "${OPENCV_VERSION_PARTS}")
	string(REGEX REPLACE ".+CV_VERSION_MAJOR[ ]+([0-9]+).*" "\\1" OPENCV_VERSION_MINOR "${OPENCV_VERSION_PARTS}")
	string(REGEX REPLACE ".+CV_VERSION_MINOR[ ]+([0-9]+).*" "\\1" OPENCV_VERSION_PATCH "${OPENCV_VERSION_PARTS}")
	
	set(OPENCV_VERSION_PLAIN "${OPENCV_VERSION_MAJOR}.${OPENCV_VERSION_MINOR}.${OPENCV_VERSION_PATCH}")
	set(OPENCV_VERSION "${OPENCV_VERSION_PLAIN}${OPENCV_VERSION_STATUS}")
	set(OPENCV_SOVERSION "${OPENCV_VERSION_MAJOR}.${OPENCV_VERSION_MINOR}")
	set(OPENCV_LIBVERSION "${OPENCV_VERSION_MAJOR}.${OPENCV_VERSION_MINOR}.${OPENCV_VERSION_PATCH}")
	set(OpenCV_VERSION "${OPENCV_VERSION_MAJOR}.${OPENCV_VERSION_MINOR}.${OPENCV_VERSION_PATCH}" CACHE PATH "Install version")
	set(OpenCV_LIBVERSION "${OPENCV_VERSION_MAJOR}${OPENCV_VERSION_MINOR}${OPENCV_VERSION_PATCH}")


	mark_as_advanced(OpenCV_VERSION)
	message("[FindOpenCV] - Found OpenCV at " ${OpenCV_DIR} ", Version "  ${OpenCV_VERSION})
endif()
unset(__find_version CACHE)

# ----------------------------------------------------------------------------
# Find the include directory
set(__INCLUDE_DIRS 
	${OpenCV_DIR}/build/include/
	${OpenCV_DIR}/builds/include/
	${OpenCV_DIR}/include/
)

find_path (
  OpenCV_INCLUDE_DIR "opencv2/core.hpp"
  HINTS ${__INCLUDE_DIRS}
  DOC "Directory of core.h header file."
  NO_DEFAULT_PATH
)

mark_as_advanced(OpenCV_INCLUDE_DIR)



# ----------------------------------------------------------------------------
# Find the library path
# 1. The script first tries to find the opencv_core file. 
# 2. If unsuccessful, it looks into opencv_world. 

set(__LIBRARY_DIRS 
	${OpenCV_DIR}/builds/lib/Release
	${OpenCV_DIR}/builds/lib/Debug
	${OpenCV_DIR}/build/lib/Release
	${OpenCV_DIR}/build/lib/Debug
)


# look for the release lib paths, start with core
if (NOT OpenCV_LIBRARY_DIR)
	# looking for opencv_core
	string(CONCAT _core_lib   "opencv_core" ${OpenCV_LIBVERSION} ".lib" )
	find_file(__find_lib ${_core_lib}  PATHS ${__LIBRARY_DIRS} ) 
	if(__find_lib)
		find_path (
		  OpenCV_LIBRARY_DIR "${_core_lib}"
		  HINTS ${__LIBRARY_DIRS}
		  DOC "Directory of opencv_core.lib file."
		  NO_DEFAULT_PATH
		)
	endif ()
	unset(__find_lib CACHE)
endif()

# continue to search for world
if (NOT OpenCV_LIBRARY_DIR)
	string(CONCAT _world_lib   "opencv_world" ${OpenCV_LIBVERSION} ".lib" )
	find_file(__find_lib ${_world_lib}  PATHS ${__LIBRARY_DIRS} ) 
	if(__find_lib)
		find_path (
		  OpenCV_LIBRARY_DIR "${_world_lib}"
		  HINTS ${__LIBRARY_DIRS}
		  DOC "Directory of opencv_world.lib file."
		  NO_DEFAULT_PATH
		)
	endif ()
	unset(__find_lib CACHE)
endif ()


# search for the debug lib path. 
if (NOT OpenCV_DEBUG_LIBRARY_DIR)
	string(CONCAT _core_lib_d   "opencv_core" ${OpenCV_LIBVERSION} "d.lib" )
	find_file(__find_lib ${_core_lib_d}  PATHS ${__LIBRARY_DIRS} ) 
	if(__find_lib)
		find_path (
		  OpenCV_DEBUG_LIBRARY_DIR "${_core_lib_d}"
		  HINTS ${__LIBRARY_DIRS}
		  DOC "Directory of opencv_core_d.lib file."
		  NO_DEFAULT_PATH
		)
	endif ()
	unset(__find_lib CACHE)
endif()
	
if (NOT OpenCV_DEBUG_LIBRARY_DIR)
	string(CONCAT _world_lib   "opencv_world" ${OpenCV_LIBVERSION} "d.lib" )
	find_file(__find_lib ${_world_lib}  PATHS ${__LIBRARY_DIRS} ) 
	if(__find_lib)
		find_path (
		  OpenCV_DEBUG_LIBRARY_DIR "${_world_lib}"
		  HINTS ${__LIBRARY_DIRS}
		  DOC "Directory of opencv_world_d.lib file."
		  NO_DEFAULT_PATH
		)
	endif ()
	unset(__find_lib CACHE)
endif()

##------------------------------------------
## Get the libraries


if(${OPENCV_VERSION_MAJOR} EQUAL "3")

	##------------------------------------------
	## Release libraries
	if ( OpenCV_LIBRARY_DIR)

		## test first if opencv_world is used	
		if(NOT OpenCV_WORLD )
			string(CONCAT _world_lib_   "opencv_world" ${OpenCV_LIBVERSION} ".lib" )
			find_file(__FIND_WORLD ${_world_lib_}  PATHS ${OpenCV_LIBRARY_DIR} ) 
			if(__FIND_WORLD)
				find_library(OpenCV_WORLD ${_world_lib_}  PATHS ${OpenCV_LIBRARY_DIR} OPTIONAL)
				if(OpenCV_WORLD)
					set(OpenCV_WORLD "optimized" ${OpenCV_WORLD})
				endif()
				set(OpenCV_LIBS ${OpenCV_WORLD}  CACHE PATH "Libraries")
				list(APPEND  ${OpenCV_WORLD} OpenCV_LIBS_EXP)
			endif ()
			unset(__FIND_WORLD CACHE)
		endif()
		
		if(NOT OpenCV_WORLD)
			unset(OpenCV_WORLD CACHE)
			
			
			##----------------------------------------------------------------------------------
			## Find the regular opencv libraries
			string(CONCAT _videostab_lib_   "opencv_videostab" ${OpenCV_LIBVERSION} ".lib" )
			find_library(OpenCV_VIDEOSTAB ${_videostab_lib_}  PATHS ${OpenCV_LIBRARY_DIR})
			if(OpenCV_VIDEOSTAB)
				set(OpenCV_VIDEOSTAB "optimized" ${OpenCV_VIDEOSTAB})
			endif()
			list(APPEND OpenCV_Lib_list ${OpenCV_VIDEOSTAB})

			string(CONCAT _videoio_lib_   "opencv_videoio" ${OpenCV_LIBVERSION} ".lib" )
			find_library(OpenCV_VIDEOIO ${_videoio_lib_}  PATHS ${OpenCV_LIBRARY_DIR})
			if(OpenCV_VIDEOIO)
				set(OpenCV_VIDEOIO "optimized" ${OpenCV_VIDEOIO})
				list(APPEND OpenCV_LIBS_EXP ${OpenCV_VIDEOIO})
			endif()
			list(APPEND OpenCV_Lib_list ${OpenCV_VIDEOIO})

			string(CONCAT _opencv_video_lib_   "opencv_video" ${OpenCV_LIBVERSION} ".lib" )
			find_library(OpenCV_VIDEO ${_opencv_video_lib_}  PATHS ${OpenCV_LIBRARY_DIR})
			if(OpenCV_VIDEO)
				set(OpenCV_VIDEO "optimized" ${OpenCV_VIDEO})
				list(APPEND OpenCV_LIBS_EXP ${OpenCV_VIDEO})
			endif()
			list(APPEND OpenCV_Lib_list ${OpenCV_VIDEO})

			string(CONCAT _opencv_ts_lib_   "opencv_ts" ${OpenCV_LIBVERSION} ".lib" )
			find_library(OpenCV_TS ${_opencv_ts_lib_}  PATHS ${OpenCV_LIBRARY_DIR})
			if(OpenCV_TS)
				set(OpenCV_TS "optimized" ${OpenCV_TS})
				list(APPEND OpenCV_LIBS_EXP ${OpenCV_TS})
			endif()
			list(APPEND OpenCV_Lib_list ${OpenCV_TS})
			
			string(CONCAT _opencv_superres_lib_   "opencv_superres" ${OpenCV_LIBVERSION} ".lib" )
			find_library(OpenCV_SUPERRES ${_opencv_superres_lib_}  PATHS ${OpenCV_LIBRARY_DIR})
			if(OpenCV_SUPERRES)
				set(OpenCV_SUPERRES "optimized" ${OpenCV_SUPERRES})
				list(APPEND OpenCV_LIBS_EXP ${OpenCV_SUPERRES})
			endif()
			list(APPEND OpenCV_Lib_list ${OpenCV_SUPERRES})
			
			string(CONCAT _opencv_stitching_lib_   "opencv_stitching" ${OpenCV_LIBVERSION} ".lib" )
			find_library(OpenCV_STITCHING ${_opencv_stitching_lib_}  PATHS ${OpenCV_LIBRARY_DIR})
			if(OpenCV_STITCHING)
				set(OpenCV_STITCHING "optimized" ${OpenCV_STITCHING})
			endif()
			list(APPEND OpenCV_Lib_list ${OpenCV_STITCHING})
			
			string(CONCAT _opencv_shape_lib_ "opencv_shape" ${OpenCV_LIBVERSION} ".lib" )
			find_library(OpenCV_SHAPE ${_opencv_shape_lib_}  PATHS ${OpenCV_LIBRARY_DIR})
			if(OpenCV_SHAPE)
				set(OpenCV_SHAPE "optimized" ${OpenCV_SHAPE})
			endif()
			list(APPEND OpenCV_Lib_list ${OpenCV_SHAPE})
			
			string(CONCAT _opencv_photo_lib_   "opencv_photo" ${OpenCV_LIBVERSION} ".lib" )
			find_library(OpenCV_PHOTO ${_opencv_photo_lib_}  PATHS ${OpenCV_LIBRARY_DIR})
			if(OpenCV_PHOTO)
				set(OpenCV_PHOTO "optimized" ${OpenCV_PHOTO})
			endif()
			list(APPEND OpenCV_Lib_list ${OpenCV_PHOTO})
			
			string(CONCAT _opencv_objdetect_lib_   "opencv_objdetect" ${OpenCV_LIBVERSION} ".lib" )
			find_library(OpenCV_OBJDETECT ${_opencv_objdetect_lib_}  PATHS ${OpenCV_LIBRARY_DIR})
			if(OpenCV_OBJDETECT)
				set(OpenCV_OBJDETECT "optimized" ${OpenCV_OBJDETECT})
			endif()
			list(APPEND OpenCV_Lib_list ${OpenCV_OBJDETECT})
			
			string(CONCAT _opencv_ml_lib_   "opencv_ml" ${OpenCV_LIBVERSION} ".lib" )
			find_library(OpenCV_ML ${_opencv_ml_lib_}  PATHS ${OpenCV_LIBRARY_DIR})
			if(OpenCV_ML)
				set(OpenCV_ML "optimized" ${OpenCV_ML})
			endif()
			list(APPEND OpenCV_Lib_list ${OpenCV_ML})

			string(CONCAT _opencv_imgproc_lib_   "opencv_imgproc" ${OpenCV_LIBVERSION} ".lib" )
			find_library(OpenCV_IMGPROC ${_opencv_imgproc_lib_}  PATHS ${OpenCV_LIBRARY_DIR})
			if(OpenCV_IMGPROC)
				set(OpenCV_IMGPROC "optimized" ${OpenCV_IMGPROC})
			endif()
			list(APPEND OpenCV_Lib_list ${OpenCV_IMGPROC})
			
			string(CONCAT _opencv_imgcodecs_lib_   "opencv_imgcodecs" ${OpenCV_LIBVERSION} ".lib" )
			find_library(OpenCV_IMGCODECS ${_opencv_imgcodecs_lib_}  PATHS ${OpenCV_LIBRARY_DIR})
			if(OpenCV_IMGCODECS)
				set(OpenCV_IMGCODECS "optimized" ${OpenCV_IMGCODECS})
			endif()
			list(APPEND OpenCV_Lib_list ${OpenCV_IMGCODECS})

			string(CONCAT _opencv_highgui_lib_   "opencv_highgui" ${OpenCV_LIBVERSION} ".lib" )
			find_library(OpenCV_HIGHGUI ${_opencv_highgui_lib_}  PATHS ${OpenCV_LIBRARY_DIR})
			if(OpenCV_HIGHGUI)
				set(OpenCV_HIGHGUI "optimized" ${OpenCV_HIGHGUI})
			endif()
			list(APPEND OpenCV_Lib_list ${OpenCV_HIGHGUI})
			
			string(CONCAT _opencv_flann_lib_   "opencv_flann" ${OpenCV_LIBVERSION} ".lib" )
			find_library(OpenCV_FLANN ${_opencv_flann_lib_}  PATHS ${OpenCV_LIBRARY_DIR})
			if(OpenCV_FLANN)
				set(OpenCV_FLANN "optimized" ${OpenCV_FLANN})
			endif()
			list(APPEND OpenCV_Lib_list ${OpenCV_FLANN})

			string(CONCAT _opencv_features2d_lib_   "opencv_features2d" ${OpenCV_LIBVERSION} ".lib" )
			find_library(OpenCV_FEATURED2D ${_opencv_features2d_lib_}  PATHS ${OpenCV_LIBRARY_DIR})
			if(OpenCV_FEATURED2D)
				set(OpenCV_FEATURED2D "optimized" ${OpenCV_FEATURED2D})
			endif()
			list(APPEND OpenCV_Lib_list ${OpenCV_FEATURED2D})
			
			string(CONCAT _opencv_dnn_lib_   "opencv_dnn" ${OpenCV_LIBVERSION} ".lib" )
			find_library(OpenCV_DNN ${_opencv_dnn_lib_}  PATHS ${OpenCV_LIBRARY_DIR})
			if(OpenCV_DNN)
				set(OpenCV_DNN "optimized" ${OpenCV_DNN})
			endif()
			list(APPEND OpenCV_Lib_list ${OpenCV_DNN})
			
			string(CONCAT _opencv_cudev_lib_   "opencv_cudev" ${OpenCV_LIBVERSION} ".lib" )
			find_library(OpenCV_CUDEV ${_opencv_cudev_lib_}  PATHS ${OpenCV_LIBRARY_DIR})
			if(OpenCV_CUDEV)
				set(OpenCV_CUDEV "optimized" ${OpenCV_CUDEV})
			endif()
			list(APPEND OpenCV_Lib_list ${OpenCV_CUDEV})
			
			string(CONCAT _opencv_core_lib_   "opencv_core" ${OpenCV_LIBVERSION} ".lib" )
			find_library(OpenCV_CORE ${_opencv_core_lib_}  PATHS ${OpenCV_LIBRARY_DIR})
			if(OpenCV_CORE)
				set(OpenCV_CORE "optimized" ${OpenCV_CORE})
			endif()
			list(APPEND OpenCV_Lib_list ${OpenCV_CORE})
			
			string(CONCAT _opencv_calib3d_lib_   "opencv_calib3d" ${OpenCV_LIBVERSION} ".lib" )
			find_library(OpenCV_CALIB3D ${_opencv_calib3d_lib_}  PATHS ${OpenCV_LIBRARY_DIR})
			if(OpenCV_CALIB3D)
				set(OpenCV_CALIB3D "optimized" ${OpenCV_CALIB3D})
			endif()
			list(APPEND OpenCV_Lib_list ${OpenCV_CALIB3D})
			
		endif ()

	else()
		message("[FindOpenCV] - ERROR - Did not find any OpenCV library")
	endif ()



	##------------------------------------------
	## Debug libraries

	if ( OpenCV_DEBUG_LIBRARY_DIR)

		## test first if opencv_world is used	
		if(NOT OpenCV_WORLD_DEBUG )
			string(CONCAT _worldd_lib_ "opencv_world" ${OpenCV_LIBVERSION} "d.lib" )
			
			find_file(__FIND_WORLD_DEBUG ${_worldd_lib_}  PATHS ${OpenCV_DEBUG_LIBRARY_DIR} ) 
			if(__FIND_WORLD_DEBUG)
				find_library(OpenCV_WORLD_DEBUG ${_worldd_lib_}  PATHS ${OpenCV_DEBUG_LIBRARY_DIR} OPTIONAL)
				if(OpenCV_WORLD_DEBUG)
					set(OpenCV_WORLD_DEBUG "debug" ${OpenCV_WORLD_DEBUG})
				endif()			
				set(OpenCV_LIBS ${OpenCV_WORLD_DEBUG}  CACHE PATH "Libraries")
			endif ()
			unset(__FIND_WORLD_DEBUG CACHE)
		endif()
		
		if(NOT OpenCV_WORLD_DEBUG)
			unset(OpenCV_WORLD_DEBUG CACHE)

			##----------------------------------------------------------------------------------
			## Find the regular opencv libraries, debug version
			string(CONCAT _videostabd_lib_   "opencv_videostab" ${OpenCV_LIBVERSION} "d.lib" )
			find_library(OpenCV_VIDEOSTAB_DEBUG ${_videostabd_lib_}  PATHS ${OpenCV_DEBUG_LIBRARY_DIR})
			if(OpenCV_VIDEOSTAB_DEBUG)
				set(OpenCV_VIDEOSTAB_DEBUG "debug" ${OpenCV_VIDEOSTAB_DEBUG})
			endif()
			
			string(CONCAT _videoiod_lib_   "opencv_videoio" ${OpenCV_LIBVERSION} "d.lib" )
			find_library(OpenCV_VIDEOIO_DEBUG ${_videoiod_lib_}  PATHS ${OpenCV_DEBUG_LIBRARY_DIR})
			if(OpenCV_VIDEOIO_DEBUG)
				set(OpenCV_VIDEOIO_DEBUG "debug" ${OpenCV_VIDEOIO_DEBUG})
			endif()

			string(CONCAT _opencv_videod_lib_   "opencv_video" ${OpenCV_LIBVERSION} "d.lib" )
			find_library(OpenCV_VIDEO_DEBUG ${_opencv_videod_lib_}  PATHS ${OpenCV_DEBUG_LIBRARY_DIR})
			if(OpenCV_VIDEO_DEBUG)
				set(OpenCV_VIDEO_DEBUG "debug" ${OpenCV_VIDEO_DEBUG})
			endif()

			string(CONCAT _opencv_tsd_lib_   "opencv_ts" ${OpenCV_LIBVERSION} "d.lib" )
			find_library(OpenCV_TS_DEBUG ${_opencv_tsd_lib_}  PATHS ${OpenCV_DEBUG_LIBRARY_DIR})
			if(OpenCV_TS_DEBUG)
				set(OpenCV_TS_DEBUG "debug" ${OpenCV_TS_DEBUG})
			endif()
			
			string(CONCAT _opencv_superresd_lib_   "opencv_superres" ${OpenCV_LIBVERSION} "d.lib" )
			find_library(OpenCV_SUPERRES_DEBUG ${_opencv_superresd_lib_}  PATHS ${OpenCV_DEBUG_LIBRARY_DIR})
			if(OpenCV_SUPERRES_DEBUG)
				set(OpenCV_SUPERRES_DEBUG "debug" ${OpenCV_SUPERRES_DEBUG})
			endif()
			
			string(CONCAT _opencv_stitchingd_lib_   "opencv_stitching" ${OpenCV_LIBVERSION} "d.lib" )
			find_library(OpenCV_STITCHING_DEBUG ${_opencv_stitchingd_lib_}  PATHS ${OpenCV_DEBUG_LIBRARY_DIR})
			if(OpenCV_STITCHING_DEBUG)
				set(OpenCV_STITCHING_DEBUG "debug" ${OpenCV_STITCHING_DEBUG})
			endif()
			
			string(CONCAT _opencv_shaped_lib_   "opencv_shape" ${OpenCV_LIBVERSION} "d.lib" )
			find_library(OpenCV_SHAPE_DEBUG ${_opencv_shaped_lib_}  PATHS ${OpenCV_DEBUG_LIBRARY_DIR})
			if(OpenCV_SHAPE_DEBUG)
				set(OpenCV_SHAPE_DEBUG "debug" ${OpenCV_SHAPE_DEBUG})
			endif()
			
			string(CONCAT _opencv_photod_lib_   "opencv_photo" ${OpenCV_LIBVERSION} "d.lib" )
			find_library(OpenCV_PHOTO_DEBUG ${_opencv_photod_lib_}  PATHS ${OpenCV_DEBUG_LIBRARY_DIR})
			if(OpenCV_PHOTO_DEBUG)
				set(OpenCV_PHOTO_DEBUG "debug" ${OpenCV_PHOTO_DEBUG})
			endif()
			
			string(CONCAT _opencv_objdetectd_lib_   "opencv_objdetect" ${OpenCV_LIBVERSION} "d.lib" )
			find_library(OpenCV_OBJDETECT_DEBUG ${_opencv_objdetectd_lib_}  PATHS ${OpenCV_DEBUG_LIBRARY_DIR})
			if(OpenCV_OBJDETECT_DEBUG)
				set(OpenCV_OBJDETECT_DEBUG "debug" ${OpenCV_OBJDETECT_DEBUG})
			endif()
			
			string(CONCAT _opencv_mld_lib_   "opencv_ml" ${OpenCV_LIBVERSION} "d.lib" )
			find_library(OpenCV_ML_DEBUG ${_opencv_mld_lib_}  PATHS ${OpenCV_DEBUG_LIBRARY_DIR})
			if(OpenCV_ML_DEBUG)
				set(OpenCV_ML_DEBUG "debug" ${OpenCV_ML_DEBUG})
			endif()

			string(CONCAT _opencv_imgprocd_lib_   "opencv_imgproc" ${OpenCV_LIBVERSION} "d.lib" )
			find_library(OpenCV_IMGPROC_DEBUG ${_opencv_imgprocd_lib_}  PATHS ${OpenCV_DEBUG_LIBRARY_DIR})
			if(OpenCV_IMGPROC_DEBUG)
				set(OpenCV_IMGPROC_DEBUG "debug" ${OpenCV_IMGPROC_DEBUG})
			endif()
			
			string(CONCAT _opencv_imgcodecsd_lib_   "opencv_imgcodecs" ${OpenCV_LIBVERSION} "d.lib" )
			find_library(OpenCV_IMGCODECS_DEBUG ${_opencv_imgcodecsd_lib_}  PATHS ${OpenCV_DEBUG_LIBRARY_DIR})
			if(OpenCV_IMGCODECS_DEBUG)
				set(OpenCV_IMGCODECS_DEBUG "debug" ${OpenCV_IMGCODECS_DEBUG})
			endif()

			string(CONCAT _opencv_highguid_lib_   "opencv_highgui" ${OpenCV_LIBVERSION} "d.lib" )
			find_library(OpenCV_HIGHGUI_DEBUG ${_opencv_highguid_lib_}  PATHS ${OpenCV_DEBUG_LIBRARY_DIR})
			if(OpenCV_HIGHGUI_DEBUG)
				set(OpenCV_HIGHGUI_DEBUG "debug" ${OpenCV_HIGHGUI_DEBUG})
			endif()
			
			string(CONCAT _opencv_flannd_lib_   "opencv_flann" ${OpenCV_LIBVERSION} "d.lib" )
			find_library(OpenCV_FLANN_DEBUG ${_opencv_flannd_lib_}  PATHS ${OpenCV_DEBUG_LIBRARY_DIR})
			if(OpenCV_FLANN_DEBUG)
				set(OpenCV_FLANN_DEBUG "debug" ${OpenCV_FLANN_DEBUG})
			endif()

			string(CONCAT _opencv_features2dd_lib_   "opencv_features2d" ${OpenCV_LIBVERSION} "d.lib" )
			find_library(OpenCV_FEATURED2D_DEBUG ${_opencv_features2dd_lib_}  PATHS ${OpenCV_DEBUG_LIBRARY_DIR})
			if(OpenCV_FEATURED2D_DEBUG)
				set(OpenCV_FEATURED2D_DEBUG "debug" ${OpenCV_FEATURED2D_DEBUG})
			endif()
			
			string(CONCAT _opencv_dnnd_lib_   "opencv_dnn" ${OpenCV_LIBVERSION} "d.lib" )
			find_library(OpenCV_DNN_DEBUG ${_opencv_dnnd_lib_}  PATHS ${OpenCV_DEBUG_LIBRARY_DIR})
			if(OpenCV_DNN_DEBUG)
				set(OpenCV_DNN_DEBUG "debug" ${OpenCV_DNN_DEBUG})
			endif()
			
			string(CONCAT _opencv_cudevd_lib_   "opencv_cudev" ${OpenCV_LIBVERSION} "d.lib" )
			find_library(OpenCV_CUDEV_DEBUG ${_opencv_cudevd_lib_}  PATHS ${OpenCV_DEBUG_LIBRARY_DIR})
			if(OpenCV_CUDEV_DEBUG)
				set(OpenCV_CUDEV_DEBUG "debug" ${OpenCV_CUDEV_DEBUG})
			endif()
			
			string(CONCAT _opencv_cored_lib_   "opencv_core" ${OpenCV_LIBVERSION} "d.lib" )
			find_library(OpenCV_CORE_DEBUG ${_opencv_cored_lib_}  PATHS ${OpenCV_DEBUG_LIBRARY_DIR})
			if(OpenCV_CORE_DEBUG)
				set(OpenCV_CORE_DEBUG "debug" ${OpenCV_CORE_DEBUG})
			endif()
			
			string(CONCAT _opencv_calib3dd_lib_   "opencv_calib3d" ${OpenCV_LIBVERSION} "d.lib" )
			find_library(OpenCV_CALIB3D_DEBUG ${_opencv_calib3dd_lib_}  PATHS ${OpenCV_DEBUG_LIBRARY_DIR})
			if(OpenCV_CALIB3D_DEBUG)
				set(OpenCV_CALIB3D_DEBUG "debug" ${OpenCV_CALIB3D_DEBUG})
			endif()
			

		endif ()
	else()
		message("[FindOpenCV] - ERROR - Did not find any OpenCV DEBUG library")
		set (OpenCV_FOUND FALSE CACHE PATH "Found opencv")
	endif ()


	##-----------------------------------------------------------------------------------------------------
	## Test for CUDA files


	if ( OpenCV_LIBRARY_DIR)

		string(CONCAT _cudawarping_lib_   "opencv_cudawarping" ${OpenCV_LIBVERSION} ".lib" )
		find_file(__find_cuda_warping ${_cudawarping_lib_}  PATHS ${OpenCV_LIBRARY_DIR} ) 
		if(__find_cuda_warping)
			find_library(OpenCV_CUDA_WARPING ${_cudawarping_lib_}  PATHS ${OpenCV_LIBRARY_DIR} )
			set(OpenCV_CUDA_WARPING "optimized" ${OpenCV_CUDA_WARPING})
		endif ()
		unset(__find_cuda_warping CACHE)

		string(CONCAT _cudastereo_lib_   "opencv_cudastereo" ${OpenCV_LIBVERSION} ".lib" )
		find_file(__find_cuda_stereo ${_cudastereo_lib_}  PATHS ${OpenCV_LIBRARY_DIR} ) 
		if(__find_cuda_stereo)
			find_library(OpenCV_CUDA_STEREO ${_cudastereo_lib_}  PATHS ${OpenCV_LIBRARY_DIR} )
			set(OpenCV_CUDA_STEREO "optimized" ${OpenCV_CUDA_STEREO})
		endif ()
		unset(__find_cuda_stereo CACHE)
		
		string(CONCAT _cudaoptflow_lib_   "opencv_cudaoptflow" ${OpenCV_LIBVERSION} ".lib" )
		find_file(__find_cuda_optflow ${_cudaoptflow_lib_}  PATHS ${OpenCV_LIBRARY_DIR} ) 
		if(__find_cuda_optflow)
			find_library(OpenCV_CUDA_OPTFLOW ${_cudaoptflow_lib_}  PATHS ${OpenCV_LIBRARY_DIR} )
			set(OpenCV_CUDA_OPTFLOW "optimized" ${OpenCV_CUDA_OPTFLOW})
		endif ()
		unset(__find_cuda_optflow CACHE)

		string(CONCAT _cudaobjdetect_lib_   "opencv_cudaobjdetect" ${OpenCV_LIBVERSION} ".lib" )
		find_file(__find__cuda_objdetect ${_cudaobjdetect_lib_}  PATHS ${OpenCV_LIBRARY_DIR} ) 
		if(__find__cuda_objdetect)
			find_library(OpenCV_CUDA_OBJDETECT ${_cudaobjdetect_lib_}  PATHS ${OpenCV_LIBRARY_DIR} )
			set(OpenCV_CUDA_OBJDETECT "optimized" ${OpenCV_CUDA_OBJDETECT})
		endif ()
		unset(__find__cuda_objdetect CACHE)

		string(CONCAT _cudalegacy_lib_   "opencv_cudalegacy" ${OpenCV_LIBVERSION} ".lib" )
		find_file(__find__cuda_legacy ${_cudalegacy_lib_}  PATHS ${OpenCV_LIBRARY_DIR} ) 
		if(__find__cuda_legacy)
			find_library(OpenCV_CUDA_LEGACY ${_cudalegacy_lib_}  PATHS ${OpenCV_LIBRARY_DIR} )
			set(OpenCV_CUDA_LEGACY "optimized" ${OpenCV_CUDA_LEGACY})
		endif ()
		unset(__find__cuda_legacy CACHE)

		string(CONCAT _cudaimgproc_lib_   "opencv_cudaimgproc" ${OpenCV_LIBVERSION} ".lib" )
		find_file(__find__cuda_imgproc ${_cudaimgproc_lib_}  PATHS ${OpenCV_LIBRARY_DIR} ) 
		if(__find__cuda_imgproc)
			find_library(OpenCV_CUDA_IMGPROC ${_cudaimgproc_lib_}  PATHS ${OpenCV_LIBRARY_DIR} )
			set(OpenCV_CUDA_IMGPROC "optimized" ${OpenCV_CUDA_IMGPROC})
		endif ()
		unset(__find__cuda_imgproc CACHE)

		string(CONCAT _cudafilters_lib_   "opencv_cudafilters" ${OpenCV_LIBVERSION} ".lib" )
		find_file(__find__cuda_filters ${_cudafilters_lib_}  PATHS ${OpenCV_LIBRARY_DIR} ) 
		if(__find__cuda_filters)
			find_library(OpenCV_CUDA_FILTERS ${_cudafilters_lib_}  PATHS ${OpenCV_LIBRARY_DIR} )
			set(OpenCV_CUDA_FILTERS "optimized" ${OpenCV_CUDA_FILTERS})
		endif ()
		unset(__find__cuda_filters CACHE)
		
		string(CONCAT _cudafeatures2d_lib_   "opencv_cudafeatures2d" ${OpenCV_LIBVERSION} ".lib" )
		find_file(__find__cuda_features2d ${_cudafeatures2d_lib_}  PATHS ${OpenCV_LIBRARY_DIR} ) 
		if(__find__cuda_features2d)
			find_library(OpenCV_CUDA_FEATURES2D ${_cudafeatures2d_lib_}  PATHS ${OpenCV_LIBRARY_DIR} )
			set(OpenCV_CUDA_FEATURES2D "optimized" ${OpenCV_CUDA_FEATURES2D})
		endif ()
		unset(__find__cuda_features2d CACHE)


		string(CONCAT _cudacodec_lib_   "opencv_cudacodec" ${OpenCV_LIBVERSION} ".lib" )
		find_file(__find__cuda_codec ${_cudacodec_lib_}  PATHS ${OpenCV_LIBRARY_DIR} ) 
		if(__find__cuda_codec)
			find_library(OpenCV_CUDA_CODEC ${_cudacodec_lib_}  PATHS ${OpenCV_LIBRARY_DIR} )
			set(OpenCV_CUDA_CODEC "optimized" ${OpenCV_CUDA_CODEC})
		endif ()
		unset(__find__cuda_codec CACHE)

		string(CONCAT _cudabgsegm_lib_   "opencv_cudabgsegm" ${OpenCV_LIBVERSION} ".lib" )
		find_file(__find__cuda_bgsegm ${_cudabgsegm_lib_}  PATHS ${OpenCV_LIBRARY_DIR} ) 
		if(__find__cuda_bgsegm)
			find_library(OpenCV_CUDA_BGSEGM ${_cudabgsegm_lib_}  PATHS ${OpenCV_LIBRARY_DIR} )
			set(OpenCV_CUDA_BGSEGM "optimized" ${OpenCV_CUDA_BGSEGM})
		endif ()
		unset(__find__cuda_bgsegm CACHE)
		
		string(CONCAT _cudaarithm_lib_   "opencv_cudaarithm" ${OpenCV_LIBVERSION} ".lib" )
		find_file(__find__cuda_arithm ${_cudaarithm_lib_}  PATHS ${OpenCV_LIBRARY_DIR} ) 
		if(__find__cuda_arithm)
			find_library(OpenCV_CUDA_ARITHM ${_cudaarithm_lib_}  PATHS ${OpenCV_LIBRARY_DIR} )
			set(OpenCV_CUDA_ARITHM "optimized" ${OpenCV_CUDA_ARITHM})
		endif ()
		unset(__find__cuda_arithm CACHE)
		
		unset(OpenCV_LIBRARIES CACHE)
		
		
	endif()



	if ( OpenCV_DEBUG_LIBRARY_DIR)

		string(CONCAT _cudawarping_lib_   "opencv_cudawarping" ${OpenCV_LIBVERSION} "d.lib" )
		find_file(__find_cuda_warping ${_cudawarping_lib_}  PATHS ${OpenCV_DEBUG_LIBRARY_DIR} ) 
		if(__find_cuda_warping)
			find_library(OpenCV_CUDA_WARPING_DEBUG ${_cudawarping_lib_}  PATHS ${OpenCV_DEBUG_LIBRARY_DIR} )
			set(OpenCV_CUDA_WARPING_DEBUG "debug" ${OpenCV_CUDA_WARPING_DEBUG})
		endif ()
		unset(__find_cuda_warping CACHE)

		string(CONCAT _cudastereo_lib_   "opencv_cudastereo" ${OpenCV_LIBVERSION} "d.lib" )
		find_file(__find_cuda_stereo ${_cudastereo_lib_}  PATHS ${OpenCV_DEBUG_LIBRARY_DIR} ) 
		if(__find_cuda_stereo)
			find_library(OpenCV_CUDA_STEREO_DEBUG ${_cudastereo_lib_}  PATHS ${OpenCV_DEBUG_LIBRARY_DIR} )
			set(OpenCV_CUDA_STEREO_DEBUG "debug" ${OpenCV_CUDA_STEREO_DEBUG})
		endif ()
		unset(__find_cuda_stereo CACHE)
		
		string(CONCAT _cudaoptflow_lib_   "opencv_cudaoptflow" ${OpenCV_LIBVERSION} "d.lib" )
		find_file(__find_cuda_optflow ${_cudaoptflow_lib_}  PATHS ${OpenCV_DEBUG_LIBRARY_DIR} ) 
		if(__find_cuda_optflow)
			find_library(OpenCV_CUDA_OPTFLOW_DEBUG ${_cudaoptflow_lib_}  PATHS ${OpenCV_DEBUG_LIBRARY_DIR} )
			set(OpenCV_CUDA_OPTFLOW_DEBUG "debug" ${OpenCV_CUDA_OPTFLOW_DEBUG})
		endif ()
		unset(__find_cuda_optflow CACHE)

		string(CONCAT _cudaobjdetect_lib_   "opencv_cudaobjdetect" ${OpenCV_LIBVERSION} "d.lib" )
		find_file(__find__cuda_objdetect ${_cudaobjdetect_lib_}  PATHS ${OpenCV_DEBUG_LIBRARY_DIR} ) 
		if(__find__cuda_objdetect)
			find_library(OpenCV_CUDA_OBJDETECT_DEBUG ${_cudaobjdetect_lib_}  PATHS ${OpenCV_DEBUG_LIBRARY_DIR} )
			set(OpenCV_CUDA_OBJDETECT_DEBUG "debug" ${OpenCV_CUDA_OBJDETECT_DEBUG})
		endif ()
		unset(__find__cuda_objdetect CACHE)

		string(CONCAT _cudalegacy_lib_   "opencv_cudalegacy" ${OpenCV_LIBVERSION} "d.lib" )
		find_file(__find__cuda_legacy ${_cudalegacy_lib_}  PATHS ${OpenCV_DEBUG_LIBRARY_DIR} ) 
		if(__find__cuda_legacy)
			find_library(OpenCV_CUDA_LEGACY_DEBUG ${_cudalegacy_lib_}  PATHS ${OpenCV_DEBUG_LIBRARY_DIR} )
			set(OpenCV_CUDA_LEGACY_DEBUG "debug" ${OpenCV_CUDA_LEGACY_DEBUG})
		endif ()
		unset(__find__cuda_legacy CACHE)

		string(CONCAT _cudaimgproc_lib_   "opencv_cudaimgproc" ${OpenCV_LIBVERSION} "d.lib" )
		find_file(__find__cuda_imgproc ${_cudaimgproc_lib_}  PATHS ${OpenCV_DEBUG_LIBRARY_DIR} ) 
		if(__find__cuda_imgproc)
			find_library(OpenCV_CUDA_IMGPROC_DEBUG ${_cudaimgproc_lib_}  PATHS ${OpenCV_DEBUG_LIBRARY_DIR} )
			set(OpenCV_CUDA_IMGPROC_DEBUG "debug" ${OpenCV_CUDA_IMGPROC_DEBUG})
		endif ()
		unset(__find__cuda_imgproc CACHE)

		string(CONCAT _cudafilters_lib_   "opencv_cudafilters" ${OpenCV_LIBVERSION} "d.lib" )
		find_file(__find__cuda_filters ${_cudafilters_lib_}  PATHS ${OpenCV_DEBUG_LIBRARY_DIR} ) 
		if(__find__cuda_filters)
			find_library(OpenCV_CUDA_FILTERS_DEBUG ${_cudafilters_lib_}  PATHS ${OpenCV_DEBUG_LIBRARY_DIR} )
			set(OpenCV_CUDA_FILTERS_DEBUG "debug" ${OpenCV_CUDA_FILTERS_DEBUG})
		endif ()
		unset(__find__cuda_filters CACHE)
		
		string(CONCAT _cudafeatures2d_lib_   "opencv_cudafeatures2d" ${OpenCV_LIBVERSION} "d.lib" )
		find_file(__find__cuda_features2d ${_cudafeatures2d_lib_}  PATHS ${OpenCV_DEBUG_LIBRARY_DIR} ) 
		if(__find__cuda_features2d)
			find_library(OpenCV_CUDA_FEATURES2D_DEBUG ${_cudafeatures2d_lib_}  PATHS ${OpenCV_DEBUG_LIBRARY_DIR} )
			set(OpenCV_CUDA_FEATURES2D_DEBUG "debug" ${OpenCV_CUDA_FEATURES2D_DEBUG})
		endif ()
		unset(__find__cuda_features2d CACHE)


		string(CONCAT _cudacodec_lib_   "opencv_cudacodec" ${OpenCV_LIBVERSION} "d.lib" )
		find_file(__find__cuda_codec ${_cudacodec_lib_}  PATHS ${OpenCV_DEBUG_LIBRARY_DIR} ) 
		if(__find__cuda_codec)
			find_library(OpenCV_CUDA_CODEC_DEBUG ${_cudacodec_lib_}  PATHS ${OpenCV_DEBUG_LIBRARY_DIR} )
			set(OpenCV_CUDA_CODEC_DEBUG "debug" ${OpenCV_CUDA_CODEC_DEBUG})
		endif ()
		unset(__find__cuda_codec CACHE)

		string(CONCAT _cudabgsegm_lib_   "opencv_cudabgsegm" ${OpenCV_LIBVERSION} "d.lib" )
		find_file(__find__cuda_bgsegm ${_cudabgsegm_lib_}  PATHS ${OpenCV_DEBUG_LIBRARY_DIR} ) 
		if(__find__cuda_bgsegm)
			find_library(OpenCV_CUDA_BGSEGM_DEBUG ${_cudabgsegm_lib_}  PATHS ${OpenCV_DEBUG_LIBRARY_DIR} )
			set(OpenCV_CUDA_BGSEGM_DEBUG "debug" ${OpenCV_CUDA_BGSEGM_DEBUG})
		endif ()
		unset(__find__cuda_bgsegm CACHE)
		
		string(CONCAT _cudaarithm_lib_   "opencv_cudaarithm" ${OpenCV_LIBVERSION} "d.lib" )
		find_file(__find__cuda_arithm ${_cudaarithm_lib_}  PATHS ${OpenCV_DEBUG_LIBRARY_DIR} ) 
		if(__find__cuda_arithm)
			find_library(OpenCV_CUDA_ARITHM_DEBUG ${_cudaarithm_lib_}  PATHS ${OpenCV_DEBUG_LIBRARY_DIR} )
			set(OpenCV_CUDA_ARITHM_DEBUG "debug" ${OpenCV_CUDA_ARITHM_DEBUG})
		endif ()
		unset(__find__cuda_arithm CACHE)
		
		unset(OpenCV_LIBRARIES_DEBUG CACHE)
			
		
		
	endif() 
	
	# for future improvements
	#list(LENGTH OpenCV_Lib_list __length )
	#foreach(arg ${OpenCV_Lib_list})
	#	message(${arg})
	#endforeach(arg)
	
	set(OpenCV_LIBS
		${OpenCV_CALIB3D} ${OpenCV_CORE} ${OpenCV_CUDEV} 
		${OpenCV_DNN} ${OpenCV_FEATURED2D} ${OpenCV_FLANN} ${OpenCV_HIGHGUI} ${OpenCV_IMGCODECS}
		${OpenCV_IMGPROC} ${OpenCV_ML} ${OpenCV_OBJDETECT} ${OpenCV_PHOTO} ${OpenCV_SHAPE}
		${OpenCV_STITCHING} ${OpenCV_SUPERRES} ${OpenCV_TS}  ${OpenCV_VIDEO}  ${OpenCV_VIDEOIO}
		${OpenCV_VIDEOSTAB} ${OpenCV_CUDA_ARITHM} ${OpenCV_CUDA_BGSEGM}
		${OpenCV_CUDA_CODEC} ${OpenCV_CUDA_FEATURES2D} ${OpenCV_CUDA_FILTERS}  ${OpenCV_CUDA_IMGPROC} 
		${OpenCV_CUDA_LEGACY} ${OpenCV_CUDA_OBJDETECT} ${OpenCV_CUDA_OPTFLOW} ${OpenCV_CUDA_STEREO} 
		${OpenCV_CUDA_WARPING}
		${OpenCV_CALIB3D_DEBUG} ${OpenCV_CORE_DEBUG} ${OpenCV_CUDEV_DEBUG}
		${OpenCV_DNN_DEBUG} ${OpenCV_FEATURED2D_DEBUG} ${OpenCV_FLANN_DEBUG} ${OpenCV_HIGHGUI_DEBUG} 
		${OpenCV_IMGCODECS_DEBUG} ${OpenCV_IMGPROC_DEBUG}  ${OpenCV_ML_DEBUG} ${OpenCV_OBJDETECT_DEBUG} 
		${OpenCV_PHOTO_DEBUG} ${OpenCV_SHAPE_DEBUG} ${OpenCV_STITCHING_DEBUG} ${OpenCV_SUPERRES_DEBUG} 
		${OpenCV_TS_DEBUG} ${OpenCV_VIDEO_DEBUG} ${OpenCV_VIDEOIO_DEBUG} ${OpenCV_VIDEOSTAB_DEBUG}
		${OpenCV_CUDA_ARITHM_DEBUG} ${OpenCV_CUDA_BGSEGM_DEBUG}
		${OpenCV_CUDA_CODEC_DEBUG} ${OpenCV_CUDA_FEATURES2D_DEBUG} ${OpenCV_CUDA_FILTERS_DEBUG} ${OpenCV_CUDA_IMGPROC_DEBUG} 
		${OpenCV_CUDA_LEGACY_DEBUG} ${OpenCV_CUDA_OBJDETECT_DEBUG} ${OpenCV_CUDA_OPTFLOW_DEBUG} ${OpenCV_CUDA_STEREO_DEBUG} 
		${OpenCV_CUDA_WARPING_DEBUG}
		CACHE PATH "Libraries")

	set (OpenCV_FOUND TRUE CACHE PATH "Found opencv")


elseif (${OPENCV_VERSION_MAJOR} EQUAL "2")

##----------------------------------------------------------------------------------------------------
## Version 2 support

	if ( OpenCV_LIBRARY_DIR)

		## test first if opencv_world is used	
		if(NOT OpenCV_WORLD )
			string(CONCAT _world_lib_   "opencv_world" ${OpenCV_LIBVERSION} ".lib" )
			find_file(__FIND_WORLD ${_world_lib_}  PATHS ${OpenCV_LIBRARY_DIR} ) 
			if(__FIND_WORLD)
				find_library(OpenCV_WORLD ${_world_lib_}  PATHS ${OpenCV_LIBRARY_DIR} OPTIONAL)
				set(OpenCV_WORLD "optimized" ${OpenCV_WORLD})
				set(OpenCV_LIBS ${OpenCV_WORLD} CACHE PATH "Libraries")		
			endif ()
			unset(__FIND_WORLD CACHE)
		endif()
		
		if(NOT OpenCV_WORLD)
			unset(OpenCV_WORLD CACHE)
			
			
			##----------------------------------------------------------------------------------
			## Find the regular opencv libraries
			string(CONCAT _videostab_lib_   "opencv_videostab" ${OpenCV_LIBVERSION} ".lib" )
			find_library(OpenCV_VIDEOSTAB ${_videostab_lib_}  PATHS ${OpenCV_LIBRARY_DIR})
			if(OpenCV_VIDEOSTAB)
				set(OpenCV_VIDEOSTAB "optimized" ${OpenCV_VIDEOSTAB})
			endif()

			string(CONCAT _opencv_video_lib_   "opencv_video" ${OpenCV_LIBVERSION} ".lib" )
			find_library(OpenCV_VIDEO ${_opencv_video_lib_}  PATHS ${OpenCV_LIBRARY_DIR})
			if(OpenCV_VIDEO)
				set(OpenCV_VIDEO "optimized" ${OpenCV_VIDEO})
			endif()

			string(CONCAT _opencv_ts_lib_   "opencv_ts" ${OpenCV_LIBVERSION} ".lib" )
			find_library(OpenCV_TS ${_opencv_ts_lib_}  PATHS ${OpenCV_LIBRARY_DIR})
			if(OpenCV_TS)
				set(OpenCV_TS "optimized" ${OpenCV_TS})
			endif()
			
			string(CONCAT _opencv_superres_lib_   "opencv_superres" ${OpenCV_LIBVERSION} ".lib" )
			find_library(OpenCV_SUPERRES ${_opencv_superres_lib_}  PATHS ${OpenCV_LIBRARY_DIR})
			if(OpenCV_SUPERRES)
				set(OpenCV_SUPERRES "optimized" ${OpenCV_SUPERRES})
			endif()
			
			string(CONCAT _opencv_stitching_lib_   "opencv_stitching" ${OpenCV_LIBVERSION} ".lib" )
			find_library(OpenCV_STITCHING ${_opencv_stitching_lib_}  PATHS ${OpenCV_LIBRARY_DIR})
			if(OpenCV_STITCHING)
				set(OpenCV_STITCHING "optimized" ${OpenCV_STITCHING})
			endif()
			
			string(CONCAT _opencv_photo_lib_   "opencv_photo" ${OpenCV_LIBVERSION} ".lib" )
			find_library(OpenCV_PHOTO ${_opencv_photo_lib_}  PATHS ${OpenCV_LIBRARY_DIR})
			if(OpenCV_PHOTO)
				set(OpenCV_PHOTO "optimized" ${OpenCV_PHOTO})
			endif()
			
			string(CONCAT _opencv_ocl_lib_   "opencv_ocl" ${OpenCV_LIBVERSION} ".lib" )
			find_library(OpenCV_OCL ${_opencv_ocl_lib_}  PATHS ${OpenCV_LIBRARY_DIR})
			if(OpenCV_OCL)
				set(OpenCV_OCL "optimized" ${OpenCV_OCL})
			endif()
			
			string(CONCAT _opencv_objdetect_lib_   "opencv_objdetect" ${OpenCV_LIBVERSION} ".lib" )
			find_library(OpenCV_OBJDETECT ${_opencv_objdetect_lib_}  PATHS ${OpenCV_LIBRARY_DIR})
			if(OpenCV_OBJDETECT)
				set(OpenCV_OBJDETECT "optimized" ${OpenCV_OBJDETECT})
			endif()
			
			string(CONCAT _opencv_nonfree_lib_   "opencv_nonfree" ${OpenCV_LIBVERSION} ".lib" )
			find_library(OpenCV_NONFREE ${_opencv_nonfree_lib_}  PATHS ${OpenCV_LIBRARY_DIR})
			if(OpenCV_NONFREE)
				set(OpenCV_NONFREE "optimized" ${OpenCV_NONFREE})
			endif()
			
			string(CONCAT _opencv_ml_lib_   "opencv_ml" ${OpenCV_LIBVERSION} ".lib" )
			find_library(OpenCV_ML ${_opencv_ml_lib_}  PATHS ${OpenCV_LIBRARY_DIR})
			if(OpenCV_ML)
				set(OpenCV_ML "optimized" ${OpenCV_ML})
			endif()
			
			string(CONCAT _opencv_legacy_lib_   "opencv_legacy" ${OpenCV_LIBVERSION} ".lib" )
			find_library(OpenCV_LEGACY ${_opencv_legacy_lib_}  PATHS ${OpenCV_LIBRARY_DIR})
			if(OpenCV_LEGACY)
				set(OpenCV_LEGACY "optimized" ${OpenCV_LEGACY})
			endif()

			string(CONCAT _opencv_imgproc_lib_   "opencv_imgproc" ${OpenCV_LIBVERSION} ".lib" )
			find_library(OpenCV_IMGPROC ${_opencv_imgproc_lib_}  PATHS ${OpenCV_LIBRARY_DIR})
			if(OpenCV_IMGPROC)
				set(OpenCV_IMGPROC "optimized" ${OpenCV_IMGPROC})
			endif()

			string(CONCAT _opencv_highgui_lib_   "opencv_highgui" ${OpenCV_LIBVERSION} ".lib" )
			find_library(OpenCV_HIGHGUI ${_opencv_highgui_lib_}  PATHS ${OpenCV_LIBRARY_DIR})
			if(OpenCV_HIGHGUI)
				set(OpenCV_HIGHGUI "optimized" ${OpenCV_HIGHGUI})
			endif()
			
			string(CONCAT _opencv_gpu_lib_   "opencv_gpu" ${OpenCV_LIBVERSION} ".lib" )
			find_library(OpenCV_GPU ${_opencv_gpu_lib_}  PATHS ${OpenCV_LIBRARY_DIR})
			if(OpenCV_GPU)
				set(OpenCV_GPU "optimized" ${OpenCV_GPU})
			endif()
			
			string(CONCAT _opencv_flann_lib_   "opencv_flann" ${OpenCV_LIBVERSION} ".lib" )
			find_library(OpenCV_FLANN ${_opencv_flann_lib_}  PATHS ${OpenCV_LIBRARY_DIR})
			if(OpenCV_FLANN)
				set(OpenCV_FLANN "optimized" ${OpenCV_FLANN})
			endif()

			string(CONCAT _opencv_features2d_lib_   "opencv_features2d" ${OpenCV_LIBVERSION} ".lib" )
			find_library(OpenCV_FEATURED2D ${_opencv_features2d_lib_}  PATHS ${OpenCV_LIBRARY_DIR})
			if(OpenCV_FEATURED2D)
				set(OpenCV_FEATURED2D "optimized" ${OpenCV_FEATURED2D})
			endif()
			
			string(CONCAT _opencv_core_lib_   "opencv_core" ${OpenCV_LIBVERSION} ".lib" )
			find_library(OpenCV_CORE ${_opencv_core_lib_}  PATHS ${OpenCV_LIBRARY_DIR})
			if(OpenCV_CORE)
				set(OpenCV_CORE "optimized" ${OpenCV_CORE})
			endif()
			
			string(CONCAT _opencv_contrib_lib_   "opencv_contrib" ${OpenCV_LIBVERSION} ".lib" )
			find_library(OpenCV_CONTRIB ${_opencv_contrib_lib_}  PATHS ${OpenCV_LIBRARY_DIR})
			if(OpenCV_CONTRIB)
				set(OpenCV_CONTRIB "optimized" ${OpenCV_CONTRIB})
			endif()
			
			string(CONCAT _opencv_calib3d_lib_   "opencv_calib3d" ${OpenCV_LIBVERSION} ".lib" )
			find_library(OpenCV_CALIB3D ${_opencv_calib3d_lib_}  PATHS ${OpenCV_LIBRARY_DIR})
			if(OpenCV_CALIB3D)
				set(OpenCV_CALIB3D "optimized" ${OpenCV_CALIB3D})
			endif()
			
		endif ()
		
		##-----------------------------------------------
		## DEBUG
				
		if ( OpenCV_DEBUG_LIBRARY_DIR)

			## test first if opencv_world is used	
			if(NOT OpenCV_WORLD )
				string(CONCAT _world_lib_ "opencv_world" ${OpenCV_LIBVERSION} "d.lib" )
				find_file(__FIND_WORLD ${_world_lib_}  PATHS ${OpenCV_DEBUG_LIBRARY_DIR} ) 
				if(__FIND_WORLD)
					find_library(OpenCV_WORLD_DEBUG ${_world_lib_}  PATHS ${OpenCV_DEBUG_LIBRARY_DIR} OPTIONAL)
					set(OpenCV_WORLD_DEBUG "debug" ${OpenCV_WORLD_DEBUG})
					set(OpenCV_LIBS ${OpenCV_WORLD_DEBUG}  CACHE PATH "Libraries")
				endif ()
				unset(__FIND_WORLD CACHE)
			endif()
		endif()
		
		if(NOT OpenCV_WORLD)
			unset(OpenCV_WORLD CACHE)
			
			
			##----------------------------------------------------------------------------------
			## Find the regular opencv libraries
			string(CONCAT _videostab_lib_   "opencv_videostab" ${OpenCV_LIBVERSION} "d.lib" )
			find_library(OpenCV_VIDEOSTAB_DEBUG ${_videostab_lib_}  PATHS ${OpenCV_DEBUG_LIBRARY_DIR})
			if(OpenCV_VIDEOSTAB_DEBUG)
				set(OpenCV_VIDEOSTAB_DEBUG "debug" ${OpenCV_VIDEOSTAB_DEBUG})
			endif()

			string(CONCAT _opencv_video_lib_   "opencv_video" ${OpenCV_LIBVERSION} "d.lib" )
			find_library(OpenCV_VIDEO_DEBUG ${_opencv_video_lib_}  PATHS ${OpenCV_DEBUG_LIBRARY_DIR})
			if(OpenCV_VIDEO_DEBUG)
				set(OpenCV_VIDEO_DEBUG "debug" ${OpenCV_VIDEO_DEBUG})
			endif()

			string(CONCAT _opencv_ts_lib_   "opencv_ts" ${OpenCV_LIBVERSION} "d.lib" )
			find_library(OpenCV_TS_DEBUG ${_opencv_ts_lib_}  PATHS ${OpenCV_DEBUG_LIBRARY_DIR})
			if(OpenCV_TS_DEBUG)
				set(OpenCV_TS_DEBUG "debug" ${OpenCV_TS_DEBUG})
			endif()
			
			string(CONCAT _opencv_superres_lib_   "opencv_superres" ${OpenCV_LIBVERSION} "d.lib" )
			find_library(OpenCV_SUPERRES_DEBUG ${_opencv_superres_lib_}  PATHS ${OpenCV_DEBUG_LIBRARY_DIR})
			if(OpenCV_SUPERRES_DEBUG)
				set(OpenCV_SUPERRES_DEBUG "debug" ${OpenCV_SUPERRES_DEBUG})
			endif()
			
			string(CONCAT _opencv_stitching_lib_   "opencv_stitching" ${OpenCV_LIBVERSION} "d.lib" )
			find_library(OpenCV_STITCHING_DEBUG ${_opencv_stitching_lib_}  PATHS ${OpenCV_DEBUG_LIBRARY_DIR})
			if(OpenCV_STITCHING_DEBUG)
				set(OpenCV_STITCHING_DEBUG "debug" ${OpenCV_STITCHING_DEBUG})
			endif()
			
			string(CONCAT _opencv_photo_lib_   "opencv_photo" ${OpenCV_LIBVERSION} "d.lib" )
			find_library(OpenCV_PHOTO_DEBUG ${_opencv_photo_lib_}  PATHS ${OpenCV_DEBUG_LIBRARY_DIR})
			if(OpenCV_PHOTO_DEBUG)
				set(OpenCV_PHOTO_DEBUG "debug" ${OpenCV_PHOTO_DEBUG})
			endif()
			
			string(CONCAT _opencv_ocl_lib_   "opencv_ocl" ${OpenCV_LIBVERSION} "d.lib" )
			find_library(OpenCV_OCL_DEBUG ${_opencv_ocl_lib_}  PATHS ${OpenCV_DEBUG_LIBRARY_DIR})
			if(OpenCV_OCL_DEBUG)
				set(OpenCV_OCL_DEBUG "debug" ${OpenCV_OCL_DEBUG})
			endif()
			
			string(CONCAT _opencv_objdetect_lib_   "opencv_objdetect" ${OpenCV_LIBVERSION} "d.lib" )
			find_library(OpenCV_OBJDETECT_DEBUG ${_opencv_objdetect_lib_}  PATHS ${OpenCV_DEBUG_LIBRARY_DIR})
			if(OpenCV_OBJDETECT_DEBUG)
				set(OpenCV_OBJDETECT_DEBUG "debug" ${OpenCV_OBJDETECT_DEBUG})
			endif()
			
			string(CONCAT _opencv_nonfree_lib_   "opencv_nonfree" ${OpenCV_LIBVERSION} "d.lib" )
			find_library(OpenCV_NONFREE_DEBUG ${_opencv_nonfree_lib_}  PATHS ${OpenCV_DEBUG_LIBRARY_DIR})
			if(OpenCV_NONFREE_DEBUG)
				set(OpenCV_NONFREE_DEBUG "debug" ${OpenCV_NONFREE_DEBUG})
			endif()
			
			string(CONCAT _opencv_ml_lib_   "opencv_ml" ${OpenCV_LIBVERSION} "d.lib" )
			find_library(OpenCV_ML_DEBUG ${_opencv_ml_lib_}  PATHS ${OpenCV_DEBUG_LIBRARY_DIR})
			if(OpenCV_ML_DEBUG)
				set(OpenCV_ML_DEBUG "debug" ${OpenCV_ML_DEBUG})
			endif()
			
			string(CONCAT _opencv_legacy_lib_   "opencv_legacy" ${OpenCV_LIBVERSION} "d.lib" )
			find_library(OpenCV_LEGACY_DEBUG ${_opencv_legacy_lib_}  PATHS ${OpenCV_DEBUG_LIBRARY_DIR})
			if(OpenCV_LEGACY_DEBUG)
				set(OpenCV_LEGACY_DEBUG "debug" ${OpenCV_LEGACY_DEBUG})
			endif()

			string(CONCAT _opencv_imgproc_lib_   "opencv_imgproc" ${OpenCV_LIBVERSION} "d.lib" )
			find_library(OpenCV_IMGPROC_DEBUG ${_opencv_imgproc_lib_}  PATHS ${OpenCV_DEBUG_LIBRARY_DIR})
			if(OpenCV_IMGPROC_DEBUG)
				set(OpenCV_IMGPROC_DEBUG "debug" ${OpenCV_IMGPROC_DEBUG})
			endif()

			string(CONCAT _opencv_highgui_lib_   "opencv_highgui" ${OpenCV_LIBVERSION} "d.lib" )
			find_library(OpenCV_HIGHGUI_DEBUG ${_opencv_highgui_lib_}  PATHS ${OpenCV_DEBUG_LIBRARY_DIR})
			if(OpenCV_HIGHGUI_DEBUG)
				set(OpenCV_HIGHGUI_DEBUG "debug" ${OpenCV_HIGHGUI_DEBUG})
			endif()
			
			string(CONCAT _opencv_gpu_lib_   "opencv_gpu" ${OpenCV_LIBVERSION} "d.lib" )
			find_library(OpenCV_GPU_DEBUG ${_opencv_gpu_lib_}  PATHS ${OpenCV_DEBUG_LIBRARY_DIR})
			if(OpenCV_GPU_DEBUG)
				set(OpenCV_GPU_DEBUG "debug" ${OpenCV_GPU_DEBUG})
			endif()
			
			string(CONCAT _opencv_flann_lib_   "opencv_flann" ${OpenCV_LIBVERSION} "d.lib" )
			find_library(OpenCV_FLANN_DEBUG ${_opencv_flann_lib_}  PATHS ${OpenCV_DEBUG_LIBRARY_DIR})
			if(OpenCV_FLANN_DEBUG)
				set(OpenCV_FLANN_DEBUG "debug" ${OpenCV_FLANN_DEBUG})
			endif()

			string(CONCAT _opencv_features2d_lib_   "opencv_features2d" ${OpenCV_LIBVERSION} "d.lib" )
			find_library(OpenCV_FEATURED2D_DEBUG ${_opencv_features2d_lib_}  PATHS ${OpenCV_DEBUG_LIBRARY_DIR})
			if(OpenCV_FEATURED2D_DEBUG)
				set(OpenCV_FEATURED2D_DEBUG "debug" ${OpenCV_FEATURED2D_DEBUG})
			endif()
			
			string(CONCAT _opencv_core_lib_   "opencv_core" ${OpenCV_LIBVERSION} "d.lib" )
			find_library(OpenCV_CORE_DEBUG ${_opencv_core_lib_}  PATHS ${OpenCV_DEBUG_LIBRARY_DIR})
			if(OpenCV_CORE_DEBUG)
				set(OpenCV_CORE_DEBUG "debug" ${OpenCV_CORE_DEBUG})
			endif()
			
			string(CONCAT _opencv_contrib_lib_   "opencv_contrib" ${OpenCV_LIBVERSION} "d.lib" )
			find_library(OpenCV_CONTRIB_DEBUG ${_opencv_contrib_lib_}  PATHS ${OpenCV_DEBUG_LIBRARY_DIR})
			if(OpenCV_CONTRIB_DEBUG)
				set(OpenCV_CONTRIB_DEBUG "debug" ${OpenCV_CONTRIB_DEBUG})
			endif()
			
			string(CONCAT _opencv_calib3d_lib_   "opencv_calib3d" ${OpenCV_LIBVERSION} "d.lib" )
			find_library(OpenCV_CALIB3D_DEBUG ${_opencv_calib3d_lib_}  PATHS ${OpenCV_DEBUG_LIBRARY_DIR})
			if(OpenCV_CALIB3D_DEBUG)
				set(OpenCV_CALIB3D_DEBUG "debug" ${OpenCV_CALIB3D_DEBUG})
			endif()
			

		endif ()
		
		set(OpenCV_LIBS
			${OpenCV_CALIB3D}  
			${OpenCV_CONTRIB} 
			${OpenCV_CORE}
			${OpenCV_FEATURED2D} 
			${OpenCV_FLANN} 
			${OpenCV_GPU} 
			${OpenCV_HIGHGUI} 
			${OpenCV_IMGPROC}
			${OpenCV_LEGACY} 
			${OpenCV_ML} 
			${OpenCV_NONFREE}
			${OpenCV_OBJDETECT} 
			${OpenCV_OCL}
			${OpenCV_PHOTO}
			${OpenCV_STITCHING}
			${OpenCV_SUPERRES} 
			${OpenCV_TS} 
			${OpenCV_VIDEO}
			${OpenCV_VIDEOSTAB}
			${OpenCV_CALIB3D_DEBUG}  
			${OpenCV_CONTRIB_DEBUG} 
			${OpenCV_CORE_DEBUG}
			${OpenCV_FEATURED2D_DEBUG} 
			${OpenCV_FLANN_DEBUG} 
			${OpenCV_GPU_DEBUG} 
			${OpenCV_HIGHGUI_DEBUG} 
			${OpenCV_IMGPROC_DEBUG}
			${OpenCV_LEGACY_DEBUG} 
			${OpenCV_ML_DEBUG} 
			${OpenCV_NONFREE_DEBUG}
			${OpenCV_OBJDETECT_DEBUG} 
			${OpenCV_OCL_DEBUG}
			${OpenCV_PHOTO_DEBUG}
			${OpenCV_STITCHING_DEBUG} 
			${OpenCV_SUPERRES_DEBUG} 
			${OpenCV_TS_DEBUG} 
			${OpenCV_VIDEO_DEBUG}
			${OpenCV_VIDEOSTAB_DEBUG}
			CACHE PATH "Libraries")
		set (OpenCV_FOUND TRUE CACHE PATH "Found opencv")

	else()
		message("[FindOpenCV] - ERROR - Did not find any OpenCV library")
		set (OpenCV_FOUND FALSE CACHE PATH "Found opencv")
	endif ()


else()
 message("[FindOpenCV] - ERROR Version < 2 is not supported by FindOpenCV.cmake")
 set (OpenCV_FOUND FALSE CACHE PATH "Found opencv")
endif()