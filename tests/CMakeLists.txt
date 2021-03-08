# TrackingExpert+ cmake file. 
# /tests-files. 
#
# source and project build file for TrackingExpert+
#
# http://blog.audio-tk.com/2015/09/01/sorting-source-files-and-projects-in-folders-with-cmake-and-visual-studioxcode/
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

set_property(GLOBAL PROPERTY USE_FOLDERS ON)

# Build the tracking test
option( TRAKINGX_BUILD_TEST_TRACKING "TrackingX Build Tracking Test" OFF)
option( TRAKINGX_BUILD_TEST_ICP "TrackingX Build ICP Test" OFF)
option( TRAKINGX_BUILD_TEST_CPF "TrackingX Build CPF Test" OFF)

add_subdirectory(test_detection)

if(ENABLE_STRUCTURE_CORE)
	add_subdirectory(test_camera)
endif()


if(${ENABLE_AZURE_KINECT} STREQUAL  ON )
	add_subdirectory(test_azure_camera)
endif()

# Build the trackaing test class
if(TRAKINGX_BUILD_TEST_TRACKING)
add_subdirectory(test_tracking)
endif()

# Build the ICP test class
if(TRAKINGX_BUILD_TEST_ICP)
add_subdirectory(test_knn)
add_subdirectory(test_icp)
add_subdirectory(test_icp_camera)
endif()

# Build the CPF test class
if(TRAKINGX_BUILD_TEST_CPF)
add_subdirectory(test_cpf)
endif()
#add_subdirectory(dev_detection)
