#  cmake file 
#  09_Multiple_Lights
#  
#  Multiple per-fragment-light example
#  This cmake file is part of CS/ME?CPRE 557
# 
#	Rafael Radkowski
#	Iowa State University
#	1620 Howe Hall
#	Ames, IA 50011
#	rafael@iastate.edu
#----------------------------------------------------------------
# Last edits
#


cmake_minimum_required(VERSION 2.6)

# policy for relative subfolders
cmake_policy(SET CMP0015 NEW)
cmake_policy(SET CMP0074 NEW)


# enable visual studio folders
set_property(GLOBAL PROPERTY USE_FOLDERS ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)



if(DEMO_TE2)
	# build the antenna and gearbox demos if the Azure Kinect (which they use unconditionally) is available
	if (ENABLE_AZURE_KINECT)
	  add_subdirectory(antenna_demo)
	  add_subdirectory(gear_demo)
	endif(ENABLE_AZURE_KINECT)
endif()

# new demo tool
if(DEMO_TE1)
	if (ENABLE_AZURE_KINECT)
		add_subdirectory(trackingexpertexample)
	endif(ENABLE_AZURE_KINECT)
endif()