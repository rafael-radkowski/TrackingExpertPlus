# @file  FindAzureKinect.cmake
# @brief Cmake script to find the Azure Kinect libraries, https://docs.microsoft.com/en-us/azure/kinect-dk/sensor-sdk-download
#
# The module finds the Azure Kinect library k4a.lib and k4arecord.lib for Windows.
# The script first looks into the Azure Kinect default installation folder C:\Program Files\Azure Kinect SDK v1.4.0
# It fetches the environment variable AZUREKINECT_DIR, if it cannot find the file at this location. 
#
# Use 
#	find_package(AzureKinect)
# in a cmake file to find it. 
# Note that this file was developed and tested with Azure Kinect SDK v1.4.0
#
# 1. Setup
# In the best case, the script will find the Azure Kinect dependencies. 
# If not, set the environment variable AZUREKINECT_DIR pointing to the main folder containing the Azure Kinect SDK folder, e.g.,  C:\Program Files\Azure Kinect SDK v1.4.0
# The folder must contain the subfolders sdk and tools
#
# 2. Variables
# The following variables are produced and usable in a CMakeList.txt file after cmake is done: 
#
#  AZUREKINECT_FOUND - system has eigen lib with correct version
#  AZUREKINECT_INCLUDE_DIR - the structure lib include directory
#  AZUREKINECT_LIBRARY - macro pointing to the azure kinect libraries k4a.lib and k4arecord.lib
#
#
# #  # tested with:
# - Azure Kinect SDK v1.4.0:  MSVC 15,2017
#
#
# Rafael Radkowski
# Iowa State University
# Virtual Reality Applications Center
# rafael@iastate.eduy
# July 17, 2020
# rafael@iastate.edu
#
# MIT License
#---------------------------------------------------------------------
#
# Last edits:
#
# 




##----------------------------------------------------------
## Init the search
set(AZUREKINECT_FOUND FALSE)

# To hide internal variables
macro ( mark_as_internal _var )
  set ( ${_var} ${${_var}} CACHE INTERNAL "hide this!" FORCE )
endmacro( mark_as_internal _var )


# default search dirs
set( _azurekinect_DEFAULT_SEARCH_DIRS 
  "C:/Program Files/Azure Kinect SDK v1.4.1"
  "C:/Program Files/Azure Kinect SDK v1.4.0"
  "D:/Program Files/Azure Kinect SDK v1.4.0"
  "C:/Program Files/Azure Kinect SDK v1.5.0"
  "D:/Program Files/Azure Kinect SDK v1.5.0"
  )
  
  
# 1. Search in the default search dirs
if(${AZUREKINECT_FOUND} MATCHES FALSE)
	find_path(AZUREKINECT_DIR 
		NAMES /sdk/include/k4a/k4a.h 
		HINTS ${_azurekinect_DEFAULT_SEARCH_DIRS}
		PATH_SUFFIXES 
	)
	
	if(AZUREKINECT_DIR)
		set(AZUREKINECT_FOUND TRUE)
		message(STATUS "[FindAzureKinect] - Found kinect dir (1) " ${AZUREKINECT_DIR})
	endif()
	
endif() # if(NOT AZUREKINECT_FOUND)


# 2. Search using the environment variable
if(${AZUREKINECT_FOUND} MATCHES FALSE)

	if (DEFINED ENV{AZUREKINECT_DIR})
		set (_azure_kinect_DIR "$ENV{AZUREKINECT_DIR}" CACHE PATH "Azure Kinect Library Installation Path." FORCE)
	elseif (DEFINED ENV{AzureKinect_DIR})
		set (_azure_kinect_DIR "$ENV{AzureKinect_DIR}" CACHE PATH "Azure Kinect Library Installation Path." FORCE)
	elseif (DEFINED ENV{AZUREKINECT_dir})
		set (_azure_kinect_DIR "$ENV{AZUREKINECT_dir}" CACHE PATH "Azure Kinect Library Installation Path." FORCE)
	elseif (DEFINED ENV{azurekinect_dir})
		set (_azure_kinect_DIR "$ENV{azurekinect_dir}" CACHE PATH "Azure Kinect Library Installation Path." FORCE)
	elseif (DEFINED ENV{AzureKinect_Dir})
		set (_azure_kinect_DIR "$ENV{AzureKinect_Dir}" CACHE PATH "Azure Kinect Library Installation Path." FORCE)
	elseif (DEFINED ENV{AZURE_KINECT_DIR})
		set (_azure_kinect_DIR "$ENV{AZURE_KINECT_DIR}" CACHE PATH "Azure Kinect Library Installation Path." FORCE)		
	endif()
	
	mark_as_internal(_azure_kinect_DIR)
	#message(STATUS "AK env var: "  ${_azure_kinect_DIR})
	
	if(_azure_kinect_DIR  )
		set(AZUREKINECT_DIR_TEMP ${_azure_kinect_DIR})# CACHE PATH "Azure Kinect Library Installation Path." FORCE)
	
		#message(STATUS "AK dir verification 1: "  ${AZUREKINECT_DIR_TEMP})
		
		# Verify the existance of the folder
		find_path(AZUREKINECT_DIR 
			NAMES /sdk/include/k4a/k4a.h 
			HINTS ${AZUREKINECT_DIR_TEMP}
			PATH_SUFFIXES 
		)
		
		#message(STATUS "AK dir verification 2: "  ${AZUREKINECT_DIR})
		
		if(AZUREKINECT_DIR )
			set(AZUREKINECT_FOUND TRUE)
			message(STATUS "[FindAzureKinect] - Found Azure Kinect dir (2) " ${AZUREKINECT_DIR})
		else()
			set(AZUREKINECT_FOUND FALSE)
			message(WARNING "[FindAzureKinect] - Did not find Azure Kinect dir (4) " ${AZUREKINECT_DIR})
		endif()
	else()
		set(AZUREKINECT_FOUND FALSE)
		message(WARNING "[FindAzureKinect] - Did not find Azure Kinect dir (3) " ${AZUREKINECT_DIR})
	endif()

endif() # if(NOT AZUREKINECT_FOUND)


# 3. Verify if the variable was found
if(${AZUREKINECT_FOUND} MATCHES FALSE)
	message(FATAL_ERROR "[FindAzureKinect] - Did not find Azure kinect dir - fatal error.")
endif()


##----------------------------------------------------------
## Find the include dir

if(NOT AZUREKINECT_INCLUDE_DIR AND AZUREKINECT_DIR)
find_path(AZUREKINECT_INCLUDE_DIR 
	NAMES k4a/k4a.h 
	HINTS
	${AZUREKINECT_DIR}/sdk/include/
	PATH_SUFFIXES k4a
)
endif() # if(NOT STRUCTURE_INCLUDE_DIR)

#message (STATUS "Include dir: " ${AZUREKINECT_INCLUDE_DIR})



##----------------------------------------------------------
## Find the library path


# 3. Find the library path
if(NOT AZUREKINECT_LIBRARY_k4a AND AZUREKINECT_DIR)

	find_file(AZUREKINECT_LIBRARY_k4a
		NAMES k4a.lib
		PATHS 
		${AZUREKINECT_DIR}/sdk/windows-desktop/amd64/release/lib
	)
	mark_as_internal(AZUREKINECT_LIBRARY_k4a)
	
endif()

if(NOT AZUREKINECT_LIBRARY_k4arecord AND AZUREKINECT_DIR)

	find_file(AZUREKINECT_LIBRARY_k4arecord
		NAMES k4arecord.lib
		PATHS 
		${AZUREKINECT_DIR}/sdk/windows-desktop/amd64/release/lib
	)
	mark_as_internal(AZUREKINECT_LIBRARY_k4arecord)
	
endif()

set(AZUREKINECT_LIBRARY 
	${AZUREKINECT_LIBRARY_k4a}  
	${AZUREKINECT_LIBRARY_k4arecord}
	CACHE PATH "Azure Kinect Library Installation Path." FORCE)

	
if(AZUREKINECT_LIBRARY)
	set(AZUREKINECT_FOUND TRUE  CACHE STRING "Found Azure Kinect")
	message(STATUS "[FindAzureKinect] - Found Azure Kinect sdk at ${AZUREKINECT_DIR}" )
endif()
