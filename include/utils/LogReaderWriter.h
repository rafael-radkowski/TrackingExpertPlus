#pragma once
/*
class LogReaderWriter

This class is a part of the performance analysis example. 
It writes parameter values related to the object detection and tracking performance
to a csv file. 

Note that this is an tool which is not necessary to enable the main functionality.  

Features:
- Create new log files
- Write tracking and object detection related information into a file


Rafael Radkowski
Iowa State University
rafael@iastate.edu
+1 (515) 294 7044
MIT License
------------------------------------------------------
Last Changes:

*/

// stl
#include <iostream>
#include <fstream>
#include <string>
#include <strstream>
#include <vector>
#include <list>
#include <numeric>
#include <filesystem>

#include <Eigen/Dense>

// GLM include files
#define GLM_FORCE_INLINE
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtx/transform.hpp>  // transformation
#include <glm/gtx/quaternion.hpp> // quaternions

// local
#include "TimeUtils.h"
#include "LogTypes.h"


using namespace std;


namespace texpert {

	class LogReaderWriter
	{
	public:

		/*
		Start a new log file
		@param path_and_file - string with the relative or absolute file.
		@return - true if file was successfully generated.
		*/
		static bool Create(string path_and_file);

		/*
		Append a dataset to a file.
		@param data - the dataset to write
		@return - true if file was successfully generated.
		*/
		static bool Write(LogData& data);

		/*
		Log all the metadata
		*/
		static bool WriteMetaData(LogMetaData& data);


			/*
		Log all the metadata
		*/
		static bool WriteResults(LogMetaData& data);


		/*
		Read and create the nodes for the Balanced Pose Tree
		@param path_and_file - string with the relative or absolute file.
		@param root - the root node of the tree.
		@param node_repository - reference to the node repository.
		*/
		static bool Read(string path_and_file, std::vector<LogData>& data);


		/*
		Write a string into a new file
		*/
		static bool FlashWrite(string path_and_file, string output);


		/*
		Write a string into a file only if this file does not exist.
		*/
		static bool FlashWriteHeader(string path_and_file, string header);

	private:

		static vector<string>  Split(string str, char delimiter);

		/*
		Write the header file
		*/
		static bool WriteHeader(string path_and_file);


		/*
		Check whether the file exits.
		*/
		static bool Exists(string path_and_file);


		/*
		Check whether the file exits and create if if not
		*/
		static bool ExistsAndCreate(string path_and_file);


	};

} //texpert 