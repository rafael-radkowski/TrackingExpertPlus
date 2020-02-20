/*
FileUtils.h/.cpp

This file contains some helpfull tools that helps with file handling and 
file searching. 

Features:
- Search for a file in adjacent folders.
- Determine whether or not a file exists. 


This file is part of CS/CPRE/ME 557 Computer Graphics

Rafael Radkowski
Iowa State University
rafael@iastate.edu
+1 (515) 294 7044
MIT License

------------------------------------------------------------------
Last edited:

Oct 22, 2019, RR
- Bugfix: fixed a bug that caused a compiler errror on Mac OS X. 

*/
#pragma once

// stl
#include <iostream>
#include <string>

// local
#include "MSVerCheck.h"
//#include "FileUtilsX.h"

using namespace std;

namespace texpert{


class FileUtils {

public:

	/*!
	Check whether or not a file exists at the given path.
	@param path_and_file - relative or absolute location and name of the file.
	@return true, if the file exists, false otherwise. 
	*/
	static bool Exists(string path_and_file);


	/*!
	Search for a file in adjacent folders
	@param path_and_file - relative or absolute location and name of the file.
	@param new_path_and_file - relative or absolute location and name of the located file or "" if no file exists. 
	@return true, if the file exists, false otherwise. 
	*/
	static bool Search(string path_and_file, string& new_path_and_file);

};



}//namespace cs557