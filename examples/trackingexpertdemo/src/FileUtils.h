/*
Helper class to read directory names and / or file names
at a given path.

The class takes a path and returns the names of all
folders / files as std::vector<string>

Rafael Radkowski
Iowa State University
June 2015
rafael@iastate.edu
----------------------------------------------------
Latest edits
*/
#pragma once

#ifndef __FILE_UTILS__
#define __FILE_UTILS__


// stl
#include <string>
#include <iostream>
#include <fstream>
#include <vector>
#include <map>
#include <string>
#ifdef _WIN32
#include <windows.h>
#include <tchar.h>
#include <stdio.h>
#include <conio.h>
#endif 

#if (_MSC_VER >= 1920 && _MSVC_LANG  == 201703L) || (__GNUC__ >= 8)
#include <filesystem>
#else
#define _SILENCE_EXPERIMENTAL_FILESYSTEM_DEPRECATION_WARNING
#include <experimental/filesystem>
#endif


//#else
//#include <dirent.h>
//#include <unistd.h>
//#endif


using namespace std;

class FileUtils
{
public:


    /*!
    Check if a file exits. 
    @param path_and_file - string containing the path and file, relative or absoulte. 
    @return true, if the file exits. 
    */
    static bool Exists(string path_and_file);

	/*
	Create a director if it does not exist.
	@param path - the path as string
	*/
	static bool CreateDirectories(string path);


	/*
	Return the names of all files at the given path
	@param path  - string containing the path
	@return std::vector with the file names. 
	*/
	static vector<string> GetFileList(string path);


	/*
	Remove a file
	@param file  - string containing the path to the file
	*/
	static bool Remove(string file);

	/*
	Create one directory
	@param path  - string containing the path
	*/
	static bool CreateDirectory(string path);

};



#endif

