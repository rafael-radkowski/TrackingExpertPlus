/*
class FileUtils

Rafael Radkowski
Iowa State University
rafael@iastate.edu
MIT License
---------------------------------------------------------------
*/

#ifndef __FILE_UTILS__
#define __FILE_UTILS__


#include <vector>
#include <iostream>
#include <string>
#include <fstream>
#ifdef _WIN32
	#include <conio.h>
    #include  <filesystem>
#else
    #include <unistd.h>
#endif

using namespace std;

class FileUtils
{
public:

    /*
    Check if a file exits. 
    @param path_and_file - string containing the path and file, relative or absoulte. 
    @return true, if the file exits. 
    */
    static bool Exists(string path_and_file);

};



#endif

