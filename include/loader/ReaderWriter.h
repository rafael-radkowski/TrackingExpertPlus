#pragma once
/*
Class ReaderWriter
This is an abstract class for all ReaderWriter classes. 

Rafael Radkowski
Iowa State University
rafael@iastate.edu
December 2019
MIT License
------------------------------------------------------------------------------------------------------
Last edits:

Feb 22, 2020, RR
- check_type removes all spaces from the extracted file type.


*/
#include <iostream>
#include <string>
#include <vector>
#include <fstream>
#include <strstream>
#include <algorithm>
#include <locale>
#include <cctype>

// Eigen
#include <Eigen/Dense>

class ReaderWriter
{
public:


	/*!
	Load a point cloud object from a file
	@param file - The file
	@param loadedNormals = The output location of the loaded normals
	@return cloud = The output of the loaded point cloud object
	*/
        // Methods commented here because they are changed to static in a subclass and virtual static methods do not make any sense
        //virtual bool Read(const std::string file, std::vector<Eigen::Vector3f>& dst_points, std::vector<Eigen::Vector3f>& dst_normals, const bool normalize = false, const bool invert_z = false) = 0;


	/*
	Write the point cloud data to a file
	@param file - string containing path and name
	@param dst_points - vector of vector3f points containing x, y, z coordinates
	@param dst_normals - vector of vector3f normal vectors index-aligned to the points.
	@param scale_points - float value > 0.0 that scales all points and normal vectors. 
	*/
        // Methods commented here because they are changed to static in a subclass and virtual static methods do not make any sense
	//virtual bool Write(std::string file, std::vector<Eigen::Vector3f>& dst_points, std::vector<Eigen::Vector3f>& dst_normals, const float scale_points = 1.0f) = 0;


protected:

	//Helper functions to check whether the string character is a number
	static bool is_number(std::string s) {
		 return !s.empty() && s.find_first_not_of("0123456789.-e") == std::string::npos;
	}


	// check the expected filetype of the object
	static bool check_type(std::string path_and_file, std::string type) {
		
		size_t idx0 = path_and_file.find_last_of(".");
		std::string t = path_and_file.substr(idx0+1, path_and_file.size() - idx0 - 1);

		// remove space
		t.erase(remove_if(t.begin(), t.end(), ::isspace), t.end());

		// convert to lower case
		std::for_each(t.begin(), t.end(), [](char & c) {
			c = ::tolower(c);
		});

		std::for_each(type.begin(), type.end(), [](char & c) {
			c = ::tolower(c);
		});

		if (type.compare(t) == 0) {
			return true;
		}
		return false;
	}


	// split the lines into single elements
	static  std::vector<std::string> split(const std::string& s, char delimiter)
	{
	   std::vector<std::string> tokens;
	   std::string token;
	   std::istringstream tokenStream(s);
	   while (std::getline(tokenStream, token, delimiter))
	   {
		  tokens.push_back(token);
	   }
	   return tokens;
	}

};
