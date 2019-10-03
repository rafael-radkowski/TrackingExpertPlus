/*
class LoaderObj

Rafael Radkowski
Iowa State University
rafael@iastate.edu
MIT License
---------------------------------------------------------------
*/

#ifndef __LOADER_OBJ__
#define __LOADER_OBJ__


#include <vector>
#include <iostream>
#include <string>
#include <fstream>
#ifdef __WIN32
	#include <conio.h>
#endif

// eigen
#include <Eigen/Dense>

// local
#include "Types.h"

using namespace std;


class LoaderObj{
private:

	//Helper functions
	static bool is_number(std::string s);

public:

	/*!
	Load a point cloud .obj object from file
	@param file = The file
	@param dst_points -	the location for the points to load
	@param dst_normals - the location for the normals to laod
	@param invert_z - inverts the z-axis in case the camera ref data is left-hand-coorindate framed.
	@param verbose - adds additional output messages in addition to error messages
	@return true - if points successfully loaded
	*/
	static bool Read(string file, vector<Eigen::Vector3f>* dst_points, vector<Eigen::Vector3f>* dst_normals, bool invert_z = true, bool verbose = false);
};

#endif