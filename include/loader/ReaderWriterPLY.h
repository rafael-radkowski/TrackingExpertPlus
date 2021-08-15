#pragma once
/*
Class ReaderWriterPLY
This class reads and writes model in the PLY file format.
The supported format is 
	x y z nx ny nz  (points, normals)
Also, the class only reads the vertices as points. 

NOTE THAT THIS CLASS IS NOT A COMPLETE PLY LOADER / WRITER. IT JUST SERVES A LIMITED PURPOSE.

Rafael Radkowski
Iowa State University
rafael@iastate.edu
December 2019
MIT License
------------------------------------------------------------------------------------------------------
Last edits:

Dec 15, 2019, RR
- Added a function to scale the outgoing object. 
- Changed the output to fixed length values
Feb 7, 2020, RR
- Added conio.h
- Added FileUtils.h

*/
// stl
#include <iostream>
#include <string>
#include <vector>
#include <fstream>
#include <strstream>
#ifdef _WIN32
#include <conio.h>
#endif
// Eigen
#include <Eigen/Dense>

// local
#include "ReaderWriter.h"
#include "FileUtilsX.h"

class ReaderWriterPLY : public ReaderWriter
{
public:

		/*!
	Load a point cloud object from a file
	@param file - The file
	@param loadedNormals = The output location of the loaded normals
	@return cloud = The output of the loaded point cloud object
	*/
	//virtual 
	static bool Read(const std::string file, std::vector<Eigen::Vector3f>& dst_points, std::vector<Eigen::Vector3f>& dst_normals, const bool normalize = false, const bool invert_z = false);


	/*
	Write the point cloud data to a file
	@param file - string containing path and name
	@param dst_points - vector of vector3f points containing x, y, z coordinates
	@param dst_normals - vector of vector3f normal vectors index-aligned to the points.
	@param scale_points - float value > 0.0 that scales all points and normal vectors. 
	*/
	//virtual 
	static bool Write(std::string file, std::vector<Eigen::Vector3f>& src_points, std::vector<Eigen::Vector3f>& src_normals, const float scale_points = 1.0f);


private:


	static void ErrorMsg(std::string msg);


};
