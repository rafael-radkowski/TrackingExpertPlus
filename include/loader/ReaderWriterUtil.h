#pragma once

// stl
#include <iostream>
#include <string>
#include <vector>
#if (_MSC_VER >= 1920 && _MSVC_LANG  == 201703L) || (__GNUC__ >= 8) 
#include <filesystem>
#else 
#define _USE_EXP
#include <experimental/filesystem>
#endif


// local 
#include "ReaderWriter.h"
#include "ReaderWriterOBJ.h"
#include "ReaderWriterPLY.h"



class ReaderWriterUtil : public ReaderWriter
{

public:
	
	/*!
	Load a point cloud object from a file
	@param file - The file
	@param loadedNormals = The output location of the loaded normals
	@return cloud = The output of the loaded point cloud object
	*/
	static bool Read(const std::string file, std::vector<Eigen::Vector3f>& dst_points, std::vector<Eigen::Vector3f>& dst_normals, const bool normalize = false, const bool invert_z = false);


	/*
	Write the point cloud data to a file
	@param file - string containing path and name
	@param dst_points - vector of vector3f points containing x, y, z coordinates
	@param dst_normals - vector of vector3f normal vectors index-aligned to the points.
	@param scale_points - float value > 0.0 that scales all points and normal vectors. 
	*/
	static bool Write(std::string file, std::vector<Eigen::Vector3f>& dst_points, std::vector<Eigen::Vector3f>& dst_normals, const float scale_points = 1.0f);

};
