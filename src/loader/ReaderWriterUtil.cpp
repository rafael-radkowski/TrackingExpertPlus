#include "ReaderWriterUtil.h"



/*!
	Load a point cloud object from a file
	@param file - The file
	@param loadedNormals = The output location of the loaded normals
	@return cloud = The output of the loaded point cloud object
	*/
	//virtual 
	bool ReaderWriterUtil::Read(const std::string file, std::vector<Eigen::Vector3f>& dst_points, std::vector<Eigen::Vector3f>& dst_normals, const bool normalize, const bool invert_z)
	{
		
		bool obj_type = check_type(file, "obj");
		if (obj_type) {
			return ReaderWriterOBJ::Read(file, dst_points, dst_normals, normalize, invert_z);
		}


		bool ply_type = check_type(file, "ply");
		if (ply_type) {
			return ReaderWriterPLY::Read(file, dst_points, dst_normals, normalize, invert_z);
		}

		std::cout << "[ERROR] - File type of file " << file << " is not supported." << std::endl;

		return false;
	}


	/*
	Write the point cloud data to a file
	@param file - string containing path and name
	@param dst_points - vector of vector3f points containing x, y, z coordinates
	@param dst_normals - vector of vector3f normal vectors index-aligned to the points.
	@param scale_points - float value > 0.0 that scales all points and normal vectors. 
	*/
	//virtual 
	bool ReaderWriterUtil::Write(std::string file, std::vector<Eigen::Vector3f>& dst_points, std::vector<Eigen::Vector3f>& dst_normals, const float scale_points)
	{
		
		bool obj_type = check_type(file, "obj");
		if (obj_type) {
			return ReaderWriterOBJ::Write(file, dst_points, dst_normals, scale_points);
		}


		bool ply_type = check_type(file, "ply");
		if (ply_type) {
			return ReaderWriterPLY::Write(file, dst_points, dst_normals, scale_points);
		}

		std::cout << "[ERROR] - File type of file " << file << " is not supported." << std::endl;

		return false;

		return true;
	}