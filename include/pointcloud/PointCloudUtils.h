#pragma once
/*
@class PointCloudUtils
@brief Class with static utility functions to compute point cloud related data.

Features:
- Calculates the centroid of a point cloud. 

Rafael Radkowski
Iowa State University 
Feb 2018
rafael@iastate.edu
MIT License
---------------------------------------------------------------
Last edits:

June 9, 2020, RR
- Added a function to calculate the point cloud centroid using a list of points as input. 
*/

#include <iostream>
#include <string>
#include <vector>


// GLM include files
#define GLM_FORCE_INLINE
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtx/transform.hpp>  // transformation
#include <glm/gtx/quaternion.hpp> // quaternions

// Eigen
#include <Eigen/Dense>

// local
#include "Types.h"

using namespace std;

namespace texpert
{

	class PointCloudUtils
	{
	public:

		/*
		Calculate the centroid of the point cloud.
		@param pc_src -  pointer to the source point cloud of type PointCloud
		@return  the vector with the centroid as x, y, z in local coordinates.
		*/
		static Eigen::Vector3f  CalcCentroid(PointCloud* pc_src);

		/*
		Calculate the centroid of the point cloud.
		@param pc_src -  pointer to the source point cloud as list of type std::vector<Eigen::Vector3f>.
		@return  the vector with the centroid as x, y, z in local coordinates.
		*/
		static Eigen::Vector3f  CalcCentroid3f(std::vector<Eigen::Vector3f>& pc_src);


	private:


	};



};