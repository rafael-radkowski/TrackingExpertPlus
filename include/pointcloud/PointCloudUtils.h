#pragma once
/*
class PointCloudUtils

Rafael Radkowski
Iowa State University 
Feb 2018
rafael@iastate.edu
MIT License
---------------------------------------------------------------
*/

#include <iostream>
#include <string>



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
		Move the point cloud points along a translation vector
		@param pc_src -  pointer to the source point cloud of type PointCloud
		@param pc_dst - pointer to the destination point cloud of type PointCloud
		@param translation - vec 3 with the translation in x, y, z/
		@return  true - if successful. 
		*/
		static Eigen::Vector3f  CalcCentroid(PointCloud* pc_src);





	private:


	};



};