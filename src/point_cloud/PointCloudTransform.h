#pragma once
/*


Rafael Radkowski
Iowa State University 
Feb 2018
rafael@iastate.edu
MIT License
*/

#include <iostream>
#include <string>
#include <cassert>


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
#include "PointCloudUtils.h"

using namespace std;

namespace isu_ar
{

	class PointCloudTransform
	{
	public:

		/*
		Move the point cloud points along a translation vector
		@param pc_src -  pointer to the source point cloud of type PointCloud
		@param translation - vec 3 with the translation in x, y, z/
		@return  true - if successful. 
		*/
		static bool Translate(PointCloud* pc_src, Eigen::Vector3f translation);


		/*
		Rotate the point cloud points around their origin
		@param pc_src -  pointer to the source point cloud of type PointCloud
		@param rotation - vec 3 with the Euler angles for a rotation arond x, y, z.
		@return  true - if successful. 
		*/
		static bool Rotate(PointCloud* pc_src, Eigen::Vector3f  euler_angles);



		/*
		Transforms the point cloud points 
		@param pc_src -  pointer to the source point cloud of type PointCloud
		@param translation - vec 3 with the translation in x, y, z
		@param rotation - vec 3 with the Euler angles for a rotation arond x, y, z.
		@return  true - if successful. 
		*/
		static bool Transform(PointCloud* pc_src, Eigen::Vector3f translation, Eigen::Vector3f  euler_angles);

	private:


	};



};