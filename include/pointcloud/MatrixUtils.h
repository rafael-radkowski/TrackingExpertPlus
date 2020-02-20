#pragma once

#pragma once
/*
class MatrixUtils

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

	class MatrixUtils
	{
	public:



		/*
		Converts an Eigen::AAffine3f transformation to a glm4 matrix
		Matrices are in column major order.
		@param matrix - affine matrix of type Affine3f (4x4).
		@return glm mat4 matrix. 
		*/
		static glm::mat4 Affine3f2Mat4(Eigen::Affine3f& matrix);


		/*
		Print an affine3 Eigen matrix.
		@param matrix - the matrix in column-major order
		*/
		static void PrintAffine3f(Eigen::Affine3f& matrix);


		/*
		Print an a glm::mat4  matrix.
		@param matrix - the matrix in column-major order
		*/
		static void PrintGlm4(glm::mat4& matrix);


		static void PrintMatrix4f(Eigen::Matrix4f& matrix);

	private:


	};



};