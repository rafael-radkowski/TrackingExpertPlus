#pragma once

/*
class MatrixConv

This class converts between Eigen Matrix4f, Eigen Affine3f, and 
glm mat4 matrices.

This singleton class assists in the conversion of matrices between the
Eigen and glm libraries, namely 4-by-4 pose matrices.  The naming convention
of these functions are as follows: (type to convert from)2(type to convert to). 

Most functions are meant to copy the contents of entire matrices into the other
type, while some other functions (such as Affine3f2Vec3Trans) copy specifically
the translation vector or rotation matrix from the first type into the second.

William Blanchard
Iowa State University
Feb 2021
wsb@iastate.edu
MIT License
---------------------------------------------------------------
Last edited:

March 18, 2021, WB
- Added documentation

Aug 27, 2021, RR
- Added Rt2Affine3f
*/

//std
#include <iostream>

//sdk
#include <Eigen/Dense>
#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>

class MatrixConv
{
	static MatrixConv* singleton;

private:
	MatrixConv();

public:

	/*
		Gets an instance of this MatrixConv
	*/
	static MatrixConv* getInstance();

	/*
		Prints a matrix organized in column major

		@param mat - a pointer to the matrix data
		@param row - the number of rows in the matrix
		@param col - the number of columns in the matrix
	*/
	void printColMjr(float* mat, int row, int col);

	/*
		Converts an Eigen::Matrix4f into a glm::mat4
		@param eigen_in - the input Eigen matrix
		@param glm_out - the output glm matrix
	*/
	void Matrix4f2Mat4(Eigen::Matrix4f& eigen_in, glm::mat4& glm_out);

	/*
		Converts an Eigen::Affine3f into a glm::mat4
		@param eigen_in - the input Eigen matrix
		@param glm_out - the output glm matrix
	*/
	void Affine3f2Mat4(Eigen::Affine3f& eigen_in, glm::mat4& glm_out);

	/*
		Converts the translation element of an Eigen::Affine3f into a glm::mat4
		@param eigen_in - the input Eigen matrix
		@param glm_out - the output glm matrix
	*/
	void Affine3f2Mat4Trans(Eigen::Affine3f& eigen_in, glm::mat4& glm_out);

	/*
		Converts the rotation element of an Eigen::Affine3f into a glm::mat4
		@param eigen_in - the input Eigen matrix
		@param glm_out - the output glm matrix
	*/
	void Affine3f2Mat4Rot(Eigen::Affine3f& eigen_in, glm::mat4& glm_out);

	/*
		Converts the translation element of an Eigen::Affine3f into a glm::vec3
		@param eigen_in - the input Eigen matrix
		@param glm_out - the output glm vector
	*/
	void Affine3f2Vec3Trans(Eigen::Affine3f& eigen_in, glm::vec3& glm_out);

	/*
		Converts the translation element of an Eigen::Matrix4f into a glm::vec3
		@param eigen_in - the input Eigen matrix
		@param glm_out - the output glm vector
	*/
	void Matrix4f2Vec3Trans(Eigen::Matrix4f& eigen_in, glm::vec3& glm_out);

	/*
		Converts the rotation element of an Eigen::Matrix4f into a glm::mat4
		@param eigen_in - the input Eigen matrix
		@param glm_out - the output glm matrix
	*/
	void Matrix4f2Mat4Rot(Eigen::Matrix4f& eigen_in, glm::mat4& glm_out);

	/*
		Converts an Eigen::Matrix4f to an Eigen::Affine3f
		@param in - the input Eigen matrix
		@param out - the output Eigen affine matrix
	*/
	void Matrix4f2Affine3f(Eigen::Matrix4f& in, Eigen::Affine3f& out);

	/*
		Converts an Eigen::Affinef to an Eigen::Matrix4f
		@param in - the input Eigen affine matrix
		@param out - the output Eigen matrix
	*/
	void Affine3f2Matrix4f(Eigen::Affine3f& in, Eigen::Matrix4f& out);

	/*
		Converts a glm::mat4 to an Eigen::Matrix4f
		@param glm_in - the input glm matrix
		@param eigen_out - the output eigen matrix
	*/
	void Mat42Matrix4f(glm::mat4& glm_in, Eigen::Matrix4f& eigen_out);

	/*
		Converts a glm::mat4 to an Eigen::Affine3f
		@param glm_in - the input glm matrix
		@param eigen_out - the output eigen matrix
	*/
	void Mat42Affine3f(glm::mat4& glm_in, Eigen::Affine3f& eigen_out);

	/*
		Converts the translation element of a glm::mat4 to an Eigen::Matrix4f
		@param glm_in - the input glm matrix
		@param eigen_out - the output eigen matrix
	*/
	void Mat42Matrix4fTrans(glm::mat4& glm_in, Eigen::Matrix4f& eigen_out);

	/*
		Converts the rotation element of a glm::mat4 to an Eigen::Matrix4f
		@param glm_in - the input glm matrix
		@param eigen_out - the output eigen matrix
	*/
	void Mat42Matrix4fRot(glm::mat4& glm_in, Eigen::Matrix4f& eigen_out);

	/*
		Converts the translation element of a glm::mat4 to an Eigen::Affine3f
		@param glm_in - the input glm matrix
		@param eigen_out - the output eigen matrix
	*/
	void Mat42Affine3fTrans(glm::mat4& glm_in, Eigen::Affine3f& eigen_out);

	/*
		Converts the rotation element of a glm::mat4 to an Eigen::Affine3f
		@param glm_in - the input glm matrix
		@param eigen_out - the output eigen matrix
	*/
	void Mat42Affine3fRot(glm::mat4& glm_in, Eigen::Affine3f& eigen_out);


	/*
		Convert a set of Euler angles and a translation into a 4x4 matrix of type Affine3f
		@param R - a vector 3 with three Euler angles (Rx, Ry, Rz)
		@param t - a vector 3 with three translations x, y, z.
		@return a affine3f matrix with the pose. 
	*/
	Eigen::Affine3f Rt2Affine3f(Eigen::Vector3f R, Eigen::Vector3f t);

private:


	/*
	Create a rotation matrix from Euler angles

	*/
	Eigen::Affine3f createRotationMatrix(float ax, float ay, float az);


	
};
