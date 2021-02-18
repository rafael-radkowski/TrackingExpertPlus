#pragma once

/*
class MatrixConv

William Blanchard
Iowa State University
Feb 2021
wsb@iastate.edu
MIT License
---------------------------------------------------------------
Last edited:

Feb 17, 2021, WB
- Renamed functions to match convention
- Date correction
*/

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
};