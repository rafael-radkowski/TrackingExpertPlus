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

June 11, 2020, WB
- Initial commit
*/

#include <Eigen/Dense>
#include <glm/glm.hpp>

class MatrixConv
{
	MatrixConv* singleton;

private:
	MatrixConv();

public:

	/*
		Gets an instance of this MatrixConv
	*/
	MatrixConv* getInstance();

	/*
		Converts an Eigen::Matrix4f into a glm::mat4
		@param eigen_in - the input Eigen matrix
		@param glm_out - the output glm matrix
	*/
	void EigenMat4ToGlmMat4(Eigen::Matrix4f& eigen_in, glm::mat4& glm_out);

	/*
		Converts an Eigen::Affine3f into a glm::mat4
		@param eigen_in - the input Eigen matrix
		@param glm_out - the output glm matrix
	*/
	void EigenAff3ToGlmMat4(Eigen::Affine3f& eigen_in, glm::mat4& glm_out);

	/*
		Converts the translation element of an Eigen::Affine3f into a glm::mat4
		@param eigen_in - the input Eigen matrix
		@param glm_out - the output glm matrix
	*/
	void EigenAff3ToGlmMat4Trans(Eigen::Affine3f& eigen_in, glm::mat4& glm_out);

	/*
		Converts the rotation element of an Eigen::Affine3f into a glm::mat4
		@param eigen_in - the input Eigen matrix
		@param glm_out - the output glm matrix
	*/
	void EigenAff3ToGlmMat4Rot(Eigen::Affine3f& eigen_in, glm::mat4& glm_out);

	/*
		Converts the translation element of an Eigen::Affine3f into a glm::vec3
		@param eigen_in - the input Eigen matrix
		@param glm_out - the output glm vector
	*/
	void EigenAff3ToGlmVec3Trans(Eigen::Affine3f& eigen_in, glm::vec3& glm_out);

	/*
		Converts the translation element of an Eigen::Matrix4f into a glm::vec3
		@param eigen_in - the input Eigen matrix
		@param glm_out - the output glm vector
	*/
	void EigenMat4ToGlmVec3Trans(Eigen::Matrix4f& eigen_in, glm::vec3& glm_out);

	/*
		Converts the rotation element of an Eigen::Matrix4f into a glm::mat4
		@param eigen_in - the input Eigen matrix
		@param glm_out - the output glm matrix
	*/
	void EigenMat4ToGlmMat4Rot(Eigen::Matrix4f& eigen_in, glm::mat4& glm_out);

	/*
		Converts an Eigen::Matrix4f to an Eigen::Affine3f
		@param in - the input Eigen matrix
		@param out - the output Eigen affine matrix
	*/
	void EigenMat4ToEigenAff3(Eigen::Matrix4f& in, Eigen::Affine3f& out);

	/*
		Converts an Eigen::Affinef to an Eigen::Matrix4f
		@param in - the input Eigen affine matrix
		@param out - the output Eigen matrix
	*/
	void EigenAff3ToEigenMat4(Eigen::Affine3f& in, Eigen::Matrix4f& out);

	/*
		Converts a glm::mat4 to an Eigen::Matrix4f
		@param glm_in - the input glm matrix
		@param eigen_out - the output eigen matrix
	*/
	void GlmMat4ToEigenMat4(glm::mat4& glm_in, Eigen::Matrix4f& eigen_out);

	/*
		Converts a glm::mat4 to an Eigen::Affine3f
		@param glm_in - the input glm matrix
		@param eigen_out - the output eigen matrix
	*/
	void GlmMat4ToEigenAff3(glm::mat4& glm_in, Eigen::Affine3f& eigen_out);

	/*
		Converts the translation element of a glm::mat4 to an Eigen::Matrix4f
		@param glm_in - the input glm matrix
		@param eigen_out - the output eigen matrix
	*/
	void GlmMat4ToEigenMat4Trans(glm::mat4& glm_in, Eigen::Matrix4f& eigen_out);

	/*
		Converts the rotation element of a glm::mat4 to an Eigen::Matrix4f
		@param glm_in - the input glm matrix
		@param eigen_out - the output eigen matrix
	*/
	void GlmMat4ToEigenMat4Rot(glm::mat4& glm_in, Eigen::Matrix4f& eigen_out);

	/*
		Converts the translation element of a glm::mat4 to an Eigen::Affine3f
		@param glm_in - the input glm matrix
		@param eigen_out - the output eigen matrix
	*/
	void GlmMat4ToEigenAff3Trans(glm::mat4& glm_in, Eigen::Affine3f& eigen_out);

	/*
		Converts the rotation element of a glm::mat4 to an Eigen::Affine3f
		@param glm_in - the input glm matrix
		@param eigen_out - the output eigen matrix
	*/
	void GlmMat4ToEigenAff3Rot(glm::mat4& glm_in, Eigen::Affine3f& eigen_out);
};