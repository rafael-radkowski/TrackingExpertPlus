#include "MatrixConv.h"

MatrixConv* MatrixConv::singleton = 0;

MatrixConv::MatrixConv()
{}

MatrixConv* MatrixConv::getInstance()
{
	if (singleton == NULL) 
	{
		singleton = new MatrixConv();
	}
	else
	{
		return singleton;
	}
}

void MatrixConv::Matrix4f2Mat4(Eigen::Matrix4f& eigen_in, glm::mat4& glm_out)
{
	memcpy(glm::value_ptr(glm_out), eigen_in.data(), 16 * sizeof(float));
}

void MatrixConv::Affine3f2Mat4(Eigen::Affine3f& eigen_in, glm::mat4& glm_out)
{
	memcpy(glm::value_ptr(glm_out), eigen_in.data(), 16 * sizeof(float));
}

void MatrixConv::Affine3f2Mat4Trans(Eigen::Affine3f& eigen_in, glm::mat4& glm_out)
{
	glm_out = glm::mat4(1.0f);
	glm_out[3][0] = eigen_in(0, 3);
	glm_out[3][1] = eigen_in(1, 3);
	glm_out[3][2] = eigen_in(2, 3);
}

void MatrixConv::Affine3f2Mat4Rot(Eigen::Affine3f& eigen_in, glm::mat4& glm_out)
{
	glm_out = glm::mat4(1.0f);
	for (int i = 0; i < 9; i++)
	{
		int row = i / 3;
		int col = i % 3;
		glm_out[col][row] = eigen_in(row, col);
	}
}

void MatrixConv::Affine3f2Vec3Trans(Eigen::Affine3f& eigen_in, glm::vec3& glm_out)
{
	glm_out = glm::vec3(eigen_in(0, 3), eigen_in(1, 3), eigen_in(2, 3));
}

void MatrixConv::Matrix4f2Vec3Trans(Eigen::Matrix4f& eigen_in, glm::vec3& glm_out)
{
	glm_out = glm::vec3(eigen_in(0, 3), eigen_in(1, 3), eigen_in(2, 3));
}

void MatrixConv::Matrix4f2Mat4Rot(Eigen::Matrix4f& eigen_in, glm::mat4& glm_out)
{
	glm_out = glm::mat4(1.0f);
	for (int i = 0; i < 9; i++)
	{
		int row = i / 3;
		int col = i % 3;
		glm_out[col][row] = eigen_in(row, col);
	}
}

void MatrixConv::Matrix4f2Affine3f(Eigen::Matrix4f& in, Eigen::Affine3f& out)
{
	out.matrix() = in.matrix();
}

void MatrixConv::Affine3f2Matrix4f(Eigen::Affine3f& in, Eigen::Matrix4f& out)
{
	out.matrix() = in.matrix();
}


void MatrixConv::Mat42Matrix4f(glm::mat4& glm_in, Eigen::Matrix4f& eigen_out)
{
	memcpy(eigen_out.data(), glm::value_ptr(glm_in), 16 * sizeof(float));
}

void MatrixConv::Mat42Affine3f(glm::mat4& glm_in, Eigen::Affine3f& eigen_out)
{
	memcpy(eigen_out.data(), glm::value_ptr(glm_in), 16 * sizeof(float));
}

void MatrixConv::Mat42Matrix4fTrans(glm::mat4& glm_in, Eigen::Matrix4f& eigen_out)
{
	eigen_out = Eigen::Matrix4f::Identity();

	eigen_out(0, 3) = glm_in[3][0];
	eigen_out(1, 3) = glm_in[3][1];
	eigen_out(2, 3) = glm_in[3][2];
}

/*
	Can be simplified to O(n) time by using modulo
*/
void MatrixConv::Mat42Matrix4fRot(glm::mat4& glm_in, Eigen::Matrix4f& eigen_out)
{
	eigen_out = Eigen::Matrix4f::Identity();

	for (int i = 0; i < 9; i++)
	{
		int row = i / 3;
		int col = i % 3;
		eigen_out(row, col) = glm_in[col][row];
	}
}

void MatrixConv::Mat42Affine3fTrans(glm::mat4& glm_in, Eigen::Affine3f& eigen_out)
{
	eigen_out = Eigen::Affine3f::Identity();

	eigen_out(0, 3) = glm_in[3][0];
	eigen_out(1, 3) = glm_in[3][1];
	eigen_out(2, 3) = glm_in[3][2];
}
void MatrixConv::Mat42Affine3fRot(glm::mat4& glm_in, Eigen::Affine3f& eigen_out)
{
	eigen_out = Eigen::Affine3f::Identity();

	for (int i = 0; i < 9; i++)
	{
		int row = i / 3;
		int col = i % 3;
		eigen_out(row, col) = glm_in[col][row];
	}
}