#include "MatrixConv.h"

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

void MatrixConv::EigenMat4ToGlmMat4(Eigen::Matrix4f& eigen_in, glm::mat4& glm_out)
{
	for (int i = 0; i < 4; i++)
	{
		for (int j = 0; j < 4; j++)
		{
			glm_out[i][j] = eigen_in(i, j);
		}
	}
}

void MatrixConv::EigenAff3ToGlmMat4(Eigen::Affine3f& eigen_in, glm::mat4& glm_out)
{
	for (int i = 0; i < 4; i++)
	{
		for (int j = 0; j < 4; j++)
		{
			glm_out[i][j] = eigen_in(i, j);
		}
	}
}

void MatrixConv::EigenAff3ToGlmMat4Trans(Eigen::Affine3f& eigen_in, glm::mat4& glm_out)
{
	glm_out = glm::mat4(1.0f);
	glm_out[0][3] = eigen_in(0, 3);
	glm_out[1][3] = eigen_in(1, 3);
	glm_out[2][3] = eigen_in(2, 3);
}

void MatrixConv::EigenAff3ToGlmMat4Rot(Eigen::Affine3f& eigen_in, glm::mat4& glm_out)
{
	glm_out = glm::mat4(1.0f);
	for (int i = 0; i < 3; i++)
	{
		for (int j = 0; j < 3; j++)
		{
			glm_out[i][j] = eigen_in(i, j);
		}
	}
}

void MatrixConv::EigenAff3ToGlmVec3Trans(Eigen::Affine3f& eigen_in, glm::vec3& glm_out)
{
	glm_out = glm::vec3(eigen_in(0, 3), eigen_in(1, 3), eigen_in(2, 3));
}

void MatrixConv::EigenMat4ToGlmVec3Trans(Eigen::Matrix4f& eigen_in, glm::vec3& glm_out)
{
	glm_out = glm::vec3(eigen_in(0, 3), eigen_in(1, 3), eigen_in(2, 3));
}

void MatrixConv::EigenMat4ToGlmMat4Rot(Eigen::Matrix4f& eigen_in, glm::mat4& glm_out)
{
	glm_out = glm::mat4(1.0f);
	for (int i = 0; i < 3; i++)
	{
		for (int j = 0; j < 3; j++)
		{
			glm_out[i][j] = eigen_in(i, j);
		}
	}
}

void MatrixConv::EigenMat4ToEigenAff3(Eigen::Matrix4f& in, Eigen::Affine3f& out)
{
	out.matrix() = in.matrix();
}

void MatrixConv::EigenAff3ToEigenMat4(Eigen::Affine3f& in, Eigen::Matrix4f& out)
{
	out.matrix() = in.matrix();
}


void MatrixConv::GlmMat4ToEigenMat4(glm::mat4& glm_in, Eigen::Matrix4f& eigen_out)
{
	for (int i = 0; i < 4; i++)
	{
		for (int j = 0; j < 4; j++)
		{
			eigen_out(i, j) = glm_in[i][j];
		}
	}
}

void MatrixConv::GlmMat4ToEigenAff3(glm::mat4& glm_in, Eigen::Affine3f& eigen_out)
{
	for (int i = 0; i < 4; i++)
	{
		for (int j = 0; j < 4; j++)
		{
			eigen_out(i, j) = glm_in[i][j];
		}
	}
}

void MatrixConv::GlmMat4ToEigenMat4Trans(glm::mat4& glm_in, Eigen::Matrix4f& eigen_out)
{
	eigen_out = Eigen::Matrix4f::Identity();

	eigen_out(0, 3) = glm_in[0][3];
	eigen_out(1, 3) = glm_in[1][3];
	eigen_out(2, 3) = glm_in[2][3];
}

void MatrixConv::GlmMat4ToEigenMat4Rot(glm::mat4& glm_in, Eigen::Matrix4f& eigen_out)
{
	eigen_out = Eigen::Matrix4f::Identity();

	for (int i = 0; i < 3; i++)
	{
		for (int j = 0; j < 3; j++)
		{
			eigen_out(i, j) = glm_in[i][j];
		}
	}
}

void MatrixConv::GlmMat4ToEigenAff3Trans(glm::mat4& glm_in, Eigen::Affine3f& eigen_out)
{
	eigen_out = Eigen::Affine3f::Identity();

	eigen_out(0, 3) = glm_in[0][3];
	eigen_out(1, 3) = glm_in[1][3];
	eigen_out(2, 3) = glm_in[2][3];
}
void MatrixConv::GlmMat4ToEigenAff3Rot(glm::mat4& glm_in, Eigen::Affine3f& eigen_out)
{
	eigen_out = Eigen::Affine3f::Identity();

	for (int i = 0; i < 3; i++)
	{
		for (int j = 0; j < 3; j++)
		{
			eigen_out(i, j) = glm_in[i][j];
		}
	}
}