#include "MatrixConv.h"
#include "RandomGenerator.h"
#include <iostream>

#include "glm/gtc/matrix_transform.hpp"

void print_mat4(glm::mat4 mat_in)
{
	std::cout << "[" << mat_in[0][0] << ", " << mat_in[1][0] << ", " << mat_in[2][0] << ", " << mat_in[3][0] << "]" << std::endl;
	std::cout << "[" << mat_in[0][1] << ", " << mat_in[1][1] << ", " << mat_in[2][1] << ", " << mat_in[3][1] << "]" << std::endl;
	std::cout << "[" << mat_in[0][2] << ", " << mat_in[1][2] << ", " << mat_in[2][2] << ", " << mat_in[3][2] << "]" << std::endl;
	std::cout << "[" << mat_in[0][3] << ", " << mat_in[1][3] << ", " << mat_in[2][3] << ", " << mat_in[3][3] << "]" << std::endl << std::endl;
}

void print_affine3f(Eigen::Affine3f aff_in)
{
	std::cout << aff_in.matrix() << std::endl << std::endl;
}

void print_matrix4f(Eigen::Matrix4f matrix_in)
{
	std::cout << matrix_in.matrix() << std::endl << std::endl;
}

void print_matrix_general(float* mat)
{
	for (int i = 0; i < 4; i++)
	{
		std::cout << "[" << mat[i] << ", " << mat[i + 4] << ", " << mat[i + 8] << ", " << mat[i + 12] << "]" << std::endl;
	}
}

bool check_error(float* comp0, float* comp1, static char* func_name)
{
	for (int i = 0; i < 16; i++)
	{
		float val0 = comp0[i];
		float val1 = comp1[i];

		if (val0 < val1 - 0.000001 || val0 > val1 + 0.000001)
		{
			std::cout << "Error in function " << func_name << ":" << std::endl;
			std::cout << "Input matrix: " << std::endl;
			print_matrix_general(comp0);

			std::cout << std::endl << "Output matrix: " << std::endl;
			print_matrix_general(comp1);
			std::cout << std::endl;

			return false;
		}
	}

	return true;
}

bool check_rot_error(float* comp0, float* comp1, static char* func_name)
{
	Eigen::Matrix4f ref = Eigen::Matrix4f::Identity();

	for (int i = 0; i < 16; i ++)
	{
		float val0 = comp0[i ];
		float val1 = comp1[i];

		if (i <= 11 && (val0 < val1 - 0.000001 || val0 > val1 + 0.000001))
		{
			std::cout << "Error in function " << func_name << ":" << std::endl;
			std::cout << "Input matrix: " << std::endl;
			print_matrix_general(comp0);

			std::cout << std::endl << "Output matrix: " << std::endl;
			print_matrix_general(comp1);
			std::cout << std::endl;

			return false;
		}
		else if (i > 11 && (val1 < ref.data()[i] - 0.000001 || val1 > ref.data()[i] + 0.000001))
		{
			std::cout << "Error in function " << func_name << ":" << std::endl;
			std::cout << "Input matrix: " << std::endl;
			print_matrix_general(comp0);

			std::cout << std::endl << "Output matrix: " << std::endl;
			print_matrix_general(comp1);
			std::cout << std::endl;

			return false;
		}
	}

	return true;
}

bool check_trans_error(float* comp0, float* comp1, static char* func_name)
{
	Eigen::Matrix4f ref = Eigen::Matrix4f::Identity();

	for (int i = 0; i < 16; i ++)
	{
		float val0 = comp0[i];
		float val1 = comp1[i];

		if (i > 11 && i <= 14 && (val0 < val1 - 0.000001 || val0 > val1 + 0.000001))
		{
			std::cout << "Error in function " << func_name << ":" << std::endl;
			std::cout << "Input matrix: " << std::endl;
			print_matrix_general(comp0);

			std::cout << std::endl << "Output matrix: " << std::endl;
			print_matrix_general(comp1);
			std::cout << std::endl;

			return false;
		}
		else if (i <= 11 && i > 14 && (val1 < ref.data()[i] - 0.000001 || val1 > ref.data()[i] + 0.000001))
		{
			std::cout << "Error in function " << func_name << ":" << std::endl;
			std::cout << "Input matrix: " << std::endl;
			print_matrix_general(comp0);

			std::cout << std::endl << "Output matrix: " << std::endl;
			print_matrix_general(comp1);
			std::cout << std::endl;

			return false;
		}
	}

	return true;
}

bool check_trans_vec_error(float* comp0, float* comp1, static char* func_name)
{
	if (comp1[0] != comp0[12] || comp1[1] != comp0[13] || comp1[2] != comp0[14])
	{
		std::cout << "Error in function " << func_name << ":" << std::endl;
		std::cout << "Input matrix: " << std::endl;
		print_matrix_general(comp0);

		std::cout << std::endl << "Output matrix: " << std::endl;
		print_matrix_general(comp1);
		std::cout << std::endl;

		return false;
	}
	return true;
}

int main(int argc, char* argv[])
{
	/*
		Initialize test
	*/
	MatrixConv* conv = MatrixConv::getInstance();


	std::vector<float> rand_trans, rand_rot;

	rand_trans = texpert::RandomGenerator::FloatPosition(-1, 1);
	rand_rot = texpert::RandomGenerator::FloatPosition(-179.99f, 179.99f);

	//Test if matrix to mat works
	glm::mat4 mat_in = glm::mat4();
	mat_in = glm::rotate(mat_in, rand_rot.at(0) * 3.14152f / 180.0f, glm::vec3(1, 0, 0));
	mat_in = glm::rotate(mat_in, rand_rot.at(1) * 3.14152f / 180.0f, glm::vec3(0, 1, 0));
	mat_in = glm::rotate(mat_in, rand_rot.at(2) * 3.14152f / 180.0f, glm::vec3(0, 0, 1));
	mat_in = glm::translate(mat_in, glm::vec3(rand_trans.at(0), rand_trans.at(1), rand_trans.at(2)));

	print_mat4(mat_in);


	Eigen::Affine3f aff_in = Eigen::Affine3f::Identity();
	conv->Mat42Affine3f(mat_in, aff_in);

	check_error(glm::value_ptr(mat_in), aff_in.data(), "Mat42Affine3f");

	//Test if affine to matrix4f works
	Eigen::Matrix4f matrix_in;
	conv->Mat42Matrix4f(mat_in, matrix_in);

	check_error(glm::value_ptr(mat_in), matrix_in.data(), "Mat42Matrix4f");




	/*
		Declare output objects
	*/
	glm::mat4 mat_out;
	glm::vec3 vec_out;
	Eigen::Affine3f aff_out;
	Eigen::Matrix4f matrix_out;


	/*
		Test functions
	*/

	conv->Matrix4f2Mat4(matrix_in, mat_out);
	check_error(matrix_in.data(), glm::value_ptr(mat_out), "Matrix4f2Mat4");

	conv->Matrix4f2Vec3Trans(matrix_in, vec_out);
	check_trans_vec_error(matrix_in.data(), glm::value_ptr(vec_out), "Matrix4f2Vec3Trans");

	conv->Matrix4f2Mat4Rot(matrix_in, mat_out);
	check_rot_error(matrix_in.data(), glm::value_ptr(mat_out), "Matrix4f2Mat4Rot");

	conv->Affine3f2Mat4(aff_in, mat_out);
	check_error(aff_in.data(), glm::value_ptr(mat_out), "Affine3f2Mat4");

	conv->Affine3f2Mat4Rot(aff_in, mat_out);
	check_rot_error(aff_in.data(), glm::value_ptr(mat_out), "Affine3f2Mat4Rot");

	conv->Affine3f2Mat4Trans(aff_in, mat_out);
	check_trans_error(aff_in.data(), glm::value_ptr(mat_out), "Affine3f2Mat4Trans");

	conv->Affine3f2Vec3Trans(aff_in, vec_out);
	check_trans_vec_error(matrix_in.data(), glm::value_ptr(vec_out), "Affine3f2Vec3Trans");

	conv->Mat42Affine3fRot(mat_in, aff_out);
	check_rot_error(glm::value_ptr(mat_in), aff_out.data(), "Mat42Affine3fRot");

	conv->Mat42Affine3fTrans(mat_in, aff_out);
	check_trans_error(glm::value_ptr(mat_in), aff_out.data(), "Mat42Affine3fTrans");

	conv->Mat42Matrix4fRot(mat_in, matrix_out);
	check_rot_error(glm::value_ptr(mat_in), matrix_out.data(), "Mat42Matrix4fRot");

	conv->Mat42Matrix4fTrans(mat_in, matrix_out);
	check_rot_error(glm::value_ptr(mat_in), matrix_out.data(), "Mat42Matrix4fTrans");

	conv->Matrix4f2Affine3f(matrix_in, aff_out);
	check_error(matrix_in.data(), aff_out.data(), "Matrix4f2Affine3f");

	conv->Affine3f2Matrix4f(aff_in, matrix_out);
	check_error(aff_in.data(), matrix_out.data(), "Affine3f2Matrix4f");
}