#include "MatrixConv.h"
#include "PointCloudTrans.h"
#include "RandomGenerator.h"
#include <iostream>

#include "glm/gtc/matrix_transform.hpp"
#include "opencv2/opencv.hpp"

#define PI 3.14159265

MatrixConv* conv;

/*
	--------------------------- MatrixConv class test helpers ---------------------------
*/
//This is done in column-major order, as per glm/Eigen standard
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
		float val0 = comp0[i];
		float val1 = comp1[i];

		if (i <= 11 && (val0 < val1 - 0.000001 || val0 > val1 + 0.000001))
		{
			std::cout << "Error in function " << func_name << " (" << i << "):" << std::endl;
			std::cout << "Input matrix: " << std::endl;
			print_matrix_general(comp0);

			std::cout << std::endl << "Output matrix: " << std::endl;
			print_matrix_general(comp1);
			std::cout << std::endl;

			return false;
		}
		else if (i > 11 && (val1 < ref.data()[i] - 0.000001 || val1 > ref.data()[i] + 0.000001))
		{
			std::cout << "Error in function " << func_name << " (" << i << "):" << std::endl;
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

/*
	--------------------------- PointCloudTrans class test helpers ---------------------------
*/
void makeRandCloudTransforms(int numIters, std::vector<Eigen::Vector3d>& eigen_pts, std::vector<Eigen::Affine3f>& eigen_trans, std::vector<cv::Vec3f>& cv_pts, std::vector<cv::Affine3f>& cv_trans)
{
	for (int i = 0; i < numIters; i++)
	{
		//Create CV affine matrices, then Eigen affine matrices
		std::vector<float> cur_translation = texpert::RandomGenerator::FloatPosition(-1, 1);
		std::vector<float> cur_rotation = texpert::RandomGenerator::FloatPosition(-PI, PI);

		cv::Matx33f R;
		cv::Rodrigues(cv::Vec3f(cur_rotation.at(0), cur_rotation.at(1), cur_rotation.at(2)), R);
		cv_trans.push_back(cv::Affine3f(R, cv::Vec3f(cur_translation.at(0), cur_translation.at(1), cur_translation.at(2))));

		Eigen::Affine3f transformation = Eigen::Affine3f();
		transformation.matrix() << R(0, 0), R(0, 1), R(0, 2), cur_translation.at(0),
			R(1, 0), R(1, 1), R(1, 2), cur_translation.at(1),
			R(2, 0), R(2, 1), R(2, 2), cur_translation.at(2),
			0, 0, 0, 1;
		eigen_trans.push_back(transformation);

		//Create points
		cur_translation = texpert::RandomGenerator::FloatPosition(-1, 1);
		cv_pts.push_back(cv::Vec3f(cur_translation.at(0), cur_translation.at(1), cur_translation.at(2)));
		eigen_pts.push_back(Eigen::Vector3d(cur_translation.at(0), cur_translation.at(1), cur_translation.at(2)));
	}
}



/*
	MatrixConv test function
*/
void doConvTest(std::vector<float> rand_trans, std::vector<float> rand_rot, int* numErrors)
{

	//Initialize error array
	for (int i = 0; i < 15; i++)
	{
		numErrors[i] = 0;
	}

	//Initialize comparison matrices and test their validity
	glm::mat4 mat_in = glm::mat4();
	mat_in = glm::rotate(mat_in, rand_rot.at(0) * 3.14152f / 180.0f, glm::vec3(1, 0, 0));
	mat_in = glm::rotate(mat_in, rand_rot.at(1) * 3.14152f / 180.0f, glm::vec3(0, 1, 0));
	mat_in = glm::rotate(mat_in, rand_rot.at(2) * 3.14152f / 180.0f, glm::vec3(0, 0, 1));
	mat_in = glm::translate(mat_in, glm::vec3(rand_trans.at(0), rand_trans.at(1), rand_trans.at(2)));

	Eigen::Affine3f aff_in = Eigen::Affine3f::Identity();
	conv->Mat42Affine3f(mat_in, aff_in);

	if(!check_error(glm::value_ptr(mat_in), aff_in.data(), "Mat42Affine3f"))
		numErrors[0]++;

	//Test if affine to matrix4f works
	Eigen::Matrix4f matrix_in;
	conv->Mat42Matrix4f(mat_in, matrix_in);

	if(!check_error(glm::value_ptr(mat_in), matrix_in.data(), "Mat42Matrix4f"))
		numErrors[1]++;




	// Declare output objects
	glm::mat4 mat_out;
	glm::vec3 vec_out;
	Eigen::Affine3f aff_out;
	Eigen::Matrix4f matrix_out;


	// Test functions
	conv->Matrix4f2Mat4(matrix_in, mat_out);
	if(!check_error(matrix_in.data(), glm::value_ptr(mat_out), "Matrix4f2Mat4"))
		numErrors[2]++;

	conv->Matrix4f2Vec3Trans(matrix_in, vec_out);
	if(!check_trans_vec_error(matrix_in.data(), glm::value_ptr(vec_out), "Matrix4f2Vec3Trans"))
		numErrors[3]++;

	conv->Matrix4f2Mat4Rot(matrix_in, mat_out);
	if(!check_rot_error(matrix_in.data(), glm::value_ptr(mat_out), "Matrix4f2Mat4Rot"))
		numErrors[4]++;

	conv->Affine3f2Mat4(aff_in, mat_out);
	if(!check_error(aff_in.data(), glm::value_ptr(mat_out), "Affine3f2Mat4"))
		numErrors[5]++;

	conv->Affine3f2Mat4Rot(aff_in, mat_out);
	if(!check_rot_error(aff_in.data(), glm::value_ptr(mat_out), "Affine3f2Mat4Rot"))
		numErrors[6]++;

	conv->Affine3f2Mat4Trans(aff_in, mat_out);
	if(!check_trans_error(aff_in.data(), glm::value_ptr(mat_out), "Affine3f2Mat4Trans"))
		numErrors[7]++;

	conv->Affine3f2Vec3Trans(aff_in, vec_out);
	if(!check_trans_vec_error(matrix_in.data(), glm::value_ptr(vec_out), "Affine3f2Vec3Trans"))
		numErrors[8]++;

	conv->Mat42Affine3fRot(mat_in, aff_out);
	if(!check_rot_error(glm::value_ptr(mat_in), aff_out.data(), "Mat42Affine3fRot"))
		numErrors[9]++;

	conv->Mat42Affine3fTrans(mat_in, aff_out);
	if(!check_trans_error(glm::value_ptr(mat_in), aff_out.data(), "Mat42Affine3fTrans"))
		numErrors[10]++;

	conv->Mat42Matrix4fRot(mat_in, matrix_out);
	if(!check_rot_error(glm::value_ptr(mat_in), matrix_out.data(), "Mat42Matrix4fRot"))
		numErrors[11]++;

	conv->Mat42Matrix4fTrans(mat_in, matrix_out);
	if(!check_trans_error(glm::value_ptr(mat_in), matrix_out.data(), "Mat42Matrix4fTrans"))
		numErrors[12]++;

	conv->Matrix4f2Affine3f(matrix_in, aff_out);
	if(!check_error(matrix_in.data(), aff_out.data(), "Matrix4f2Affine3f"))
		numErrors[13]++;

	conv->Affine3f2Matrix4f(aff_in, matrix_out);
	if(!check_error(aff_in.data(), matrix_out.data(), "Affine3f2Matrix4f"))
		numErrors[14]++;
}

/*
	PointCloudTrans Test Function
*/
void doPointTransTest(int numIters, int* numErrors)
{

	// Initialize error array
	for (int i = 0; i < 4; i++)
	{
		numErrors[i] = 0;
	}

	// Initialize transform matrix/point arrays
	std::vector<cv::Affine3f> cv_trans = std::vector<cv::Affine3f>();
	std::vector<cv::Vec3f> cv_pts = std::vector<cv::Vec3f>();

	std::vector<Eigen::Affine3f> eigen_trans = std::vector<Eigen::Affine3f>();
	std::vector<Eigen::Vector3d> eigen_pts = std::vector<Eigen::Vector3d>();

	//Initialize random values
	makeRandCloudTransforms(numIters, eigen_pts, eigen_trans, cv_pts, cv_trans);

	bool err_occurred = false;

	//Iterate through each matrix
	for (int mat = 0; mat < numIters; mat++)
	{
		cv::Affine3f cur_cv = cv_trans.at(mat);
		Eigen::Affine3f cur_eig = eigen_trans.at(mat);

		//Iterate through each point
		for (int pt = 0; pt < numIters; pt++)
		{
			cv::Vec3f cv_res = cur_cv * cv_pts.at(pt);
			Eigen::Vector3d eigen_res = PointCloudTrans::Transform(cur_eig, eigen_pts.at(pt));

			//Compare the results from each transform
			for (int idx = 0; idx < 3; idx++)
			{
				if (cv_res(idx) < eigen_res(idx) - 0.00001 || cv_res(idx) > eigen_res(idx) + 0.00001)
				{
					err_occurred = true;
					break;
				}
			}
			if (err_occurred)
			{
				numErrors[0]++;
				err_occurred = false;
			}
		}

		//Test point cloud transform
		std::vector<Eigen::Vector3d> eigen_pts_transformed = PointCloudTrans::Transform(eigen_trans.at(mat), eigen_pts);

		for (int i = 0; i < numIters; i++)
		{
			cv::Vec3f res = cv_trans.at(mat) * cv_pts.at(i);

			for (int idx = 0; idx < 3; idx++)
			{
				if (res(idx) < eigen_pts_transformed.at(i)(idx) - 0.00001 || res(idx) > eigen_pts_transformed.at(i)(idx) + 0.00001)
				{
					err_occurred = true;
					break;
				}
			}
		}
		if (err_occurred)
		{
			numErrors[1]++;
			err_occurred = false;
		}
	}

	//Iterate through each matrix
	for (int mat = 0; mat < numIters; mat++)
	{
		cv::Affine3f cur_cv = cv_trans.at(mat);
		Eigen::Affine3f cur_eig = eigen_trans.at(mat);


		//Iterate through each point
		for (int pt = 0; pt < numIters; pt++)
		{
			cv_pts.at(pt) = cur_cv * cv_pts.at(pt);

			PointCloudTrans::TransformInPlace(cur_eig, eigen_pts.at(pt));

			//Compare the results from each transform
			for (int idx = 0; idx < 3; idx++)
			{
				if (cv_pts.at(pt)(idx) < eigen_pts.at(pt)(idx) - 0.00001 || cv_pts.at(pt)(idx) > eigen_pts.at(pt)(idx) + 0.00001)
				{
					err_occurred = true;
					break;
				}
			}
			if (err_occurred)
			{
				numErrors[2]++;
				err_occurred = false;
			}
		}

		//Test point cloud transform
		PointCloudTrans::TransformInPlace(cur_eig, eigen_pts);

		for (int i = 0; i < numIters; i++)
		{
			cv_pts.at(i) = cur_cv * cv_pts.at(i);

			for (int idx = 0; idx < 3; idx++)
			{
				if (cv_pts.at(i)(idx) < eigen_pts.at(i)(idx) - 0.00001 || cv_pts.at(i)(idx) > eigen_pts.at(i)(idx) + 0.00001)
				{
					err_occurred = true;
					break;
				}
			}
		}
		if (err_occurred)
		{
			numErrors[3]++;
			err_occurred = false;
		}
	}
}


/*
	Main test function
*/
int main(int argc, char* argv[])
{
	std::cout << std::endl << "<----------------------Begin MatrixConv tests----------------------->" << std::endl;

	/*
		Initialize test
	*/
	conv = MatrixConv::getInstance();
	int numIters = 10;

	std::vector<float> rand_trans, rand_rot;

	int* err = (int*)malloc(15 * sizeof(int));


	/*
		Run tests
	*/
	for(int i = 0; i < numIters; i++)
	{
		rand_trans = texpert::RandomGenerator::FloatPosition(-1, 1);
		rand_rot = texpert::RandomGenerator::FloatPosition(-179.99f, 179.99f);
		doConvTest(rand_trans, rand_rot, err);
	}

	/*
		Output MatrixConv test results
	*/
	std::cout << "Mat42Affine3f Err: " << ((float)err[0] / (float)numIters) * 100.0f << "%" << std::endl;
	std::cout << "Mat42Matrix4f Err: " << ((float)err[1] / (float)numIters) * 100.0f << "%" << std::endl;
	std::cout << "Matrix4f2Mat4 Err: " << ((float)err[2] / (float)numIters) * 100.0f << "%" << std::endl;
	std::cout << "Matrix4f2Vec3Trans Err: " << ((float)err[3] / (float)numIters) * 100.0f << "%" << std::endl;
	std::cout << "Matrix4f2Mat4Rot Err: " << ((float)err[4] / (float)numIters) * 100.0f << "%" << std::endl;
	std::cout << "Affine3f2Mat4 Err: " << ((float)err[5] / (float)numIters) * 100.0f << "%" << std::endl;
	std::cout << "Affine3f2Mat4Rot Err: " << ((float)err[6] / (float)numIters) * 100.0f << "%" << std::endl;
	std::cout << "Affine3f2Mat4Trans Err: " << ((float)err[7] / (float)numIters) * 100.0f << "%" << std::endl;
	std::cout << "Affine3f2Vec3Trans Err: " << ((float)err[8] / (float)numIters) * 100.0f << "%" << std::endl;
	std::cout << "Mat42Affine3fRot Err: " << ((float)err[9] / (float)numIters) * 100.0f << "%" << std::endl;
	std::cout << "Mat42Affine3fTrans Err: " << ((float)err[10] / (float)numIters) * 100.0f << "%" << std::endl;
	std::cout << "Mat42Matrix4fRot Err: " << ((float)err[11] / (float)numIters) * 100.0f << "%" << std::endl;
	std::cout << "Mat42Matrix4fTrans Err: " << ((float)err[12] / (float)numIters) * 100.0f << "%" << std::endl;
	std::cout << "Matrix4f2Affine3f Err: " << ((float)err[13] / (float)numIters) * 100.0f << "%" << std::endl;
	std::cout << "Affine3f2Matrix4f Err: " << ((float)err[14] / (float)numIters) * 100.0f << "%" << std::endl;

	/*
	End MatrixConv tests
	Begin PointCloudTrans tests
	*/
	std::cout << std::endl << "<----------------------Begin PointCloudTrans tests----------------------->" << std::endl;
	doPointTransTest(numIters, err);

	/*
		Output PointCloudTrans test results
	*/
	std::cout << "Transform (Point) Err: " << ((float)err[0] / (float)(numIters * numIters)) * 100.0f << "%" << std::endl;
	std::cout << "Transform (Cloud) Err: " << ((float)err[1] / (float)numIters) * 100.0f << "%" << std::endl;
	std::cout << "TransformInPlace (Point) Err: " << ((float)err[2] / (float)(numIters * numIters)) * 100.0f << "%" << std::endl;
	std::cout << "TransformInPlace (Cloud) Err: " << ((float)err[3] / (float)numIters) * 100.0f << "%" << std::endl;
}