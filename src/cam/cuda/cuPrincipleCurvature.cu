

// stl
#include <stdio.h>
#include <iostream>
#include <vector>
#include <algorithm>

// cuda
#include "cuda_runtime.h"
#include "cusolverRf.h"
#include "cublas_v2.h"
#include "cutil_math.h"
#include "cutil_matrix.h"







/*
Compute the principle curvatures for all points in the vector
@param points - array with all points of type float3
@param normals - array with all normal vectors
@param number_nn - integer with the number of nearest neighbors for each point
@param nn_indices - array with all nearest neighbor indices
@param dst_curvatures - vector of type float2 with all curvatures per point
*/
__global__ void cuComputePrincipleCurvature( float3* points, float3* normals, int number_nn, int* nn_indices, float2* dst_curvatures)
{

	int index = (blockIdx.x * blockDim.x) + threadIdx.x;

	float3 N = normals[index];
	float3 P = points[index];

	cuMatrix3 I = cuMatrix3::Identity();

	cuMatrix3 M = I - makeMatrix(N, N);

	float3 normal;
	float3 centroid; 
	centroid.x = 0.0; centroid.y = 0.0; centroid.z = 0.0;


	float3 projected_normals[24];

	for (size_t idx = 1; idx < number_nn; idx++)
	{
		int n_idx = nn_indices[number_nn* index + idx];
		normal = normals[n_idx];

		projected_normals[idx -1] = M * normal;
		centroid += projected_normals[idx - 1];
	}

	// estimate the centroid
	float nn_indices_float = (float)(number_nn);
	centroid /= ((float)(number_nn)- 1.0);


	cuMatrix3 covariance_matrix;
	covariance_matrix.setZero();

	float mean_xy, mean_xz, mean_yz;
	float3 mean;


	for (size_t idx = 0; idx < number_nn-1; idx++)
	{
		mean = projected_normals[idx] - centroid;

		mean_xy = mean.x * mean.y;
		mean_xz = mean.x * mean.z;
		mean_yz = mean.y * mean.z;

		covariance_matrix.data[0] = mean.x * mean.x;
		covariance_matrix.data[3] += (float)(mean_xy);
		covariance_matrix.data[6] += (float)(mean_xz);

		covariance_matrix.data[1] += (float)(mean_xy);
		covariance_matrix.data[4] += mean.y * mean.y;
		covariance_matrix.data[7] += (float)(mean_yz);

		covariance_matrix.data[2] += (float)(mean_xz);
		covariance_matrix.data[5] += (float)(mean_yz);
		covariance_matrix.data[8] += mean.z * mean.z;
	}


	/*
	Matrix I 
	Eigen::Matrix3f I = Eigen::Matrix3f::Identity();


	// project matrix into tangent plane
	Eigen::Matrix3f  M = I - normal_idx * normal_idx.transpose();

	// project normals into tangent plane
	Eigen::Vector3f normal;
	Eigen::Vector3f centroid;
	centroid.setZero();

	vector<Eigen::Vector3f> projected_normals(nn_indices.size() - 1);

	for (size_t idx = 1; idx < nn_indices.size(); idx++) // the first one is the point itself in the nn matrix
	{
		// osg to eigen
		Eigen::Vector3f eigen_normal(normals[nn_indices[idx]].x(), normals[nn_indices[idx]].y(), normals[nn_indices[idx]].z());
		normal = eigen_normal;

		projected_normals[idx - 1] = M * normal;
		centroid += projected_normals[idx - 1];
	}

	// estimate the centroid
	centroid /= static_cast<float> (nn_indices.size() - 1);

	Eigen::Matrix3f covariance_matrix;
	covariance_matrix.setZero();

	double mean_xy, mean_xz, mean_yz;
	Eigen::Vector3f mean;

	for (size_t idx = 0; idx < projected_normals.size(); idx++)
	{
		mean = projected_normals[idx] - centroid;

		mean_xy = mean[0] * mean[1];
		mean_xz = mean[0] * mean[2];
		mean_yz = mean[1] * mean[2];

		covariance_matrix(0, 0) += mean[0] * mean[0];
		covariance_matrix(0, 1) += static_cast<float>(mean_xy);
		covariance_matrix(0, 2) += static_cast<float>(mean_xz);

		covariance_matrix(1, 0) += static_cast<float>(mean_xy);
		covariance_matrix(1, 1) += mean[1] * mean[1];
		covariance_matrix(1, 2) += static_cast<float>(mean_yz);

		covariance_matrix(2, 0) += static_cast<float>(mean_xz);
		covariance_matrix(2, 1) += static_cast<float>(mean_yz);
		covariance_matrix(2, 2) += mean[2] * mean[2];
	}

	//extract the eigenvalues and eigenvectors
	EigenSolver<Eigen::Matrix3f> es(covariance_matrix);

	Eigen::Matrix3f  eigenvalues = es.pseudoEigenvalueMatrix();
	Eigen::Matrix3f  eigenvectors = es.pseudoEigenvectors();

	float index_size = 1.0 / static_cast<float> (projected_normals.size());

	vector<float> ev(3);
	ev[0] = eigenvalues(0);
	ev[1] = eigenvalues(4);
	ev[2] = eigenvalues(8);

	std::sort(ev.begin(), ev.end());


	pc1 = ev[2] * index_size;
	pc2 = ev[1] * index_size;

	return true;
	*/
}