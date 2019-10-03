/*
class cutil_matrix

Rafael Radkowski
Iowa State University
rafael@iastate.edu
MIT License
---------------------------------------------------------------
*/

#pragma once
// stl
#include <stdio.h>
#include <iostream>
#include <vector>
#include <algorithm>

// cuda
#include "cuda_runtime.h"


typedef struct
{
	int width;
	int height;
	float* elements;


} cuMatrix;



typedef struct _cuMatrix3
{
	int width;
	int height;
	int size;
	float data[16];

	__device__ __host__ _cuMatrix3() {
		width = 3;
		height = 3;
		size = width * height;
		data[0] = 1.0;
		data[4] = 1.0;
		data[8] = 1.0;
	}

	float& __device__ __host__ operator[](int i) {
		return data[i];
	}

	float& __device__ __host__ operator()(int row, int col)
	{
		return data[row * width + col];
	}


	_cuMatrix3 __device__ __host__ transpose(void)
	{
		cuMatrix3  R;
		R[0] = data[0];
		R[1] = data[3];
		R[2] = data[6];

		R[3] = data[1];
		R[4] = data[4];
		R[5] = data[7];

		R[6] = data[2];
		R[7] = data[5];
		R[8] = data[8];


		return R;
	}

	_cuMatrix3 __device__ __host__ setZero(void)
	{
		for (int i = 0; i < size; i++) data[i] = 0.0;
	}
	


	inline static __device__ __host__ _cuMatrix3 Identity(void)
	{
		cuMatrix3  R;
		for (int i = 0; i < R.size; i++)R[i] = 0.0;

		R[0] = 1;
		R[4] = 1;
		R[8] = 1;
		return R;
	}

} cuMatrix3;



// Matrix4 functions
////////////////////////////////////////////////////////////////////////////////

inline __device__ __host__ cuMatrix3 operator*(cuMatrix3 A, cuMatrix3 B) {

	cuMatrix3  R;
	R(0, 0) = A[0] * B[0] + A[3] * B[1] + A[6] * B[2];
	R(1, 0) = A[1] * B[0] + A[4] * B[1] + A[7] * B[2];
	R(2, 0) = A[2] * B[0] + A[5] * B[1] + A[8] * B[2];

	R(0, 1) = A[0] * B[3] + A[3] * B[4] + A[6] * B[5];
	R(1, 1) = A[1] * B[3] + A[4] * B[4] + A[7] * B[5];
	R(2, 1) = A[2] * B[3] + A[5] * B[4] + A[8] * B[5];

	R(0, 2) = A[0] * B[6] + A[3] * B[7] + A[6] * B[8];
	R(1, 2) = A[1] * B[6] + A[4] * B[7] + A[7] * B[8];
	R(2, 2) = A[2] * B[6] + A[5] * B[7] + A[8] * B[8];



	return R;
}

inline __device__ __host__ float3 operator*(cuMatrix3 A, float3 b) {

	float3 R;

	R.x = A[0] * b.x + A[3] * b.y * A[6] * b.z;
	R.y = A[1] * b.x + A[4] * b.y * A[7] * b.z;
	R.z = A[2] * b.x + A[5] * b.y * A[8] * b.z;
}


inline __device__ __host__ cuMatrix3 operator-(cuMatrix3 A, cuMatrix3 B) {
	cuMatrix3  R;

	for (int i = 0; i < A.size; i++) R[i] = A[i] - B[i];

	return R;
}



inline __device__ __host__ cuMatrix3 makeMatrix(float3 A, float3 At)
{
	cuMatrix3  R;
	R(0, 0) = A.x * At.x;
	R(0, 1) = A.x * At.y;
	R(0, 2) = A.x * At.z;

	R(1, 0) = A.y * At.x;
	R(1, 1) = A.y * At.y;
	R(1, 2) = A.y * At.z;

	R(2, 0) = A.z * At.x;
	R(2, 1) = A.z * At.y;
	R(2, 2) = A.z * At.z;

	return R;
}

