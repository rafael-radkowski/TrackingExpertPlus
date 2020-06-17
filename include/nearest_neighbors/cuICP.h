#pragma once


// stl
#include <iostream>
#include <vector>
#include <numeric>
#include <algorithm>
#include <functional>

// cuda
#include <curand.h>
#include <curand_kernel.h>

// local
#include "Cuda_Types.h"
#include "cuICPMemory.h"

class cuICP {

public:

	/*
	Constructor
	*/
	cuICP();
	~cuICP();
	

	/*
	Process the cuda code. 
	*/
	void outlierReject( int M, float* search_points, float* search_normals, int N, float* camera_points, float* camera_normals, std::vector<float>& results );



	void transform(int N, float cx, float cy, float cz);


private:

	/*
	Init
	*/
	void init(void);




	//----------------------------------------------------
	// Members

	// the camera point cloud data on the device
	Cuda_Point*			_dev_data_arr;

	// Search query points
	Cuda_Point*			_dev_query_points;

	// Search results
	MyMatches*			_dev_query_results;

	// the camera normals, device pointer
	float3*				_dev_camera_normals;

	// the query normals
	float3*				_dev_querry_normals;

	// memory to store the outlier results
	float*				_dev_outliers;
};