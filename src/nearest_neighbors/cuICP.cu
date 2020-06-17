#include "cuICP.h"

const int ICP_THREADS_PER_BLOCK = 32;


__device__ bool cu_test_distance(const float3& p0, const float3& p1, const float& max_distance)
{
	float distance = sqrt( (float)( pow(p1.x - p0.x, 2.0f)  + pow(p1.y - p0.y, 2.0f) + pow(p1.z - p0.z, 2.0f) ));
	return (distance < max_distance) ? true : false;
 
}



__device__ float cu_length(const float3& p0)
{
	return sqrt( (float)( pow(p0.x, 2.0f)  + pow(p0.y, 2.0f) + pow(p0.z, 2.0f) ));
}

__device__ float3 cu_normalize(const float3& p0)
{
	float3 p;
	float l = cu_length(p0);
	p.x = p0.x/l;
	p.y = p0.y/l;
	p.z = p0.z/l;

	return p;
}


__device__ bool cu_test_angle(const float3& n0, const float3& n1, const float& max_angle)
{
	// normalize
	float3 n0_ = cu_normalize(n0);
	float3 n1_ = cu_normalize(n1);

	// calculate the dot product
	float dot = n0_.x * n1_.x + n0_.y * n1_.y + n0_.z * n1_.z;

	// for into -1 , 1 range for acos
	dot = fmax(-1.0f, fmin(1.0f, dot));
	
	// return the angle
	float angle = acos(dot);

	return ((angle <= max_angle) || (angle >= (2*3.14159265358979323846 - max_angle)) ? true : false);
}




__global__ void cu_icp_reject_outliers( const int N, const MyMatches* matches, const Cuda_Point* query_point, const Cuda_Point* camera_point,  
										const float3* query_normals, const float3* camera_normals,  
										const float max_dist, const float max_angle, float* results)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	//int i = (blockIdx.x * blockDim.x) + threadIdx.x;
	//int j = (blockIdx.y * blockDim.y) + threadIdx.y;

	if(idx > N) return;

	results[idx] = 0.0; 

	int idx0 = matches[idx].matches[0].first;
	int idx1 = matches[idx].matches[0].second;
	float3 p0, p1;
	p0.x = query_point[idx0]._data[0];
	p0.y = query_point[idx0]._data[1]; 
	p0.z = query_point[idx0]._data[2];

	p1.x = camera_point[idx1]._data[0];
	p1.y = camera_point[idx1]._data[1];
	p1.z = camera_point[idx1]._data[2];


	// check for maximum distance
	bool ret1 = cu_test_distance(p0, p1, max_dist);

	// check for normals
	bool ret2 = cu_test_angle(query_normals[idx0], camera_normals[idx1], max_angle);

	// write back 0 or 1. 
	results[idx] =  (ret1 == true && ret2 == true) ? 1.0f : 0.0f; 

}


__global__ void cu_transform(const int N, const Cuda_Point* query_point, const float3* query_normals) {

	int idx = blockIdx.x * blockDim.x + threadIdx.x;
}

//--------------------------------------------------------------------------------------------------------

/*
Constructor
*/
cuICP::cuICP()
{
	_dev_data_arr = NULL;
	_dev_query_points = NULL;
	_dev_query_results = NULL;
	_dev_camera_normals = NULL;
	_dev_querry_normals = NULL;
	_dev_outliers = NULL;

	// allocate gpu memory
	cuICPMemory::AllocateMemory();
}

cuICP::~cuICP()
{
	// free the memory
	cuICPMemory::FreeMemory();
}
	

/* 
Process the cuda code. 
*/
void cuICP::outlierReject( int N, float* search_points, float* search_normals, int M, float* camera_points, float* camera_normals, std::vector<float>& results)
{
	if (_dev_data_arr == NULL) {
		init();
	}

	cudaError err = cudaMemcpy(_dev_querry_normals, search_normals, N*3*sizeof(float), cudaMemcpyHostToDevice);
	if (err != 0) { 
		std::cout << "\n[cuICP] - cudaMemcpy error (1).\n"; 
	}
	err = cudaGetLastError();
	if (err != 0) {
		std::cout << "\n[cuICP] - cudaMemcpy error (2).\n"; 
	}


	err = cudaMemcpy(_dev_camera_normals, camera_normals, M*3*sizeof(float), cudaMemcpyHostToDevice);
	if (err != 0) { 
		std::cout << "\n[cuICP] - cudaMemcpy error (3).\n"; 
	}
	err = cudaGetLastError();
	if (err != 0) {
		std::cout << "\n[cuICP] - cudaMemcpy error (4).\n"; 
	}

	int blocks = N / ICP_THREADS_PER_BLOCK;
	cu_icp_reject_outliers<<<blocks+1,  ICP_THREADS_PER_BLOCK>>>(N, _dev_query_results, _dev_query_points, _dev_data_arr,  
															_dev_querry_normals, _dev_camera_normals,  
															0.1f, 45.0f/180.0f*3.1415f, _dev_outliers);

	
	results.resize(N);
	err = cudaMemcpy(&results[0], (float*)_dev_outliers, N*sizeof(float), cudaMemcpyDeviceToHost);
	if (err != 0) { 
		std::cout << "\n[cuICP] - cudaMemcpy error (3).\n"; 
	}



}


void cuICP::transform(int N, float cx, float cy, float cz)
{

}

/*
Init
*/
void cuICP::init(void)
{
	std::cout << "Init cuda memory" << endl;

	// camera points
	_dev_data_arr = cuICPMemory::GetCameraDataPtr();
	if(_dev_data_arr == NULL) std::cout << "[ERROR] - cuICP Device memory error (1)" << std::endl;
	
	_dev_query_points  = cuICPMemory::GetQuerryDataPtr();
	if(_dev_query_points == NULL) std::cout << "[ERROR] - cuICP Device memory error (2)" << std::endl;
	
	_dev_query_results = cuICPMemory::GetSearchResultsPtr();
	if(_dev_query_results == NULL) std::cout << "[ERROR] - cuICP Device memory error (3)" << std::endl;

	_dev_querry_normals = cuICPMemory::GetQuerryNormalPtr();
	if(_dev_querry_normals == NULL) std::cout << "[ERROR] - cuICP Device memory error (4)" << std::endl;

	_dev_camera_normals =  cuICPMemory::GetCameraNormalPtr();
	if(_dev_camera_normals == NULL) std::cout << "[ERROR] - cuICP Device memory error (5)" << std::endl;

	_dev_outliers = cuICPMemory::GetOutlierResultsPtr();
	if(_dev_outliers == NULL) std::cout << "[ERROR] - cuICP Device memory error (6)" << std::endl;
}