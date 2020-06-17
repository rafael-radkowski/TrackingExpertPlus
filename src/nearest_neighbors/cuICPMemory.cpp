
#include "cuICPMemory.h"


namespace nscuICPMemory{

	//-----------------------------------------------------------
	// Note that the knn already puts those data onto the gpu.
	// No reason to do it a second time. 

	// the camera point cloud data on the device
	Cuda_Point*			g_dev_data_arr;


	// Search query points
	Cuda_Point*			g_dev_query_points;

	// Search results
	MyMatches*			g_dev_query_results;

	//-----------------------------------------------------------

	// memory for the normal vectors on the device. 
	float3*				g_dev_query_normals;

	// memory for the normal vectors on the device. 
	float3*				g_dev_camera_normals;

	// outlier results
	float*				g_dev_outliers;			

	int					allocate_counter;

	int					width = 1000;
	int					height = 1000;

}


using namespace nscuICPMemory;


bool cuICPMemory::AllocateMemory(void)
{
	allocate_counter = 0;


	int point_size = width* height * 3 * sizeof(float);  // three channels
	int data_size = width* height  * sizeof(float);  // three channels

	// an array for the normal vectors  A(i) = {n0, n1, n2, ...., nN } with each element stores a nomral vector n_i = {nx, ny, nz} as float3
	if (g_dev_query_normals == NULL)
	{
		cudaError err = cudaMalloc((void **)&g_dev_query_normals, (unsigned int)(point_size));
		if (err != 0) { std::cout << "\n[cuICPMemory] - cudaMalloc error (1).\n"; }
		else allocate_counter++;
	}

	// an array for the normal vectors  A(i) = {n0, n1, n2, ...., nN } with each element stores a nomral vector n_i = {nx, ny, nz} as float3
	if (g_dev_camera_normals == NULL)
	{
		cudaError err = cudaMalloc((void **)&g_dev_camera_normals, (unsigned int)(point_size));
		if (err != 0) { std::cout << "\n[cuICPMemory] - cudaMalloc error (2).\n"; }
		else allocate_counter++;
	}

	if (g_dev_outliers == NULL)
	{
		cudaError err = cudaMalloc((void **)&g_dev_outliers, (unsigned int)(data_size));
		if (err != 0) { std::cout << "\n[cuICPMemory] - cudaMalloc error (3).\n"; }
		else allocate_counter++;
	}

	

	if (allocate_counter < 3) {
		std::cout << "[ERROR] - cuICPMemory reports memory allocation errors, alloc = " << allocate_counter << "," << std::endl;
		return false;
	}
	else{
		std::cout << "[INFO] - Memory successfully allocated, alloc = " << allocate_counter << "," << std::endl;
	}
	return true;
}


//static 
bool cuICPMemory::FreeMemory(void)
{
	if (g_dev_query_normals != NULL)
	{
		cudaError err = cudaFree((void **)&g_dev_query_normals);
	}

	// an array for the normal vectors  A(i) = {n0, n1, n2, ...., nN } with each element stores a nomral vector n_i = {nx, ny, nz} as float3
	if (g_dev_camera_normals != NULL)
	{
		cudaError err = cudaFree((void **)&g_dev_camera_normals);
	}

	if (g_dev_outliers != NULL)
	{
		cudaError err = cudaFree((void **)&g_dev_outliers);
	}


	return true;
}



Cuda_Point*	cuICPMemory::GetCameraDataPtr(void)
{
	return g_dev_data_arr;
}

 
void cuICPMemory::SetCameraDataPtr(Cuda_Point* ptr)
{
	g_dev_data_arr = ptr;
}


Cuda_Point*	cuICPMemory::GetQuerryDataPtr(void)
{
	return g_dev_query_points;
}


void cuICPMemory::SetQuerryDataPtr(Cuda_Point* ptr)
{
	g_dev_query_points = ptr;
}


MyMatches*	cuICPMemory::GetSearchResultsPtr(void)
{
	return g_dev_query_results;
}


void  cuICPMemory::SetSearchResultsPtr(MyMatches* ptr)
{
	g_dev_query_results = ptr;
}

//static 
float3* cuICPMemory::GetQuerryNormalPtr(void)
{
	return g_dev_query_normals;
}


//static
float3* cuICPMemory::GetCameraNormalPtr(void)
{
	return g_dev_camera_normals;
}


//static
float* cuICPMemory::GetOutlierResultsPtr(void)
{
	return g_dev_outliers;
}