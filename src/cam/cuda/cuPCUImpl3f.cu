

// cuda
#include "cuda_runtime.h"


// local
#include "cutil_math.h"
#include "cuPCUImpl.h"

/*
Create a uniform sample pattern that will be used for uniform sampling
Each element in the output grid is either 0 or 1 where 1 means to keep the point and 0 means to remove the point. 
*/
__global__  void pcu_create_uniform_sample_pattern(int width, int height, int steps, unsigned short* dst_pattern) {


	int i = (blockIdx.x * blockDim.x) + threadIdx.x; // col
	int j = (blockIdx.y * blockDim.y) + threadIdx.y; // row

	int index = (j * width) + (i );

	if (j % steps == 0 && i % steps == 0)
		dst_pattern[index] = 1;  // use 255 to render the image
	else 
		dst_pattern[index] = 0;
}



/*
Create a uniformly distributed point cloud by sampling over normal vectors and points.
*/
__global__ void pcu_uniform_sampling(float3* src_points, float3* src_normals, int width, int height, unsigned short* sampling_pattern, float3* dst_points, float3* dst_normals)
{
	int i = (blockIdx.x * blockDim.x) + threadIdx.x; // col
	int j = (blockIdx.y * blockDim.y) + threadIdx.y; // row

	int index = (j * width) + i;

	float3 p = src_points[index];
	float3 n = src_normals[index];
	float pattern_value = (float)sampling_pattern[index];

	dst_points[index] = p * pattern_value;
	dst_normals[index] = n * pattern_value;
}




/*
Create a uniformly distributed point cloud by sampling over normal vectors and points.
*/
__global__ void pcu_uniform_sampling(float3* src_points, float3* src_normals, int width, int height, bool cp_active, float* params, unsigned short* sampling_pattern, float3* dst_points, float3* dst_normals)
{

	int i = (blockIdx.x * blockDim.x) + threadIdx.x; // col
	int j = (blockIdx.y * blockDim.y) + threadIdx.y; // row

	int index = (j * width) + i;

	float3 p = src_points[index];
	float3 n = src_normals[index];
	float pattern_value = (float)sampling_pattern[index];

	float v = -1000000.0;
	float clipped = 1.0;
	if(cp_active) v = params[0] * p.x + params[1] * p.y + params[2] * p.z - params[3];
	if (v >= params[4]) clipped = 0.0;

	dst_points[index] = p * pattern_value * clipped;
	dst_normals[index] = n * pattern_value * clipped;
}






__global__ void pcu_init_random_sample_states(unsigned int seed, int width, int height, curandState* states_dev)
{

	int i = (blockIdx.x * blockDim.x) + threadIdx.x; // col
	int j = (blockIdx.y * blockDim.y) + threadIdx.y; // row

	int index = (j * width) + i;

	curand_init(seed, index, 0, &states_dev[index]);
}


/*
Create a random sample pattern to be used for random sampling
Each element in the output grid is either 0 or 1 where 1 means to keep the point and 0 means to remove the point.
@param width - the width of the image in pixel
@param height - the height of the image in pixels
@param steps - the steps to jump in pixels
@param dst_pattern - a pointer to the device memory for the pattern of size width x height x size(unsigned short)
*/
__global__  void pcu_create_random_sample_pattern(unsigned int seed, int width, int height, float percentage, int max_points, unsigned short* dst_pattern)
{

	int i = (blockIdx.x * blockDim.x) + threadIdx.x; // col
	int j = (blockIdx.y * blockDim.y) + threadIdx.y; // row

	int index = (j * width) + i;

	__shared__ curandState state;

	curand_init(seed, index, 0, &state);

	float result_f = curand_uniform(&state);



	/*
	__shared__ curandState_t state;
	__shared__ bool init;
	if (!init)
	{
		curand_init(seed, 
			0,
			0, 
			&state);
		init = true;
	}

	float result = curand_uniform(&state + index);

	float result_f = (float)result;
	*/
	if (result_f < 0.5)
		dst_pattern[index] = 255;  // use 255 to render the image
	else
		dst_pattern[index] = 0;
}



/*
Create a random sample pattern using a list of pixels that should be used for this pattern
@param  index_list - a list with pixel values of the pixels that should be used in this image.
	The index of 1 indicates that the pixel is to be used. An index of 0 rejects the pixel. 
	The index_list ist of grid size with width x height. 
@param width - the width of the image in pixel
@param height - the height of the image in pixels
@param max_points - the number of max points to be used
@param dst_pattern - a pointer to the device memory for the pattern of size width x height x size(unsigned short)
*/
__global__  void pcu_create_random_sample_pattern(int* index_list, int width, int height, int max_points, unsigned short* dst_pattern)
{

	int i = (blockIdx.x * blockDim.x) + threadIdx.x; // col
	int j = (blockIdx.y * blockDim.y) + threadIdx.y; // row

	int index = (j * width) + i;


	if (index_list[index] == 1)
		dst_pattern[index] = 1;  // use 255 to render the image
	else
		dst_pattern[index] = 0;


}



/*
Create a uniformly distributed point cloud by sampling over normal vectors and points.
*/
__global__ void pcu_random_sampling(float3* src_points, float3* src_normals, int width, int height, unsigned short* sampling_pattern, float3* dst_points, float3* dst_normals)
{
	int i = (blockIdx.x * blockDim.x) + threadIdx.x; // col
	int j = (blockIdx.y * blockDim.y) + threadIdx.y; // row

	int index = (j * width) + i;

	float3 p = src_points[index];
	float3 n = src_normals[index];
	float pattern_value = (float)sampling_pattern[index];  // is 0 or 1

	dst_points[index] = p * pattern_value;
	dst_normals[index] = n * pattern_value;
}





/*
Create a uniformly distributed point cloud by sampling over normal vectors and points.
*/
__global__ void pcu_random_sampling(float3* src_points, float3* src_normals, int width, int height, bool cp_active, float* params, unsigned short* sampling_pattern, float3* dst_points, float3* dst_normals)
{

	int i = (blockIdx.x * blockDim.x) + threadIdx.x; // col
	int j = (blockIdx.y * blockDim.y) + threadIdx.y; // row

	int index = (j * width) + i;

	float3 p = src_points[index];
	float3 n = src_normals[index];
	float pattern_value = (float)sampling_pattern[index]; // is either 0 or 1

	float v = -1000000.0;
	float clipped = 1.0;
	if (cp_active) v = params[0] * p.x + params[1] * p.y + params[2] * p.z - params[3];
	if (v >= params[4]) clipped = 0.0;

	dst_points[index] = p * pattern_value * clipped;
	dst_normals[index] = n * pattern_value * clipped;
}

