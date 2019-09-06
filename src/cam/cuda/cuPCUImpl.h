#pragma once

/*
This file contains cuda device code implementing different poinnt cloud utils functions

Features:
 - Create a sample uniform pattern


 Rafael Radkowski
 Iowa State University
 Ames, IA 50011
 rafael@iastate.edu
 Aug. 17, 2017
*/


#include <curand.h>
#include <curand_kernel.h>


/*
Create a uniform sample pattern to be used for uniform sampling
Each element in the output grid is either 0 or 1 where 1 means to keep the point and 0 means to remove the point.
@param width - the width of the image in pixel
@param height - the height of the image in pixels
@param steps - the steps to jump in pixels
@param dst_pattern - a pointer to the device memory for the pattern of size width x height x size(unsigned short)
*/
__global__  void pcu_create_uniform_sample_pattern(int width, int height, int steps, unsigned short* dst_pattern);



__global__ void pcu_init_random_sample_states(unsigned int seed, int width, int height, curandState* states_dev);

/*
Create a random sample pattern to be used for random sampling
Each element in the output grid is either 0 or 1 where 1 means to keep the point and 0 means to remove the point.
@param width - the width of the image in pixel
@param height - the height of the image in pixels
@param percentage - 
@param max_points - the number of max points to be used
@param dst_pattern - a pointer to the device memory for the pattern of size width x height x size(unsigned short)
*/
__global__  void pcu_create_random_sample_pattern(unsigned int seed, int width, int height, float percentage, int max_points, unsigned short* dst_pattern);



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
__global__  void pcu_create_random_sample_pattern(int* index_list,  int width, int height, int max_points, unsigned short* dst_pattern);


/*
Create a uniformly distributed point cloud by sampling over normal vectors and points. 
*/
__global__ void pcu_uniform_sampling(float3* src_points, float3* src_normals, int width, int height,  unsigned short* sampling_pattern, float3* dst_points, float3* dst_normals);


/*
Create a uniformly distributed point cloud by sampling over normal vectors and points.
*/
__global__ void pcu_uniform_sampling(float3* src_points, float3* src_normals, int width, int height, bool cp_active, float* params, unsigned short* sampling_pattern, float3* dst_points, float3* dst_normals);




/*
Create a randomly distributed point cloud by sampling over normal vectors and points.
*/
__global__ void pcu_random_sampling(float3* src_points, float3* src_normals, int width, int height, bool cp_active, float* params, unsigned short* sampling_pattern, float3* dst_points, float3* dst_normals);