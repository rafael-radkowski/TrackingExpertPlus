#include "cuDeviceMemory3f.h"


namespace cuDevMem3f_cuDeviceMemory3f
{

	// The input image on the device
	//  OpenNI returns a short (2 byte per pixel).
	float* g_cu_image_dev = NULL;


	// Memory for a float3 array for all points as P = {p0, p1, p2, ..., pN} with pi = {x, y, z}
	float3* g_cu_point_output_dev = NULL;


	// Memory for a float3 array for all normal vector per point as N = {n0, n1, n2, ..., pN} with ni = {nx, ny, nz}
	float3* g_cu_normals_output_dev = NULL;

	// An array made of five parameters to define a cutting plane
	float* g_cu_cutting_plane_params = NULL;

	// debug images to visualize the output
	float* g_cu_image_output_dev = NULL;

	// debug image to visualize the normals 
	float* g_cu_image_normals_out_dev = NULL;


	// the width and the height that was used for the last initialization proces
	int g_cu_width = -1;
	int g_cu_height = -1;

	const int allocations = 7;

	int allocate_counter = 0;
}


using namespace cuDevMem3f_cuDeviceMemory3f;
using namespace texpert; 

/*
Allocate device memory for an image of width x height x channels.
@param width - the width of the image in pixels
@param height - the height of the image in pixels
@param channels - the number of channels. A depth image should have only 1 channel.
*/
void cuDevMem3f::AllocateDeviceMemory(int width, int height, int channels)
{
	// Moves the allocation counter to the global namespace. 
	// from running when the memory is alreay allocated. The api is global and 
	// e.g. multiple attempts to connect to a camera result in an assertation.

	if (g_cu_width == -1 || g_cu_height == -1) {
		g_cu_width = width;
		g_cu_height = height;
		allocate_counter++;
	}

	int input_size = width* height* channels * sizeof(float);
	int output_size = width* height * 3 * sizeof(float);  // three channels

	//---------------------------------------------------------------------------------
	// Allocating memory

	// image memory on device. It stores the input image, the depth values as array A(i) = {d0, d1, d2, ...., dN} as float
	if (g_cu_image_dev == NULL)
	{
		cudaError err = cudaMalloc((void **)&g_cu_image_dev, (unsigned int)(input_size));
		if (err != 0) { std::cout << "\n[cuDevMem] - cudaMalloc error.\n"; }
		else allocate_counter++;
	}

	// an output array where each array element A(i) = {p0, p1, p2, p3, ......, pN} stores a point p_i = {x,y,z} as float3
	if (g_cu_point_output_dev == NULL)
	{
		cudaError err = cudaMalloc((void **)&g_cu_point_output_dev, (unsigned int)(output_size));
		if (err != 0) { std::cout << "\n[cuDevMem] - cudaMalloc error.\n"; }
		else allocate_counter++;
	}

	// an output array where each array element A(i) = {n0, n1, n2, ...., nN } stores a nomral vector n_i = {nx, ny, nz} as float3
	if (g_cu_normals_output_dev == NULL)
	{
		cudaError err = cudaMalloc((void **)&g_cu_normals_output_dev, (unsigned int)(output_size));
		if (err != 0) { std::cout << "\n[cuDevMem] - cudaMalloc error.\n"; }
		else allocate_counter++;
	}

	if (g_cu_cutting_plane_params == NULL)
	{
		cudaError err = cudaMalloc((void **)&g_cu_cutting_plane_params, (unsigned int)(5 * sizeof(float)));
		if (err != 0) { std::cout << "\n[cuDevMem] - cudaMalloc error.\n"; }
		else allocate_counter++;
	}

	// an output image array where each array element A(i) stores a point's components p ={x, y, z} -> x or y or z, the position of the point as float.
	// This memory can be used to render the output as image. 
	if (g_cu_image_output_dev == NULL)
	{
		cudaError err = cudaMalloc((void **)&g_cu_image_output_dev, (unsigned int)(output_size));
		if (err != 0) { std::cout << "\n[cuDevMem] - cudaMalloc error.\n"; }
		else allocate_counter++;
	}


	// an output image gird where each array element A(i) stores the normal vector component of n = {nx, ny, nz} -> nx, ny, nz as float
	// This memory can be used to render the output as image
	if (g_cu_image_normals_out_dev == NULL)
	{
		cudaError err = cudaMalloc((void **)&g_cu_image_normals_out_dev, (unsigned int)(output_size));
		if (err != 0) { std::cout << "\n[cuDevMem] - cudaMalloc error.\n"; }
		else allocate_counter++;
	}



	assert(allocate_counter == allocations);

	if (allocate_counter != allocations)
	{
		_cprintf("\n[cuDevMem] - Memory allocation error. Not all memory got allocated.\n");
		
	}

}



/*
Frees all the memory
*/
void cuDevMem3f::FreeAll(void)
{
	if(g_cu_image_dev != NULL)
		cudaFree(g_cu_image_dev);
	
	if (g_cu_point_output_dev != NULL)
		cudaFree(g_cu_point_output_dev);

	if (g_cu_normals_output_dev != NULL)
		cudaFree(g_cu_normals_output_dev);

	if (g_cu_image_output_dev != NULL)
		cudaFree(g_cu_image_output_dev);

	if (g_cu_image_normals_out_dev != NULL)
		cudaFree(g_cu_image_normals_out_dev);

	if (g_cu_cutting_plane_params != NULL)
		cudaFree(g_cu_cutting_plane_params);
	

	g_cu_width = -1;
	g_cu_height = -1;

	allocate_counter = 0;
}



/*
Return the pointer to the input image memory;
@return - pointer of type unsigned short to the device memory
*/
//unsigned short* cuDevMem::DevInImagePtr(void)
float* cuDevMem3f::DevInImagePtr(void)
{
	return g_cu_image_dev;
}

/*
Return the pointer to the output memory for points stored as float3 array;
@return - pointer of type float3 to the device memory
*/
float3* cuDevMem3f::DevPointPtr(void)
{
	return g_cu_point_output_dev;
}

/*
Return the pointer to the output memory for normal vectors stored as float3 array;
@return - pointer of type float3 to the device memory
*/
float3* cuDevMem3f::DevNormalsPtr(void)
{
	return g_cu_normals_output_dev;
}

/*
Return a pointer to the memory that stores the points' x, y, z, values as array, organized as image grid
[x0 | y0 | z0 | x1 | y1 | z1 | .... | zN | yN | zN ]
*/
float* cuDevMem3f::DevPointImagePtr(void)
{
	return g_cu_image_output_dev;
}

/*
Return a pointer to the memory that stores the points normal vector components nx, ny, nz values as array, organized as image grid
[nx0 | ny0 | nz0 | nx1 | ny1 | nz1 | .... | nzN | nyN | nzN ]
*/
float* cuDevMem3f::DevNormalsImagePtr(void)
{
	return g_cu_image_normals_out_dev;
}


/*
An array for different parameters which are required on the device
*/
//static 
float* cuDevMem3f::DevParamsPtr(void)
{
	return g_cu_cutting_plane_params;
}