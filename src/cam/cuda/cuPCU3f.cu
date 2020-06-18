#include "../../../include/camera/cuda/cuPCU3f.h"

// To use the memory that is defined in cuDeviceMemory/
// If this flag is not set, the device memory that is locally allocated is used. 
#define GLOBAL_CUDA_MEMORY


// local
#include "cutil_math.h"
#include "cuDeviceMemory3f.h"
#include "cuPCUImpl3f.h"

// stl
#include <conio.h>

using namespace texpert;

// To read image data out from the device. 
//#define DEBUG_CUPCU


namespace texpert_cuPCU3f
{

	// The input image on the device
	float*	image_dev;

	// debug images to visualize the output
	float* image_output_dev = NULL;
	float* image_normals_out_dev = NULL;

	// the points and normal vectors
	float3* point_output_dev = NULL;
	float3* normals_output_dev = NULL;


	// the points and normal vectors with all  NAN and [0,0,0] points removed. 
	float3* point_output_clean_dev = NULL;
	float3* normals_output_clean_dev = NULL;
	int* count_dev = NULL; // returns the number of NaN elements that were removed.


	int* dev_index_list = NULL;


	// Memory for sampling pattern, for the uniform sample operation.
	// It stores the sample pattern with 0 and 1. 
	unsigned short* g_cu_sampling_dev = NULL;

	// The number of random pattern that should be used. 
	// The pattern are generated before the frame reader is started and 
	// changed to simulate "random"
	const int max_number_of_random_patterns = 10;

	// the index of the current random pattern in use
	int g_current_random_pattern_index = 0;

	// Memory for a random sampling pattern
	unsigned short* g_cu_random_sampling_dev[max_number_of_random_patterns];


	// 16 threads appears to be a few ms faster than 32 for a 640x480 image
	// due to a few simple runtime tests. 
	const int THREADS_PER_BLOCK = 32;

}

using namespace texpert_cuPCU3f;


__global__ void pcu_cleanup_points(float3* src_points, float3* src_normals, int width, int height, int start, float3* dst_points, float3* dst_normals, int* good_points )
{
	int i = (blockIdx.x * blockDim.x) + threadIdx.x;
	int j = (blockIdx.y * blockDim.y) + threadIdx.y;

	int index = (j * width) + (i);

	__shared__ int count;
	if (count <= 0) count = start;

	if (src_points[index].x == 0 && src_points[index].y == 0 && src_points[index].z == 0) return;

	int local_id = count; count++;
	dst_points[local_id] = src_points[index];
	dst_normals[local_id] = src_normals[index];

	(*good_points) = count;
}




/*
Project a image element at position [x, y] from image space into 3D space using the focal length of the camera.
@param x, y - the x and y position of the image element in the image grid in pixels. 
@param pixel_depth - the depth of this particular pixel as float value.
@param focal_length_x - the focal lenght of the camera in pixel. x is the horizontal axis.
@param focal_length_y - the focal lenght of the camera in pixel. y is the vertical axis.
@param float cx, cy -  the principle point
@param px, py, pz - pointers to the point values for z, y, z.
*/
__device__ void pcu_project_point(float x, float y, float pixel_depth, float focal_length_x, float focal_length_y, float cx, float cy, float* px, float* py, float* pz)
{

	const float fl_const_x = 1.0f / focal_length_x;
	const float fl_const_y = 1.0f / focal_length_y;

	// Conversion from millimeters to meters
	static const float conv_fac = 0.001f;

	float x_val = 0.0f;
	float y_val = 0.0f;
	float z_val = 0.0f;

	if(!isnan(pixel_depth)){
		x_val = -(float) x * pixel_depth * fl_const_x * conv_fac + cx;
		y_val = -(float) y * pixel_depth * fl_const_y * conv_fac + cy;
		z_val = pixel_depth *conv_fac;
	}

	(*px) = x_val;
	(*py) = y_val;
	(*pz) = z_val;

}



/*
Calculates the normal vectors for points. The points must be organized in a grid. 
@param src_points - the points organized in a float grid of size [width x height] and stored as float3 array.
@param width - the width of the image
@param height - the height of the image
@param step_size - the number of points that should be stepped over for the normal vector calculation
@param flip_normal - parameter is multiplied with normal vector. Set to 1 to keep the normal vector, set to -1 to flip it.
@param dst_normals - a float3 array to store the normal vectors as float3 [nx, ny, nz]
@param dst_image - the normal vector data is stored in an image as RGB values. For debug purpose.
*/
__global__ void pcu_calculate_normal_vector( float3* src_points, int width, int height,  int step_size, float flip_normal,  float3* dst_normals, float* dst_image)
{
	int i = (blockIdx.x * blockDim.x) + threadIdx.x;
	int j = (blockIdx.y * blockDim.y) + threadIdx.y;

	int index_center = (j * width ) + (i );
	int size = width * height;

	// for visualization
	int index_out = (j * width * 3) + (i * 3); // three channels for the output image

	// for the src_points of type float, the channel size is 1
	int index_north = ((j - step_size) * width ) + (i  );
	int index_east = (j * width ) + ((i + step_size ));
	int index_south = ((j + step_size) * width ) + (i);
	int index_west = ( j * width ) + ((i- step_size));


	float3 reset_point; reset_point.x = 0.0f; reset_point.y = 0.0f; reset_point.z =0.0f;
	float3 aveNormal; aveNormal.x = 0.0f; aveNormal.y = 0.0f; aveNormal.z = 0.0f; 
	float3 center = src_points[index_center];
	int pointsUsed = 0;

	

	const float max_dist = 0.2f;
	//if (i >= step_size && i < width - step_size  && j >= step_size && j < height - step_size)

	//cross product quadrant 1
	if (i < width-step_size  && j >= step_size)
	{
		
		
		float3 north = src_points[index_north];
		float3 east = src_points[index_east];
		float3 temp = cross(east-center, north - center);
		
		if (isfinite(temp.x) && fabs(east.z - center.z) < max_dist & fabs(north.z - center.z) < max_dist)
		{
			temp = normalize(temp);
			if(!isnan(temp.x)){
				aveNormal += temp;
				pointsUsed++;
			}
		}		
	}

	//cross product quadrant 2
	if (i >= step_size   && j >= step_size)
	{
		float3 north = src_points[index_north];
		float3 west = src_points[index_west];
		float3 temp = cross( north - center, west - center);
		if (isfinite(temp.x) && abs(west.z - center.z) < max_dist && abs(north.z - center.z) < max_dist )
		{
			temp = normalize(temp);
			if(!isnan(temp.x)){
				aveNormal += temp;
				pointsUsed++;
			}
		}
	}

	//cross product quadrant 3
	if (i >= step_size  &&  j < height - step_size)
	{
		float3 south = src_points[index_south];
		float3 west = src_points[index_west];
		float3 temp = cross( west - center, south - center);
		if (isfinite(temp.x) && abs(west.z - center.z) < max_dist && abs(south.z - center.z) < max_dist)
		{
			temp = normalize(temp);
			if(!isnan(temp.x)){
				aveNormal += temp;
				pointsUsed++;
			}
		}
	}


	//cross product quadrant 4
	if ( i < width - step_size &&  j < height - step_size)
	{
		float3 south = src_points[index_south];
		float3 east= src_points[index_east];
		float3 temp = cross( south - center, east - center);
		if (isfinite(temp.x) && abs(east.z - center.z) < max_dist && abs(south.z - center.z) < max_dist)
		{
			temp = normalize(temp);
			if(!isnan(temp.x)){
				aveNormal += temp;
				pointsUsed++;
			}
		}
	}

	// check whether a normal vector exists
	if(pointsUsed > 0){
		aveNormal /= (pointsUsed);

		//make unit vector
		aveNormal = flip_normal * normalize( aveNormal);
		src_points[index_center] = center;
	}else
	{
		// we do not want a point if we do not have a normal vector
		src_points[index_center] = reset_point;
	}
	
	dst_normals[index_center] = aveNormal;
	

	// for visualization
	dst_image[index_out] = aveNormal.z * 255.0;
	dst_image[index_out+1] = aveNormal.y * 255.0;
	dst_image[index_out+2] = aveNormal.x * 255.0;

}


/*
Project all points into 3D space
@param image - the input image of type unsigned short. One channel (with depth images) of size width x height.
@param width - the width of the image in pixel
@param height - the height of the immage in pixel
@param channels - the channels. Should be only one. 
@param focal length - the focal length of the camera.
@param dst_point - pointer to the array with float3 that store all points. 
@param dst_iamge - image with three channels that store the [x, y, z] position of each point in an image frame [i, j]
					The image is for debug purposes or visualization
*/
__global__  void pcu_project_image_points(float *image, int width, int height, int channels, float focal_length_x, float focal_length_y, float cx, float cy, float3* dst_point, float* dst_image) {

	int i = (blockIdx.x * blockDim.x) + threadIdx.x;
	int j = (blockIdx.y * blockDim.y) + threadIdx.y;
	//int k = (blockIdx.z * blockDim.z) + threadIdx.z;


	int index = (j * width * channels) + (i * channels); 
	int index_out = (j * width * 3) + (i * 3); // three channels for the output image

	float depth = (float)image[index];
	float px = 0.0;
	float py = 0.0;
	float pz = 0.0;

	//------------------------------------------------------------------------------
	// projection

	pcu_project_point((float)i-width/2, (float)j-height/2, (float)image[index], focal_length_x, focal_length_y, cx, cy,  &px, &py, &pz);

	// RR, May 5, 2018
	// Adopted to get a mean value from cuPositionFromRect
	const float scale = 1.0;  // to see some colors, scale to something dfferent than 1. 

	// The coordinates are flipped in pcu_project_point, thus, negative
	dst_image[index_out] =  (px * scale);
	dst_image[index_out + 1] = (py * scale); 
	dst_image[index_out + 2] = (pz * scale);
	

	dst_point[index].x = px;
	dst_point[index].y = py;
	dst_point[index].z = pz;

}



/*********************************************************************************************************************************************************************************************
Create a point cloud from a depth image
@param src_image_ptr - a pointer to the image of size [wdith x height x channels ] stored as an array of type float which stores the depth values as
A(i) = {d0, d1, d2, ..., dN} in mm.
@param width - the width of the image in pixels
@param height - the height of the image in pixels
@param channels - the number of channels. A depth image should have only 1 channel.
@param focal_legnth - the focal length of the camera in pixel
@param float cx, cy -  the principle point
@param step_size - for normal vector calculations. The interger specifies how many steps a neighbor sould be away no obtain a vector for normal vector calculation.
Minimum step_size is 1.
@param points - a vector A(i) = {p0, p1, p2, ..., pN} with all points p_i = {px, py, pz} as float3.
@param normals - a vector A(i) = {n0, n1, n2, ..., nN} with all normal vectors n_i = {nx, ny, nz} as float3
@param to_host - if true, the device normals and points are copied back to the host. If false, this is skipped and the  data in points and normals remains empty.
NOTE, COPYING DATA REQUIRES A SIGNIFICANT AMOUNT OF TIME AND SHOULD ONLY BE EXECUTED AT THE VERY LAST STEP
**********************************************************************************************************************************************************************************************/
//static 
int cuPCU3f::CreatePointCloud(float* src_image_ptr, int width, int height, int channels, float focal_length_x, float focal_length_y, float cx, float cy, int step_size, float normal_flip, vector<float3>& points, vector<float3>& normals, bool to_host)
{
/*
	std::ofstream off("test_cu.csv", std::ofstream::out);

	cv::Mat img(480,640,CV_32FC1, src_image_ptr);
	for (int i = 0; i < 480; i++) {
		for (int j = 0; j < 640; j++) {
			off << img.at<float>(i,j) << ",";
		}
		off << "\n";
	}
	off.close();
*/


	step_size = (step_size <= 0) ? 1 : step_size;

	int input_size = width* height* channels * sizeof( float);
	int output_size = width* height* 3 * sizeof(float);  // three channels

	dim3 threads_per_block(THREADS_PER_BLOCK, THREADS_PER_BLOCK, 1);
	dim3 blocks(width / threads_per_block.x,
		height / threads_per_block.y,
		1);

	points.resize(width*height);
	normals.resize(width*height);


	//---------------------------------------------------------------------------------
	// Allocating memory
	// Moved to cuPCU::AllocateDeviceMemory()
	//
	// Allocate memory with AllocateDeviceMemory(.....)

	//---------------------------------------------------------------------------------
	// Copy memory

	cudaError err = cudaMemcpy(image_dev, (float*)src_image_ptr, input_size, cudaMemcpyHostToDevice);
	if (err != 0) { 
		std::cout << "\n[KNN] - cudaMemcpy error.\n"; 
	}
	err = cudaGetLastError();
	if (err != 0) {
		std::cout << "\n[KNN] - cudaMemcpy error (2).\n"; 
	}

	//---------------------------------------------------------------------------------
	// Process the image

	// compute the points 
	pcu_project_image_points << <blocks, threads_per_block >> > (image_dev, width, height, channels, focal_length_x, focal_length_y, cx, cy, point_output_dev, image_output_dev);
	err = cudaGetLastError();
	if (err != 0) { std::cout << "\n[KNN] - points processing error.\n"; }

	cudaDeviceSynchronize();

	// compute normal vectors
	pcu_calculate_normal_vector << <blocks, threads_per_block >> > (point_output_dev, width, height, step_size, normal_flip, normals_output_dev, image_normals_out_dev);
	err = cudaGetLastError();
	if (err != 0) { std::cout << "\n[KNN] - normals points processing error.\n"; }

	cudaDeviceSynchronize();


	if (!to_host) return 1;
		
	//---------------------------------------------------------------------------------
	// Return the data

	cudaMemcpy(&points[0], point_output_dev, output_size, cudaMemcpyDeviceToHost);
	cudaMemcpy(&normals[0], normals_output_dev, output_size, cudaMemcpyDeviceToHost);

	//int removed = 0;
	//cudaMemcpy(&removed, count_dev, sizeof(int), cudaMemcpyDeviceToHost);

	//cout << "Removed " << removed << " points" << endl;


#ifdef DEBUG_CUPCU

	// renders an image of the outcomeing dataset on screen.
	// Note. INCREASE THE SCALE IN pcu_project_image_points(...) TO SEE SOME COLOR.
	// The values are scaled to meters.
	cv::Mat output_points = cv::Mat::zeros(height, width, CV_32FC3);
	cv::Mat output_normals = cv::Mat::zeros(height, width, CV_32FC3);
	cudaMemcpy((float*)output_points.data, image_output_dev, output_size, cudaMemcpyDeviceToHost);
	cudaMemcpy((float*)output_normals.data, image_normals_out_dev, output_size, cudaMemcpyDeviceToHost);

	cv::Mat test_out, test_out_2, test_outf, test_out_2f;
	output_points.convertTo(test_out, CV_8UC3);
	output_normals.convertTo(test_out_2, CV_8UC3);
	cv::flip(test_out, test_outf,1);
	cv::flip(test_out_2, test_out_2f,1);
	cv::imshow("Range image out", test_outf);
	cv::imshow("Normal image out", test_out_2f);
	cv::waitKey();
#endif

	cudaDeviceSynchronize();

	return 1;
}





/*
Init the device memory. The device memory can be re-used. So no need to always create new memory.
@param width - the width of the image in pixels
@param height - the height of the image in pixels
@param channels - the number of channels. A depth image should have only 1 channel.
*/
//static 
void cuPCU3f::AllocateDeviceMemory(int width, int height, int channels)
{


	//---------------------------------------------------------------------------------
	// Allocating memory

	cuDevMem3f::AllocateDeviceMemory(width, height, channels);

	image_dev = cuDevMem3f::DevInImagePtr();
	image_output_dev = cuDevMem3f::DevPointImagePtr();
	image_normals_out_dev = cuDevMem3f::DevNormalsImagePtr();
	point_output_dev = cuDevMem3f::DevPointPtr();
	normals_output_dev = cuDevMem3f::DevNormalsPtr();
}


/*
Free all device memory
*/
//static
void cuPCU3f::FreeDeviceMemory(void)
{

	cudaFree(image_dev);
	cudaFree(image_output_dev);
	cudaFree(image_normals_out_dev);
	cudaFree(point_output_dev);
	cudaFree(normals_output_dev);
	cudaFree(normals_output_clean_dev);
	cudaFree(point_output_clean_dev);


}



/*
Copy the depth image from the device and return it as an openCV mat.
@param depth_image - reference to the depth image as OpenCV Mat of type CV_32FC3
*/
//static 
void cuPCU3f::GetDepthImage(cv::Mat& depth_image)
{
	int output_size = depth_image.rows* depth_image.cols * 3 * sizeof(float);
	cv::Mat output_points = cv::Mat::zeros(depth_image.rows, depth_image.cols, CV_32FC3);
	cudaMemcpy((float*)output_points.data, image_output_dev, output_size, cudaMemcpyDeviceToHost);
	cv::flip(output_points, depth_image, 1);
	
}

/*
Copy the normal vectors encoded into a rgb image from the device and return it as an openCV mat.
@param depth_image - reference to the normal vector image as OpenCV Mat of type CV_32FC3
*/
//static 
void cuPCU3f::GetNormalImage(cv::Mat& normal_image)
{
	int output_size = normal_image.rows* normal_image.cols * 3 * sizeof(float);
	cv::Mat output_normals = cv::Mat::zeros(normal_image.rows, normal_image.cols, CV_32FC3);
	cudaMemcpy((float*)output_normals.data, image_normals_out_dev, output_size, cudaMemcpyDeviceToHost);
	cv::flip(output_normals, normal_image, 1);
}




/*
Create a sample pattern to uniformly remove points from the point set
@param width - the width of the image
@param height - the height of the image
@param sampling_steps - the number of pixels the pattern should step over in each frame
*/
//static 
void cuSample3f::CreateUniformSamplePattern(int width, int height, int sampling_steps)
{

	//------------------------------------------------------------------------------
	// create the sample pattern

	if (g_cu_sampling_dev != NULL) cudaFree(g_cu_sampling_dev);


	int pattern_size = width* height * 1 * sizeof(float);
	// image memory on device. It stores the input image, the depth values as array A(i) = {d0, d1, d2, ...., dN} as float
	cudaError err = cudaMalloc((void **)&g_cu_sampling_dev, (unsigned int)(pattern_size));
	if (err != 0) { std::cout << "\n[KNN] - cudaMalloc error.\n"; }



	dim3 threads_per_block(THREADS_PER_BLOCK, THREADS_PER_BLOCK, 1);
	dim3 blocks(width / threads_per_block.x,
		height / threads_per_block.y,
		1);


	pcu_create_uniform_sample_pattern << <blocks, threads_per_block >> > ( width, height, sampling_steps, g_cu_sampling_dev);
	err = cudaGetLastError();
	if (err != 0) { std::cout << "\n[KNN] - points processing error.\n"; }


	/*

	// For debugging only

	cudaDeviceSynchronize();



	int output_size = width* height * 1 * sizeof(unsigned short);  // three channels

	int size = sizeof(unsigned short);

	cv::Mat output_pattern = cv::Mat::zeros(height, width, CV_16UC1);
	int t = output_pattern.type();
	
	size_t sizeInBytes = output_pattern.total() * output_pattern.elemSize();

	cudaMemcpy((unsigned short*)output_pattern.data, g_cu_sampling_dev, output_size, cudaMemcpyDeviceToHost);

	cv::Mat test_out;
	output_pattern.convertTo(test_out, CV_8UC3);


	cv::imshow("Pattern image out", test_out);
	cv::waitKey();
	*/
}



/*
Create a sample pattern to randomly remove points from the point set
@param width - the width of the image
@param height - the height of the image
@param percentage - a percentage value between 0 and 1 with 1 = 100%
*/
//static 
void cuSample3f::CreateRandomSamplePattern(int width, int height, int max_points, float percentage)
{

	if (g_cu_random_sampling_dev[0] != NULL) for(auto mem: g_cu_random_sampling_dev) cudaFree(mem);

	int pattern_size = width* height * sizeof(float);
	// image memory on device. It stores the input image, the depth values as array A(i) = {d0, d1, d2, ...., dN} as float
	
	for (auto i = 0; i < max_number_of_random_patterns; i++) {
		cudaError err = cudaMalloc((void **)&g_cu_random_sampling_dev[i], (unsigned int)(pattern_size));
		if (err != 0) { _cprintf("\n[cuSample] - cudaMalloc error.\n"); }
	}

	dim3 threads_per_block(THREADS_PER_BLOCK, THREADS_PER_BLOCK, 1);
	dim3 blocks(width / threads_per_block.x,
		height / threads_per_block.y,
		1);

	
	// allocate memory
	cudaError err = cudaMalloc((void **)&dev_index_list, (unsigned int)(width * height * sizeof(int)));
	if (err != 0) { _cprintf("\n[cuSample] - cudaMalloc error.\n"); }

	srand(time(NULL));
	for (auto i = 0; i < max_number_of_random_patterns; i++) {

		vector<int> index_list(width * height, 0);
		//vector<int> numbers;

		for (auto i = 0; i < max_points;i++) {
		//	bool found = false;
			int x = -1; //int count = 0;
			//while (!found)
			{
				x = (rand()*(RAND_MAX + 1) + rand()) % (width * height);
				//vector<int>::iterator p = std::find(numbers.begin(), numbers.end(), x);
				//if (p == numbers.end()) {
				//	found = true;
				//numbers.push_back(x);
				//}
				//if (count++ > 100000) break; // deadlock prevention
			}
			index_list[x] = 1;
		}

		//_cprintf("\nFound %i samples", numbers.size());

		err = cudaMemcpy(dev_index_list, (int*)index_list.data(), (unsigned int)(width * height * sizeof(int)), cudaMemcpyHostToDevice);
		if (err != 0) { _cprintf("\n[cuSample] - cudaMemcpy error.\n"); }

		pcu_create_random_sample_pattern << <blocks, threads_per_block >> > (dev_index_list, width, height, max_points, g_cu_random_sampling_dev[i]);
		err = cudaGetLastError();
		if (err != 0) { _cprintf("\n[cuSample] - pcu_create_random_sample_pattern error.\n"); }



		/*
		// For debugging only

		cudaDeviceSynchronize();

		

		int output_size = width* height * 1 * sizeof(unsigned short);  // three channels

		int size = sizeof(unsigned short);

		cv::Mat output_pattern = cv::Mat::zeros(height, width, CV_16UC1);
		int t = output_pattern.type();

		size_t sizeInBytes = output_pattern.total() * output_pattern.elemSize();

		cudaMemcpy((unsigned short*)output_pattern.data, g_cu_random_sampling_dev[i], output_size, cudaMemcpyDeviceToHost);

		cv::Mat test_out;
		output_pattern.convertTo(test_out, CV_8UC3);


		cv::imshow("Pattern image out", test_out);
		cv::waitKey();

		*/
	}

}




/**********************************************************************************************************************************************************************************************
Create a point cloud from a depth image. The point set is uniformly sampled.

NOTE, use the function CreateSamplePattern() to create the pattern for sampling

@param src_image_ptr - a pointer to the image of size [wdith x height x channels ] stored as an array of type unsigned short which stores the depth values as
A(i) = {d0, d1, d2, ..., dN}
@param width - the width of the image in pixels
@param height - the height of the image in pixels
@param channels - the number of channels. A depth image should have only 1 channel.
@param focal_legnth - the focal length of the camera in pixel
@param normal_radius - for normal vector calculations. The interger specifies how many steps a neighbor sould be away no obtain a vector for normal vector calculation.
Minimum normal_radius is 1.
@param normal_flip - set this value to -1 to flip the normal vectors, otherwise to 1. Ignore all other values. normal_flip is multiplied with the normal vector.  
@param points - a vector A(i) = {p0, p1, p2, ..., pN} with all points p_i = {px, py, pz} as float3.
@param normals - a vector A(i) = {n0, n1, n2, ..., nN} with all normal vectors n_i = {nx, ny, nz} as float3
@param to_host - if true, the device normals and points are copied back to the host. If false, this is skipped and the  data in points and normals remains empty.
NOTE, COPYING DATA REQUIRES A SIGNIFICANT AMOUNT OF TIME AND SHOULD ONLY BE EXECUTED AT THE VERY LAST STEP
**********************************************************************************************************************************************************************************************/
//static 
void cuSample3f::UniformSampling(float* src_image_ptr, int width, int height, float focal_length_x, float focal_length_y, float cx, float cy, int normal_radius, float normal_flip, bool cp_enabled, vector<float3>& points, vector<float3>& normals, bool to_host)
{
	// Uniform sample pattern must be initialized in advance. 
	assert(g_cu_sampling_dev != NULL);


	dim3 threads_per_block(THREADS_PER_BLOCK, THREADS_PER_BLOCK, 1);
	dim3 blocks(width / threads_per_block.x,
		height / threads_per_block.y,
		1);


	//-----------------------------------------------------------
	// Create the point cloud

	cuPCU3f::CreatePointCloud((float*)src_image_ptr, width, height, 1, focal_length_x, focal_length_y, cx, cy, normal_radius, normal_flip, points, normals, false);



	//-----------------------------------------------------------
	// Sampling
	pcu_uniform_sampling <<< blocks, threads_per_block >>> (point_output_dev, normals_output_dev, width, height, cp_enabled, cuDevMem3f::DevParamsPtr(),  g_cu_sampling_dev, point_output_dev, normals_output_dev);
	cudaError err = cudaGetLastError();
	if (err != 0) { std::cout << "\n[KNN] - points processing error.\n"; }

	cudaDeviceSynchronize();


	//---------------------------------------------------------------------------------
	// Return the data
	if (!to_host) return;
	int output_size = width* height * 3 * sizeof(float);  // three channels
	cudaMemcpy(&points[0], point_output_dev, output_size, cudaMemcpyDeviceToHost);
	cudaMemcpy(&normals[0], normals_output_dev, output_size, cudaMemcpyDeviceToHost);
}






/**********************************************************************************************************************************************************************************************
Create a point cloud from a depth image. The point set is randomly sampled.

NOTE, use the function CreateRandomSamplePattern() to create the pattern for sampling

@param src_image_ptr - a pointer to the image of size [wdith x height x channels ] stored as an array of type unsigned short which stores the depth values as
A(i) = {d0, d1, d2, ..., dN}
@param width - the width of the image in pixels
@param height - the height of the image in pixels
@param channels - the number of channels. A depth image should have only 1 channel.
@param focal_legnth - the focal length of the camera in pixel
@param normal_radius - for normal vector calculations. The interger specifies how many steps a neighbor sould be away no obtain a vector for normal vector calculation.
Minimum normal_radius is 1.
@param normal_flip - set this value to -1 to flip the normal vectors, otherwise to 1. Ignore all other values. normal_flip is multiplied with the normal vector.  
@param points - a vector A(i) = {p0, p1, p2, ..., pN} with all points p_i = {px, py, pz} as float3.
@param normals - a vector A(i) = {n0, n1, n2, ..., nN} with all normal vectors n_i = {nx, ny, nz} as float3
@param to_host - if true, the device normals and points are copied back to the host. If false, this is skipped and the  data in points and normals remains empty.
NOTE, COPYING DATA REQUIRES A SIGNIFICANT AMOUNT OF TIME AND SHOULD ONLY BE EXECUTED AT THE VERY LAST STEP
**********************************************************************************************************************************************************************************************/
//static 
void cuSample3f::RandomSampling(float* src_image_ptr, int width, int height, float focal_length, int normal_radius, float normal_flip, bool cp_enabled, vector<float3>& points, vector<float3>& normals, bool to_host)
{
	// Uniform sample pattern must be initialized in advance. 
	assert(g_cu_sampling_dev != NULL);


	dim3 threads_per_block(THREADS_PER_BLOCK, THREADS_PER_BLOCK, 1);
	dim3 blocks(width / threads_per_block.x,
		height / threads_per_block.y,
		1);


	//-----------------------------------------------------------
	// Create the point cloud
	cuPCU3f::CreatePointCloud((float*)src_image_ptr, width, height, 1, focal_length, focal_length, 0.0 ,0.0, normal_radius, normal_flip, points, normals, false);



	//-----------------------------------------------------------
	// Sampling
	pcu_uniform_sampling << < blocks, threads_per_block >> > (point_output_dev, normals_output_dev, width, height, cp_enabled, cuDevMem3f::DevParamsPtr(),
		g_cu_random_sampling_dev[(g_current_random_pattern_index++)% max_number_of_random_patterns], point_output_dev, normals_output_dev);
	cudaError err = cudaGetLastError();
	if (err != 0) { std::cout << "\n[KNN] - points processing error.\n"; }

	cudaDeviceSynchronize();


	//---------------------------------------------------------------------------------
	// Return the data
	if (!to_host) return;
	int output_size = width* height * 3 * sizeof(float);  // three channels
	cudaMemcpy(&points[0], point_output_dev, output_size, cudaMemcpyDeviceToHost);
	cudaMemcpy(&normals[0], normals_output_dev, output_size, cudaMemcpyDeviceToHost);
}





/*
Set parameters for a cutting plane that removes points from the point set.
The plane is defined by A * x + B * y + C * z = D
// where a point gets removed if D - current_D > CP_THRESHOLD
*/
//static 
void cuSample3f::SetCuttingPlaneParams(float a, float b, float c, float d, float threshold)
{
	float* params = cuDevMem3f::DevParamsPtr();

	float host_params[5];

	host_params[0] = a;
	host_params[1] = b;
	host_params[2] = c;
	host_params[3] = d;
	host_params[4] = threshold;

	cudaError err = cudaMemcpy(params, (float*)host_params, 5 * sizeof(float), cudaMemcpyHostToDevice);
	if (err != 0) { std::cout << "\n[cuSample] - cudaMemcpy error.\n"; }
}


