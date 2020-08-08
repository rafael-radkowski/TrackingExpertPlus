#include "cuFilter.h"

// local
#include "cuDeviceMemory3f.h"


using namespace texpert;

struct  {

	float* devImageMem = NULL;
	float* devImageIn = NULL;
	bool   use_global_memory = true;
	bool   filter_enabled = true;
	bool	verbose = true;

	// bilateral filter
	float	sigmaI = 12.0;
	float	sigmaS = 6.0;
	int		kernel_size = 5;

	unsigned int memory_counter = 0;

}cuFilterMemory;


// due to a few simple runtime tests. 
const int THREADS_PER_BLOCK = 32;

//------------------------------------------------------------------------------------------------------------------
// Cuda kernel


__device__ float distance(int x, int y, int i, int j) {
    return float(sqrtf(powf(x - i, 2.0) + powf(y - j, 2.0)));
}


__device__ float gaussian(float x, float sigma) {
    
	float exp1 = expf(-(powf(x, 2.0))/(2.0 * powf(sigma, 2.0) ) );
	float value2 = (2.0 * CV_PI * powf(sigma, 2.0));
	return exp1/value2;

}

__device__ void cu_get_indices(int x, int y, int width, int height, int* dst_array)
{
	int half = int(height / 2);

	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			dst_array[ i * width + j ] = ( (y - (half - i))) * width + (x - (half - j));
		}
	}
}



/* 
	Apply a bilateral filter on the image. 
	@param src_image_ptr -	A image of size [width x height x channels] organized as an array [v0, v1, v2, ......., V_N].
							The function expects that src_image_ptr is the image in host memory since the function copies the image to device memory. 
							The values are expected to be floats in mm. 
	@param width -			The width of the image in pixels. 
	@param height -			The height of the image in pixels. 
	@param dst_image_ptr -	A float pointer to store (return) the location of the results. Depending on the setting of to_host, the 
							pointer points to either device memory (to_host == false) or copies the entire dataset back to the host (to_host == true). 
	@param to_host			Indicate whether or not the function shoudl copy the image data back to the host. 
					
	

		0  1   2   3   4
	 ---------------------
	 | 0 | 1 | 2 | 3 | 4 |
	 ---------------------
	 | 5 | 6 | 7 | 8 | 9 |
	 ---------------------
	 | 10| 11| 12| 13| 14|		12 is the current pixel
	 ---------------------
	 | 15| 16| 17| 18| 19|
	 ---------------------
	 | 20| 21| 22| 23| 24|
	 ---------------------

 */
__global__ void cu_apply_bilateral_filter5(float* src_image_ptr, int width, int height, int kernel_diameter, float sigmaI, float sigmaS, float* dst_image_ptr)
{
	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;
	

	float iFiltered = 0.0;
	float wP = 0.0;
	int pixel_index = ((y) * width ) + (x );
	int half = int(kernel_diameter / 2);

	if (x < half || y < half || x >(width - half) || y >(height - half)) {
		dst_image_ptr[pixel_index] = 0.0;
		return;
	}

	for (int i = 0; i < kernel_diameter; i++) {
		for (int j = 0; j < kernel_diameter; j++) {
			int neighbor_index =  (y - (half - i)) * width + (x - (half - j));

			float gi = gaussian(src_image_ptr[neighbor_index] - src_image_ptr[pixel_index], sigmaI);
			float gs = gaussian( distance( x, y, x - (half - j), y - (half - i) ), sigmaS);
			float w = gi * gs;
			iFiltered = iFiltered + src_image_ptr[neighbor_index] * w;
            wP = wP + w;
		}
	}

	iFiltered = iFiltered / wP;
    dst_image_ptr[pixel_index] = iFiltered;

}

//------------------------------------------------------------------------------------------------------------------
// Class 

int ncount  = 0;

//static 
void cuFilter::ApplyBilateralFilter(float* src_image_ptr, int width, int height, int channels, float* dst_image_ptr, bool to_host )
{
	
	int input_size = width* height* channels * sizeof( float);
	int output_size = width* height* channels * sizeof(float);  // three channels

	// --------------------------------------------------
	// copy operation
	// if disabled, the functoin needs still to cover the copy operation and copy the data to the device
	if (!cuFilterMemory.filter_enabled) {
		cudaError err = cudaMemcpy(cuFilterMemory.devImageMem , (float*)src_image_ptr, input_size, cudaMemcpyHostToDevice);

		// return device pointer
		dst_image_ptr = cuFilterMemory.devImageMem;

		if(!to_host) return;

		cudaMemcpy(dst_image_ptr, cuFilterMemory.devImageMem, output_size, cudaMemcpyDeviceToHost);


		err = cudaGetLastError();
		if (err != 0) {
			std::cout << "\n[cuFilter] - cudaMemcpy error (6).\n"; 
		}
	}

	if (ncount == 60) {
		std::ofstream of("input_points.csv", std::ofstream::out);
		for (int i = 0; i < height; i++) {
			for (int j = 0; j < width; j++) {
				of << src_image_ptr[i * width + j] << ",";
			}
			of << "\n";
		}
		of.close();
	}


	dim3 threads_per_block(THREADS_PER_BLOCK, THREADS_PER_BLOCK, 1);
	dim3 blocks(width / threads_per_block.x,
		height / threads_per_block.y,
		1);

	cudaError err = cudaMemcpy(cuFilterMemory.devImageIn , (float*)src_image_ptr, input_size, cudaMemcpyHostToDevice);
	if (err != 0) { 
		std::cout << "\n[cuFilter] - cudaMemcpy error (1).\n"; 
	}
	err = cudaGetLastError();
	if (err != 0) {
		std::cout << "\n[cuFilter] - cudaMemcpy error (2).\n"; 
	}

	cudaDeviceSynchronize();

	// compute normal vectors
	cu_apply_bilateral_filter5 << <blocks, threads_per_block >> > (	cuFilterMemory.devImageIn, width, height, cuFilterMemory.kernel_size, 
																	cuFilterMemory.sigmaI, cuFilterMemory.sigmaS, cuFilterMemory.devImageMem);
	err = cudaGetLastError();
	if (err != 0) { std::cout << "\n[cuFilter] - filter processing error.\n"; }

	cudaDeviceSynchronize();

	// return device pointer
	dst_image_ptr = cuFilterMemory.devImageMem;



	if (ncount == 60) {

		std::vector<float> temp_out(width*height,0);
		
		cudaMemcpy(&temp_out[0], cuFilterMemory.devImageMem, output_size, cudaMemcpyDeviceToHost);

		std::ofstream of2("output_points.csv", std::ofstream::out);
		for (int i = 0; i < height; i++) {
			for (int j = 0; j < width; j++) {
				of2 << temp_out[i * width + j] << ",";
			}
			of2 << "\n";
		}
		of2.close();
	}


	ncount++;
	if(!to_host) return;

	cudaMemcpy(dst_image_ptr, cuFilterMemory.devImageMem, output_size, cudaMemcpyDeviceToHost);


	err = cudaGetLastError();
	if (err != 0) {
		std::cout << "\n[cuFilter] - cudaMemcpy error (3).\n"; 
	}
}


/*!
Set the filter params for the bilateral filter. 
@param params - struct of type params. 
*/
//static 
void  cuFilter::SetBilateralFilterParams(cuFilter::Params params)
{
	cuFilterMemory.kernel_size = std::max(3, std::min(params.kernel_size, 25));
	cuFilterMemory.kernel_size = cuFilterMemory.kernel_size - ( 1 - cuFilterMemory.kernel_size%2 ); // an odd number yields a 1, -> 1 - 1  = 0; Even subtracts one. 

	cuFilterMemory.sigmaS = std::max(0.1f, std::min(params.sigmaS, 1000.0f));
	cuFilterMemory.sigmaI = std::max(0.1f, std::min(params.sigmaI, 1000.0f));

	if( cuFilterMemory.verbose) {
		std::cout << "[INFO] - cuFilter::kernel_size = " << cuFilterMemory.kernel_size << std::endl;
		std::cout << "[INFO] - cuFilter::sigmaI = " << cuFilterMemory.sigmaI << std::endl;
		std::cout << "[INFO] - cuFilter::sigmaS = " << cuFilterMemory.sigmaS << std::endl;
	}
}


//static 
void cuFilter::Enable(bool enable)
{
	cuFilterMemory.filter_enabled = enable;
}


/*
Init the device memory. The device memory can be re-used. So no need to always create new memory.
@param width - the width of the image in pixels
@param height - the height of the image in pixels
@param channels - the number of channels. A depth image should have only 1 channel.
*/
//static
void cuFilter::AllocateDeviceMemory(int width, int height, int channels)
{
	if(cuFilterMemory.use_global_memory){
		cuFilterMemory.devImageMem = cuDevMem3f::DevTempImagePtr();
		cuFilterMemory.devImageIn = cuDevMem3f::DevInImagePtr();
	}
	else{

		if (cuFilterMemory.devImageMem == NULL)
		{
			int input_size = width* height* channels * sizeof(float);

			cudaError err = cudaMalloc((void **)&cuFilterMemory.devImageMem, (unsigned int)(input_size));
			if (err != 0) { 
				std::cout << "\n[cuFilter] - cudaMalloc error (devImageMem).\n";
			}
			else{
				if (cuFilterMemory.devImageMem != NULL)
					cuFilterMemory.memory_counter += (unsigned int)(input_size);
			}
		}

		if (cuFilterMemory.devImageIn == NULL)
		{
			int input_size = width* height* channels * sizeof(float);

			cudaError err = cudaMalloc((void **)&cuFilterMemory.devImageIn, (unsigned int)(input_size));
			if (err != 0) { 
				std::cout << "\n[cuFilter] - cudaMalloc error (devImageIn).\n";
			}
			else{
				if (cuFilterMemory.devImageIn != NULL)
					cuFilterMemory.memory_counter += (unsigned int)(input_size);
			}
		}

	}
}


/*
Free all device memory
*/
//static 
void  cuFilter::FreeDeviceMemory(void)
{
	if(!cuFilterMemory.use_global_memory){
		if (cuFilterMemory.devImageMem != NULL){
			cudaFree(cuFilterMemory.devImageMem);
			cuFilterMemory.memory_counter = 0;
		}
	}
}

/*!
Return the device memory this class allocated. 
@return int - allocated memory in bytes. 
*/
//static 
int  cuFilter::GetMemoryCount(void)
{
	return cuFilterMemory.memory_counter;
}