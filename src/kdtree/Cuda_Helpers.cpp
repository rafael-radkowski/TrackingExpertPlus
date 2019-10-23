#include "Cuda_Helpers.h"

#include<stdio.h>
using namespace texpert; 

// local
#include "CudaErrorCheck.cu"

//static 
bool Cuda_Helpers::CudaSetDevice(int device)
{
	cudaError_t cudaStatus = cudaSetDevice(device);
	if (cudaStatus != cudaSuccess) {
		std::cerr << "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?" << endl;
		return false;
	}
}



void Cuda_Helpers::SyncCuda(void)
{
	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cudaError_t cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
	}
}


void Cuda_Helpers::HostToDeviceInt(int * device, int* data, int size)
{
	// Copy input vectors from host memory to GPU buffers.
	cudaError_t cudaStatus = cudaMemcpy(device, data, size * sizeof(int), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!\n ");
	}
}



void Cuda_Helpers::DeviveToHostInt(int* host_data, int * device_data, int size)
{
	// Copy input vectors from host memory to GPU buffers.
	cudaError_t cudaStatus = cudaMemcpy(host_data, device_data, size * sizeof(int), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!\n");
	}
}




void Cuda_Helpers::HostToDeviceFloat(float * device, float* data, int size)
{
	// Copy input vectors from host memory to GPU buffers.
	cudaError_t cudaStatus = cudaMemcpy(device, data, size * sizeof(float), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!\n");
	}
}


void Cuda_Helpers::DeviceToHostFloat(float* host_data, float * device_data, int size)
{
	// Copy input vectors from host memory to GPU buffers.
	cudaError_t cudaStatus = cudaMemcpy(host_data, device_data, size * sizeof(float), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!\n");
	}
}


//static 


bool Cuda_Helpers::CheckDeviceCapabilities(void)
{

	int nDevices;

	cudaGetDeviceCount(&nDevices);
	for (int i = 0; i < nDevices; i++) {
		cudaDeviceProp prop;
		cudaGetDeviceProperties(&prop, i);
		printf("Device Number: %d\n", i);
		printf("  Device name: %s\n", prop.name);
		printf("  Memory Clock Rate (KHz): %d\n",
			prop.memoryClockRate);
		printf("  Memory Bus Width (bits): %d\n",
			prop.memoryBusWidth);
		printf("  Peak Memory Bandwidth (GB/s): %f\n",
			2.0*prop.memoryClockRate*(prop.memoryBusWidth / 8) / 1.0e6);
		cout << "  Compute Capability: " << prop.major << "." << prop.minor  << "\n" << endl;
	}

	if(nDevices > 0) return true;
	return false;

}