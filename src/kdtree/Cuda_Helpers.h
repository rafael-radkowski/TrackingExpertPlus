#pragma once
/*
class Cuda_Helpers

Rafael Radkowski
Iowa State University
rafael@iastate.edu
MIT License
---------------------------------------------------------------
*/

// cuda
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

// stl
#include <iostream>
#include <vector>

// local
#include "CudaErrorCheck.cu"

using namespace std;

namespace texpert {

class Cuda_Helpers
{
public:

	/**
	Set the cuda device
	@param device, a device number started with 0
	@return, true, if the device could be set. 
	*/
	static bool CudaSetDevice(int device);

	/**
	Sync all cuda devices
	*/
	static void SyncCuda(void);

	/**
	Copy integer values from 
	*/
	static void HostToDeviceInt(int * device, int* data, int size);

	static void DeviveToHostInt(int* host_data, int * device_data, int size);

	static void HostToDeviceFloat(float * device, float* data, int size);

	static void DeviceToHostFloat(float* host_data, float * device_data, int size);

	template <typename T>
	static void DeviceToHost( T* device_data, T* host_data, size_t size);

	template <typename T>
	static void HostToDevice( T* host_data, T* device_data, size_t size);

	static bool CheckDeviceCapabilities(void);



};



template <typename T>
void Cuda_Helpers::DeviceToHost(T* device_data, T* host_data, size_t size)
{

	// Copy input vectors from  GPU buffers to host memory.
	cudaError_t cudaStatus = cudaMemcpy(host_data, device_data, size * sizeof(T), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!\n");
	}
}

//static 
template <typename T>
void Cuda_Helpers::HostToDevice(T* host_data, T * device_data, size_t size)
{

	// Copy input vectors from host memory to GPU buffers.
	cudaError_t cudaStatus = cudaMemcpy(device_data, host_data, size * sizeof(T), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!\n");
	}
}

} //texpert 
