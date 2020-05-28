// cuda
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

// stl
#include <iostream>
#include <vector>
#include <cassert>
#include <conio.h>

// Define this to turn on error checking
#if _DEBUG
#define CUDA_ERROR_CHECK
#endif

#define CudaSafeCall( err ) __cudaSafeCall( err, __FILE__, __LINE__ )

#define CudaCheckError()    __cudaCheckError( __FILE__, __LINE__ )

__device__ __host__ inline void __cudaSafeCall(cudaError err, const char *file, const int line)
{
#ifdef CUDA_ERROR_CHECK
	assert(err == cudaSuccess);
	if (cudaSuccess != err)
	{
	#ifdef __CUDA_ARCH__
		printf("cudaSafeCall() failed at %s:%i : %s\n",
			file, line, cudaGetErrorString(err));
	#else
		fprintf(stderr, "cudaSafeCall() failed at %s:%i : %s\n",
			file, line, cudaGetErrorString(err));
		_cprintf( "cudaSafeCall() failed at %s:%i : %s\n",
			file, line, cudaGetErrorString(err));
	#endif
	}
#endif

	return;
}

__device__ __host__ inline void __cudaCheckError(const char *file, const int line)
{
#ifdef CUDA_ERROR_CHECK
	cudaError err = cudaGetLastError();
	assert(err == cudaSuccess);
	if (cudaSuccess != err)
	{
	#ifdef __CUDA_ARCH__
		printf("cudaCheckError() failed at %s:%i : %s\n",
			file, line, cudaGetErrorString(err));
	#else
		fprintf(stderr, "cudaCheckError() failed at %s:%i : %s\n",
			file, line, cudaGetErrorString(err));
	#endif
	}
#endif

	return;
}
