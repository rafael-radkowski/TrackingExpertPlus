#pragma once

/**
Required for class members that should be available on device and host. 
e.g.
class Myclass
{
	CUDA_MEMBER Myclass();

	CUDA_MEMBER void doSomething(void);
};
*/
#ifdef __CUDACC__
#define CUDA_MEMBER __host__ __device__
#else
#define CUDA_MEMBER
#endif 
