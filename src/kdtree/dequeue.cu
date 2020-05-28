#include "dequeue.h"
#include <cmath>

__host__ __device__ int int_log2(int x) {
// faster version using special CUDA functions
#ifdef __CUDA_ARCH__
	// Counting from LSB to MSB, number of bits before last '1'
	int n_lower_bits = (8 * sizeof(x)) - __clz(x) - 1;
	return n_lower_bits;
#else
	int ret = 0;
	while (x >>= 1) ++ret;
	return ret;
#endif
}