
// cuda
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <device_functions.h>
#include <conio.h>

//local
#include "sort.h"
//#define SEQ_ONLY

__constant__ int SORT_BASE;
__constant__ int SORT_BASE_EXP;
int HOST_SORT_BASE = -1;
int HOST_SORT_BASE_EXP = -1;

template<typename T>
__host__ __device__ inline T ceil_div(T a, T b) {
	return (a + b - 1) / b;
}


__global__ void serial_radixsort(uint32_t* arr, int max, int n_chunks, double chunk_width, int* index, uint32_t* temp_memory, int N) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= n_chunks) {
		return;
	}
	
	// left and right bounds for sorting, inclusive and exclusive, respectively
	int l_bound = idx == 0 ? chunk_width * idx : (int)(chunk_width * idx) + 1;
	int r_bound = min(N, (int)(chunk_width * (idx + 1)));

	//partial_radixsort_func(arr, max, l_bound, r_bound-l_bound, index, temp_memory, N);
	//int exp_pow = 0;
	//for (int exp = 1; max / exp > 0; exp *= SORT_BASE) {
	for (int exp_pow = 0; max / (1 << exp_pow) > 0; exp_pow += SORT_BASE_EXP) {

		//int output[MAX_POINTS]; // output array
		//int output_index[MAX_POINTS]; // output array
		int i;
#ifndef SEQ_ONLY
		uint16_t count[(1 << MAX_SORT_BASE_EXP)] = { 0 };
		assert((r_bound - l_bound) < (1 << 16));
#else
		uint32_t count[(1 << MAX_SORT_BASE_EXP)] = { 0 };
#endif

		// Store count of occurrences in count[]
		for (i = l_bound; i < r_bound; i++) {
			count[(arr[i] >> exp_pow) % SORT_BASE]++;
		}

		// Change count[i] so that count[i] now contains actual
		//  position of this digit in output[]
		for (i = 1; i < SORT_BASE; i++) {
			count[i] += count[i - 1];
		}

		// Build the output array
		for (i = r_bound - 1; i >= l_bound; i--)
		{
			assert(i < N);
			int key = (arr[i] >> exp_pow) % SORT_BASE;
			temp_memory[l_bound + count[key] - 1] = arr[i];
			temp_memory[l_bound + N + count[key] - 1] = index[i];
			count[key]--;
		}

		// Copy the output array to arr[], so that arr[] now
		// contains sorted numbers according to current digit
		for (i = l_bound; i < r_bound; i++)
		{
			arr[i] = temp_memory[i];
			index[i] = temp_memory[i + N];
		}

		//exp_pow += SORT_BASE_EXP;
	}
}

__global__ void serial_insertionsort(uint32_t* arr, int max, int n_chunks, double chunk_width, int* index, int N) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= n_chunks) {
		return;
	}

	// left and right bounds for sorting, inclusive and exclusive, respectively
	int l_bound = idx == 0 ? chunk_width * idx : (int)(chunk_width * idx) + 1;
	int r_bound = min(N, (int)(chunk_width * (idx + 1)));


	for (int i = 1; i < r_bound - l_bound; ++i) {
		int j = i;
		while (j > 0 && arr[l_bound + j - 1] > arr[l_bound + j]) {
			int old_j_arr = arr[l_bound + j];
			arr[l_bound + j] = arr[l_bound + j - 1];
			arr[l_bound + j - 1] = old_j_arr;

			int old_j_idx = index[l_bound + j];
			index[l_bound + j] = index[l_bound + j - 1];
			index[l_bound + j - 1] = old_j_idx;
			j--;
		}
	}
}

//// A function to do counting sort of arr[] according to
//// the digit represented by exp.
//__device__ void  parallel_countSort(int *arr, const int start, const int n, const int offset, int exp, int *index, int *output)
//{
//	int size = start + n;
//
//	//int output[MAX_POINTS]; // output array
//	//int output_index[MAX_POINTS]; // output array
//	int i, count[10] = { 0 };
//
//	// Store count of occurrences in count[]
//	for (i = start; i < size; i++)
//		count[(arr[i] / exp) % 10]++;
//
//	// Change count[i] so that count[i] now contains actual
//	//  position of this digit in output[]
//	for (i = 1; i < 10; i++)
//		count[i] += count[i - 1];
//
//	// Build the output array
//	for (i = size - 1; i >= start; i--)
//	{
//		output[start + count[(arr[i] / exp) % 10] - 1] = arr[i];
//		output[offset + start + size + count[(arr[i] / exp) % 10] - 1] = index[i];
//		count[(arr[i] / exp) % 10]--;
//	}
//
//	// Copy the output array to arr[], so that arr[] now
//	// contains sorted numbers according to current digit
//	for (i = start; i < size; i++)
//	{
//		arr[i] = output[i];
//		index[i] = output[offset + i + size];
//	}
//}
//
//
///*!
//Radix sort
//The main function to that sorts a part of arr[], starting at start and ranges of size n using countsort
//NOTE, this search cannot run in parallel
//@param arr, the integer array that shall be sorted
//@param start, the start index at wto sort per sub-array
//@param max, the overall number of all points to sort by this parallel call of radix sort.
//@param index - a auxiliary index array that keeps the index of each point.
//@param temp_memory - temporary memory to store the output data. It must be of 2 * n
//*/
//__global__ void parallel_radixsort(int* arr, const int n, const int max, int *index, int *temp_memory)
//{
//	int i = blockIdx.x;// * blockDim.x + threadIdx.x;
//
//	int start = i * n;
//	int offset = max;
//
//	// Find the maximum number to know number of digits
//	int m = partial_getMax(arr, start, n);
//
//	// Do counting sort for every digit. Note that instead
//	// of passing digit number, exp is passed. exp is 10^i
//	// where i is current digit number
//	for (int exp = 1; m / exp > 0; exp *= 10)
//		parallel_countSort(arr, start, n, offset, exp, index, temp_memory);
//
//	//for(int i=start;i<n+start; i++)
//	//	index[i] = offset;
//}



//////////////////////////////////////////////////////////////////////////////////////////////////
//                         Parallel chunked radix sort
//////////////////////////////////////////////////////////////////////////////////////////////////


__global__ void printChunkHist(int* hist, int chunk_id, int chunk_width, int tpb) {
	int blocks_per_chunk = 1 + (chunk_width + 1 + tpb - 1) / tpb; // ceil(chunk_width / tpb)
	int sums[32] = {0};
	for (int d = 0; d < SORT_BASE; d++) {
		printf("digit %02d: ", d);
		for (int b = 0; b < blocks_per_chunk; b++) {
			int idx = (chunk_id * blocks_per_chunk * SORT_BASE) + b + (d * blocks_per_chunk);
			sums[b] += hist[idx];
			printf("%03d ", hist[idx]);
		}
		printf("\n");
	}
	printf("sums      ");
	for (int i = 0; i < 32; i++) {
		printf("%03d ", sums[i]);
	}
	printf("\n");
}

// Parallel chunked count
__global__ void PC_count(uint32_t* arr, int* global_hist, int N, float chunk_width, int tpb, int exp_pow) {
	extern __shared__ int block_hist[];

	// Initialize counts to 0
	if (threadIdx.x < SORT_BASE) {
		block_hist[threadIdx.x] = 0;
		block_hist[threadIdx.x + SORT_BASE] = 0;
	}
	__syncthreads();

	// Index in the array for this thread
	int pt_idx = threadIdx.x + blockDim.x * blockIdx.x;

	// Chunk for this point
	int this_chunk = pt_idx / (chunk_width);
	// Chunk for the first element in this block
	int base_chunk = ((blockDim.x * blockIdx.x)) / chunk_width;
	// Relative chunk for this block, for indexing into the histogram
	int relative_chunk = this_chunk - base_chunk;

	// If this point index exceeds the array bounds or is part of a previously created node, don't process it
	bool splitting = ceil(pt_idx / chunk_width) < ((pt_idx + 1) / chunk_width) && pt_idx != 0;
	if (pt_idx < N && !splitting) {
		// Add this to the block-local histogram, for the correct chunk
		int hist_idx_chunk0 = (arr[pt_idx] >> exp_pow) % SORT_BASE;
		atomicAdd(&block_hist[hist_idx_chunk0 + (relative_chunk * SORT_BASE)], 1);
	}

	__syncthreads();

	//int blocks_per_chunk = 1 + (chunk_width + tpb - 1) / tpb; // ceil(chunk_width / tpb)
	int blocks_per_chunk = 1 + (int)ceil(chunk_width / tpb);
	//V2: int blocks_per_chunk = ceil_div((int)chunk_width - tpb, 2*tpb) * 2 + (((int)chunk_width & tpb - 1) != 0);
	// Index of first point in this chunk
	int chunk_first_idx = this_chunk == 0 ? chunk_width * this_chunk : (int)(chunk_width * this_chunk) + 1;
	// V2: int chunk_first_idx = base_chunk == 0 ? chunk_width * base_chunk : (int)(chunk_width * base_chunk) + 1;
	// Block index of the first block in this chunk
	int chunk_first_block = chunk_first_idx / tpb;
	int relative_block_idx = blockIdx.x - chunk_first_block;

	// Point index at the end of this block
	int pt_end_block = (blockIdx.x + 1) * tpb - 1;
	// Chunk at the end of this block
	int chunk_end_block = pt_end_block / chunk_width;

	// Add local block histogram to global histogram

	if (threadIdx.x < SORT_BASE) {
		int global_hist_start_b0 = (base_chunk * blocks_per_chunk * SORT_BASE) + relative_block_idx;
		atomicAdd(&global_hist[global_hist_start_b0 + (threadIdx.x * blocks_per_chunk)], block_hist[threadIdx.x]);
		//V2: global_hist[global_hist_start_b0 + (threadIdx.x * blocks_per_chunk)] = block_hist[threadIdx.x];
		// TODO: Will this overflow the memory for the last chunk? (&& blockIdx.x < blockDim.x?)
		if (chunk_end_block != base_chunk) {
			int global_hist_start_b1 = ((base_chunk + 1) * blocks_per_chunk * SORT_BASE);
			//V2: int global_hist_start_b1 = global_hist_start_b0 + 1;
			atomicAdd(&global_hist[global_hist_start_b1 + (threadIdx.x * blocks_per_chunk)], block_hist[threadIdx.x + SORT_BASE]);
			//V2: global_hist[global_hist_start_b1 + (threadIdx.x * blocks_per_chunk)] = block_hist[threadIdx.x + SORT_BASE];
		}
	}
}

__global__ void distributeCounts(const uint32_t* __restrict__ arr, int* __restrict__ global_hist, int* __restrict__ index, uint32_t* __restrict__ output, int N, float chunk_width, int tpb, int exp_pow) {
	int idx = threadIdx.x + blockDim.x * blockIdx.x;

	//int blocks_per_chunk = 1 + (chunk_width + tpb - 1) / tpb; // ceil(chunk_width / tpb)
	int blocks_per_chunk = 1 + (int)ceil(chunk_width / tpb);
	// V2: int blocks_per_chunk = ceil_div((int)chunk_width - tpb, 2 * tpb) * 2 + (((int)chunk_width & tpb - 1) != 0);
	//int total_chunks = (N + chunk_width - 1) / chunk_width; // ceil(N / chunk_width)
	int total_chunks = ceil(N / chunk_width);

	if (idx >= total_chunks * blocks_per_chunk) {
		return;
	}

	int chunk_idx = idx / blocks_per_chunk;

	// Endpoint of the histogram range (exclusive)
	// Subtract 1 because the final point is part of a splitting node
	int chunk_end = min(N, (int) (chunk_width * (chunk_idx + 1)));
	int chunk_start = idx == 0 ? chunk_width * chunk_idx : (int)(chunk_width * chunk_idx) + 1;

	// Block index within this chunk
	int relative_block_idx = idx % blocks_per_chunk;
	// Block index relative to the entire array (equivalent to block indices from PC_count)
	int global_block_idx = (chunk_start / tpb) + relative_block_idx;
	int hist_start = max(chunk_start, global_block_idx * tpb);
	int hist_end = min(chunk_end, (global_block_idx + 1) * tpb);
	
	int global_hist_start = (chunk_idx * blocks_per_chunk * SORT_BASE) + relative_block_idx;
	for (int i = hist_end - 1; i >= hist_start; i--) {
		int key = (arr[i] >> exp_pow) % SORT_BASE;
		// Access the summed histogram. Add chunk_idx to account for the points that are part of a node, which don't get counted in the histogram
		// After accessing the value, decrement the value in global_hist (The -- operator)
		int into_idx = (--global_hist[global_hist_start + key * blocks_per_chunk]) + chunk_idx;
		output[into_idx] = arr[i];
		output[N + into_idx] = index[i];
	}
}


/*
Moves points from temporary array back to normal array, and updates the index array
*/
__global__ void move_temp(uint32_t* arr, uint32_t* temp_arr, int* index, const int N) {
	int idx = threadIdx.x + blockDim.x * blockIdx.x;
	if (idx >= N) {
		return;
	}
	arr[idx] = temp_arr[idx];
	index[idx] = temp_arr[idx + N];
}

__global__ void print_chunk(uint32_t* arr, int chunk_idx, int chunk_width) {
	

	int chunk_start = (chunk_width) * chunk_idx;
	for (int i = 0; i < chunk_width-1; i++) {
		printf("%d, ", arr[chunk_start + i]);
	}
	printf("\n");
}

ChunkedSorter::ChunkedSorter(int n_points, int min_tpb){// : g_allocator(true) {

	// The most global memory we will need for per-block, per-chunk histograms
	// Worst case is every block covers two chunks. Se have N/TPB blocks, so we
	// will need two histograms per block, and each histogram is <sort_base> integers
	// TODO: Can change histogram from ints to short or byte, depending on TPB
	int max_sort_base = 1 << MAX_SORT_BASE_EXP;
	size_t max_hist_items = (int)ceil((double)n_points / min_tpb) * 2 * max_sort_base;
	hist_memory_size = max_hist_items * sizeof(int);
	CudaSafeCall( cudaMalloc(&d_blocked_hists, hist_memory_size) );

	// Calculate necessary size of temporary memory for CUB scan
	cub::DeviceScan::InclusiveSum(d_scan_temp, scan_temp_size, d_blocked_hists, d_blocked_hists, max_hist_items);
	CudaSafeCall(cudaMalloc(&d_scan_temp, scan_temp_size));

	// TODO: necessary?
	cudaDeviceSynchronize();

}

ChunkedSorter::~ChunkedSorter() {
	CudaSafeCall(cudaFree(d_blocked_hists));
	CudaSafeCall(cudaFree(d_scan_temp));
}

void ChunkedSorter::setBase(int exp_pow) {
	int sort_base = 1 << exp_pow;
	if (HOST_SORT_BASE_EXP == exp_pow) {
		// Already set
		return;
	}
	cudaMemcpyToSymbol(SORT_BASE_EXP, &exp_pow, sizeof(int), 0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(SORT_BASE, &sort_base, sizeof(int), 0, cudaMemcpyHostToDevice);
	CudaCheckError();
	HOST_SORT_BASE_EXP = exp_pow;
	HOST_SORT_BASE = sort_base;
}

void ChunkedSorter::sort(uint32_t* arr, int* d_index, uint32_t* temp_memory, int N, int level, int max_elem, int tpb) {
	assert(tpb >= HOST_SORT_BASE);

	float chunk_width = (float)N / (1 << level);
	int blocks_per_chunk = 1 + (int)ceil(chunk_width / tpb);
	// V2: int blocks_per_chunk = ceil_div((int)chunk_width - tpb, 2 * tpb) * 2 + (((int)chunk_width & tpb - 1) != 0);
	int n_global_chunks = blocks_per_chunk * (1 << level);
	int n_hist_items = n_global_chunks * HOST_SORT_BASE;
	size_t used_hist_size = n_hist_items * sizeof(int);


	// Allocate temporary storage for parallel scan
	//void* d_temp_storage = NULL;
	//size_t temp_storage_bytes = 0;
	//
	//g_allocator.DeviceAllocate(&d_temp_storage, temp_storage_bytes);

	// Do counting sort for every digit. Note that instead
	// of passing digit number, exp is passed. exp is 10^i
	// where i is current digit number
	int exp_pow = 0;
	for (int exp = 1; max_elem / exp > 0; exp *= HOST_SORT_BASE) {
		// Reset histogram counts to 0
		CudaSafeCall(cudaMemset(d_blocked_hists, 0, used_hist_size));
		//cudaDeviceSynchronize();

		// Counts occurences of digits
		size_t smem_size = (1 << HOST_SORT_BASE_EXP) * 2 * sizeof(int);
		PC_count<<<(N + tpb - 1) / tpb, tpb, smem_size >>>(arr, d_blocked_hists, N, chunk_width, tpb, exp_pow);
		//cudaDeviceSynchronize();
		CudaCheckError();

		// Run
		cub::DeviceScan::InclusiveSum(d_scan_temp, scan_temp_size, d_blocked_hists, d_blocked_hists, n_hist_items);
		//cudaDeviceSynchronize();
		CudaCheckError();
		
		distributeCounts<<<(n_global_chunks + tpb - 1) / tpb, tpb>>>(arr, d_blocked_hists, d_index, temp_memory, N, chunk_width, tpb, exp_pow);

		//cudaDeviceSynchronize();
		CudaCheckError();
		move_temp<<<(N + tpb - 1) / tpb, tpb>>>(arr, temp_memory, d_index, N);

		//cudaDeviceSynchronize();
		CudaCheckError();

		exp_pow += HOST_SORT_BASE_EXP;
	}
}