#pragma once

#include "Cuda_Helpers.h"
#include "Cuda_Common.h"
#include "cub/cub.cuh"

#include <stdio.h>
#include <assert.h>
#include "cuda_runtime.h"

#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/execution_policy.h>


//#define SORT_BASE_EXP 3
//#define SORT_BASE (1 << SORT_BASE_EXP)
#define MIN_SORT_BASE_EXP 3
#define MAX_SORT_BASE_EXP 8

/*!
Radix sort
The main function to that sorts a part of arr[], starting at start and ranges of size n using countsort
NOTE, this search cannot run in parallel
@param arr, the integer array that shall be sorted
@param start, the start index at which the sort should stat
@param n - the number of elements that should be sorted after start
@param index - a auxiliary index array that keeps the index of each point.
@param temp_memory - temporary memory to store the output data. It must be of 2 * n
@param N - the number of elements in the complete array, or the median point for temp_memory
*/
__global__ void partial_radixsort(int* arr, const int start, const int n, int *index, int *temp_memory, const int N, bool prnt);
__device__ void partial_radixsort_func(int* arr, int max, const int start, const int n, int *index, int *temp_memory, const int N, bool prnt);

/*!
@param arr - the data
@param n - the number of elements per array, the number of elements that should get sorted
@param max - the maximum number of elements
@param index - a pointer to the index array.
@param temp_memory - temporary memory to store the output data temporary. must be 2 * max_n.
*/
//__global__ void parallel_radixsort(int* arr, const int n, const int max_n, int *index, int *temp_memory);
__global__ void serial_radixsort(uint32_t* arr, int max, int n_chunks, double chunk_width, int* index, uint32_t* temp_memory, int N);
__global__ void serial_insertionsort(uint32_t* arr, int max, int n_chunks, double chunk_width, int* index, int N);

class ChunkedSorter {
public:

	ChunkedSorter(int n_points, int min_tpb);
	~ChunkedSorter();

	void sort(uint32_t* arr, int* d_index, uint32_t* temp_memory, int N, int level, int max_elem, int tpb);
	void setBase(int exp_pow);

private:

	// Caching allocator for device memory
	//cub::CachingDeviceAllocator g_allocator; 

	size_t hist_memory_size;

	// Global memory per-block histograms
	int* d_blocked_hists;

	// Temporary device memory for CUB device scan
	void* d_scan_temp = NULL;
	size_t scan_temp_size = 0;
};