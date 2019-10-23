#pragma once
/*
class dequeue

Rafael Radkowski
Iowa State University
rafael@iastate.edu
MIT License
---------------------------------------------------------------
*/
// cuda
#include "cuda_runtime.h"

// local
#include "Cuda_KdTree.h"
//class Cuda_KdNode;


#define MAX_Q 32

class Q_Node {
public:
	Cuda_KdNode* node;
	float priority;
};

class PriorityQueue {
private:
	Q_Node data[MAX_Q]; // Array to hold heap values
	__host__ __device__ void percolateDown(Q_Node item, int idx);
	__host__ __device__ void percolateDownMin(Q_Node item, int idx);
	__host__ __device__ void percolateDownMax(Q_Node item, int idx);
	__host__ __device__ void PriorityQueue::bubbleUp(int idx);
	__host__ __device__ void PriorityQueue::bubbleUpMin(int idx);
	__host__ __device__ void PriorityQueue::bubbleUpMax(int idx);
	__host__ __device__ int parent(const int i) {
		return (i - 1) / 2;
	}
	__host__ __device__ int childLeft(const int i) {
		return 2 * i + 1;
	}
	__host__ __device__ int childRight(const int i) {
		return 2 * i + 2;
	}
	__host__ __device__ void PriorityQueue::swap(int idx1, int idx2);
	__host__ __device__ int largest_idx();
public:
	// Number of items contained
	int size;
	__host__ __device__ PriorityQueue() {
		size = 0;
	}
	__host__ __device__ ~PriorityQueue() {}
	// Inserts a new item into the queue
	__host__ __device__ void insert(Cuda_KdNode* node, float weight);
	// Returns and removes the item with lowest value
	__host__ __device__ Q_Node removeMin();
	// Returns and removes the item with highest value
	__host__ __device__ Q_Node removeMax();
	__host__ __device__ Q_Node peekMax();

};