#pragma once
#include "cuda_runtime.h"


template <typename T>
class Q_Node {
public:
	T node;
	float priority;
};

template <typename T, int MAX_Q>
class PriorityQueue {
private:
	Q_Node<T> data[MAX_Q]; // Array to hold heap values
	__host__ __device__ void percolateDown(Q_Node<T> item, int idx);
	__host__ __device__ void percolateDownMin(Q_Node<T> item, int idx);
	__host__ __device__ void percolateDownMax(Q_Node<T> item, int idx);
	__host__ __device__ void bubbleUp(int idx);
	__host__ __device__ void bubbleUpMin(int idx);
	__host__ __device__ void bubbleUpMax(int idx);
	__host__ __device__ int parent(const int i) {
		return (i - 1) / 2;
	}
	__host__ __device__ int childLeft(const int i) {
		return 2 * i + 1;
	}
	__host__ __device__ int childRight(const int i) {
		return 2 * i + 2;
	}
	__host__ __device__ void swap(int idx1, int idx2);
	__host__ __device__ int largest_idx();
public:
	// Number of items contained
	int size;
	__host__ __device__ PriorityQueue(): size(0) {}
	// Inserts a new item into the queue
	__host__ __device__ void insert(const T node, float weight);
	// Returns and removes the item with lowest value
	__host__ __device__ Q_Node<T> removeMin();
	// Returns and removes the item with highest value
	__host__ __device__ Q_Node<T> removeMax();
	__host__ __device__ Q_Node<T> peekMax();

};

/*
////////////////////////////////////////////////////////////////////////
--------------------Priority Queue Implementation-----------------------
////////////////////////////////////////////////////////////////////////
*/

__host__ __device__ int int_log2(int x);

template <typename T, int MAX_Q>
__host__ __device__ void PriorityQueue<T, MAX_Q>::insert(const T node_data, float weight) {
	//assert(size < MAX_Q); // Must not overfill array
	//Q_Node newN = { node, weight };

	// If full, replace the current highest with this, only if it is lower
	if (size == MAX_Q) {
		if (peekMax().priority >= weight) {
			removeMax();
		}
		else {
			return;
		}
	}
	size += 1;
	int idx = size - 1;
	data[idx] = { node_data, weight };
	bubbleUp(idx);
	//while (idx > 0) {
	//	int parent_idx = parent(idx);
	//	if (data[parent_idx].priority > weight) {
	//		data[idx] = data[parent_idx];
	//		idx = parent_idx;
	//	}
	//	else {
	//		break;
	//	}
	//}
	//data[idx] = { node, weight };
}

template <typename T, int MAX_Q>
__host__ __device__ void PriorityQueue<T, MAX_Q>::swap(int idx1, int idx2) {
	Q_Node<T> original_idx1_val = data[idx1];
	data[idx1] = data[idx2];
	data[idx2] = original_idx1_val;
}

template <typename T, int MAX_Q>
__host__ __device__ void PriorityQueue<T, MAX_Q>::bubbleUp(int idx) {
	if (int_log2(idx + 1) % 2 == 0) {
		if (parent(idx) >= 0 && data[idx].priority > data[parent(idx)].priority) {
			swap(idx, parent(idx));
			bubbleUpMax(parent(idx));
		}
		else {
			bubbleUpMin(idx);
		}
	}
	else {
		if (parent(idx) >= 0 && data[idx].priority < data[parent(idx)].priority) {
			swap(idx, parent(idx));
			bubbleUpMin(parent(idx));
		}
		else {
			bubbleUpMax(idx);
		}
	}
}

template <typename T, int MAX_Q>
__host__ __device__ void PriorityQueue<T, MAX_Q>::bubbleUpMin(int idx) {
	// While idx has a grandparent
	while (idx >= 3) {
		int grandparent_idx = parent(parent(idx));
		if (data[idx].priority < data[grandparent_idx].priority) {
			swap(idx, grandparent_idx);
			idx = grandparent_idx;
		}
		else {
			break;
		}
	}
}

template <typename T, int MAX_Q>
__host__ __device__ void PriorityQueue<T, MAX_Q>::bubbleUpMax(int idx) {
	// While idx has a grandparent
	while (idx >= 3) {
		int grandparent_idx = parent(parent(idx));
		if (data[idx].priority > data[grandparent_idx].priority) {
			swap(idx, grandparent_idx);
			idx = grandparent_idx;
		}
		else {
			break;
		}
	}
}

template <typename T, int MAX_Q>
__host__ __device__ Q_Node<T> PriorityQueue<T, MAX_Q>::removeMin() {
	//assert(size > 0); // Canot remove from empty queue
	Q_Node<T> to_return = data[0];
	size--;
	data[0] = data[size];
	percolateDown(data[size], 0);
	return to_return;
}

template <typename T, int MAX_Q>
__host__ __device__ int PriorityQueue<T, MAX_Q>::largest_idx() {
	int largest;
	if (size >= 2) {
		if (size == 2) {
			// Only 2 nodes, so maximum is the only node on level 1 (the first max level)
			largest = 1;
		}
		else {
			// Largest of left or right on level 1
			if (data[1].priority > data[2].priority) {
				largest = 1;
			}
			else { largest = 2; }
		}
	}
	else { largest = 0; } // Only one node
	return largest;
}

template <typename T, int MAX_Q>
__host__ __device__ Q_Node<T> PriorityQueue<T, MAX_Q>::removeMax() {
	int largest = largest_idx();
	Q_Node<T> to_return = data[largest];
	size--;
	data[largest] = data[size];
	percolateDown(data[size], largest);
	return to_return;
}

template <typename T, int MAX_Q>
__host__ __device__ Q_Node<T> PriorityQueue<T, MAX_Q>::peekMax() {
	int largest = largest_idx();
	return data[largest];
}

template <typename T, int MAX_Q>
__host__ __device__ void PriorityQueue<T, MAX_Q>::percolateDown(Q_Node<T> item, int idx) {
	if (int_log2(idx + 1) % 2 == 0) {
		// Even, min-level
		percolateDownMin(item, idx);
	}
	else {
		// Odd, max-level
		percolateDownMax(item, idx);
	}
	//// If smallest child is less than item
	//if (data[smallestChild].priority < item.priority) {
	//	// Move smallest child up
	//	data[idx] = data[smallestChild];
	//	idx = smallestChild;
	//}
	//// Otherwise done
	//else {
	//	break;
	//}
}

template <typename T, int MAX_Q>
__host__ __device__ void PriorityQueue<T, MAX_Q>::percolateDownMin(Q_Node<T> item, int idx) {
	// While node at idx has children
	while (childLeft(idx) < size) {
		// Find the smallest among children and grandchildren
		int descendents[] = {/*childLeft(idx), */childRight(idx),
			childLeft(childLeft(idx)), childRight(childLeft(idx)),
			childLeft(childRight(idx)), childRight(childRight(idx)) };
		int smallestChild = childLeft(idx);
		for (int desc_idx : descendents) {
			if (desc_idx < size && data[desc_idx].priority < data[smallestChild].priority) {
				smallestChild = desc_idx;
			}
		}

		if (smallestChild > childRight(idx)) {
			// smallest is a grandchild
			if (data[smallestChild].priority < data[idx].priority) {
				swap(idx, smallestChild);
				if (data[smallestChild].priority > data[parent(smallestChild)].priority) {
					swap(parent(smallestChild), smallestChild);
				}
				idx = smallestChild;
			}
			else {
				break;
			}
		}
		else {
			// smallest is a child
			if (data[smallestChild].priority < data[idx].priority) {
				swap(idx, smallestChild);
				idx = smallestChild;
			}
			break;
		}
	}
}

template <typename T, int MAX_Q>
__host__ __device__ void PriorityQueue<T, MAX_Q>::percolateDownMax(Q_Node<T> item, int idx) {
	// While node at idx has children
	while (childLeft(idx) < size) {
		// Find the smallest among children and grandchildren
		int descendents[] = {/*childLeft(idx), */childRight(idx),
			childLeft(childLeft(idx)), childRight(childLeft(idx)),
			childLeft(childRight(idx)), childRight(childRight(idx)) };
		int smallestChild = childLeft(idx);
		for (int desc_idx : descendents) {
			if (desc_idx < size && data[desc_idx].priority > data[smallestChild].priority) {
				smallestChild = desc_idx;
			}
		}

		if (smallestChild > childRight(idx)) {
			// smallest is a grandchild
			if (data[smallestChild].priority > data[idx].priority) {
				swap(idx, smallestChild);
				if (data[smallestChild].priority < data[parent(smallestChild)].priority) {
					swap(parent(smallestChild), smallestChild);
				}
				idx = smallestChild;
			}
			else {
				break;
			}
		}
		else {
			// smallest is a child
			if (data[smallestChild].priority > data[idx].priority) {
				swap(idx, smallestChild);
				idx = smallestChild;
			}
			break;
		}
	}
}
