#pragma once
#include "dequeue.h"

/*
////////////////////////////////////////////////////////////////////////
--------------------Priority Queue Implementation-----------------------
////////////////////////////////////////////////////////////////////////
*/

__host__ __device__ int int_log2(int x) {
	int ret = 0;
	while (x >>= 1) ++ret;
	return ret;
}

__host__ __device__ void PriorityQueue::insert(Cuda_KdNode* node, float weight) {
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
	data[idx] = { node, weight };
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

__host__ __device__ void PriorityQueue::swap(int idx1, int idx2) {
	Q_Node original_idx1_val = data[idx1];
	data[idx1] = data[idx2];
	data[idx2] = original_idx1_val;
}

__host__ __device__ void PriorityQueue::bubbleUp(int idx) {
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

__host__ __device__ void PriorityQueue::bubbleUpMin(int idx) {
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

__host__ __device__ void PriorityQueue::bubbleUpMax(int idx) {
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

__host__ __device__ Q_Node PriorityQueue::removeMin() {
	//assert(size > 0); // Canot remove from empty queue
	Q_Node to_return = data[0];
	size--;
	data[0] = data[size];
	percolateDown(data[size], 0);
	return to_return;
}

__host__ __device__ int PriorityQueue::largest_idx() {
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

__host__ __device__ Q_Node PriorityQueue::removeMax() {
	int largest = largest_idx();
	Q_Node to_return = data[largest];
	size--;
	data[largest] = data[size];
	percolateDown(data[size], largest);
	return to_return;
}

__host__ __device__ Q_Node PriorityQueue::peekMax() {
	int largest = largest_idx();
	return data[largest];
}

__host__ __device__ void PriorityQueue::percolateDown(Q_Node item, int idx) {
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

__host__ __device__ void PriorityQueue::percolateDownMin(Q_Node item, int idx) {
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
__host__ __device__ void PriorityQueue::percolateDownMax(Q_Node item, int idx) {
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