#pragma once

#include "Cuda_KdTree.h"
#include "sort.h"
#include "dequeue.h"

#include "cub/cub.cuh"
#include <stdio.h>
#include <inttypes.h>

#include "CudaErrorCheck.cu"

#define EPSILON 0.01
#define MAX_KD_DIM 3

//#define _DEBUG_OUT

const int TPB = 256;
int SORT_TPB = 128;
const int POINT_RES = 1024; // Value to scale points by when converting to int

#define MAX_Q 32

using namespace texpert;

//--------------------------------------------------------------------------------------
// Cuda kernels

/**
Transforms each axis to the range [0,1]
*/
__global__ void transform_data(const MyPoint* data, uint32_t* __restrict__ data_out_x, uint32_t* __restrict__ data_out_y, uint32_t* __restrict__ data_out_z, int * __restrict__ index, const int size, const float* mins, const float* maxes)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= size) {
		return;
	}
	index[idx] = idx;
	float x_range = maxes[0] - mins[0];
	float y_range = maxes[1] - mins[1];
	float z_range = maxes[2] - mins[2];
	data_out_x[idx] = ((data[idx][0] - mins[0]) / x_range) * float(POINT_RES);
	data_out_y[idx] = ((data[idx][1] - mins[1]) / y_range) * float(POINT_RES);
	data_out_z[idx] = ((data[idx][2] - mins[2]) / z_range) * float(POINT_RES);

	// second half of the array keeps the data always in its original location
	data_out_x[size + idx] = data_out_x[idx];
	data_out_y[size + idx] = data_out_y[idx];
	data_out_z[size + idx] = data_out_z[idx];


}

/**
Splits the data into separate arrays
*/
__global__ void split_data(const MyPoint* data, float* __restrict__ data_out_x, float* __restrict__ data_out_y, float* __restrict__ data_out_z, const int size)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= size) {
		return;
	}
	data_out_x[idx] = data[idx][0];
	data_out_y[idx] = data[idx][1];
	data_out_z[idx] = data[idx][2];
}


/**
Reorganizes the remaing arrays
@param - data_in_x, data_in_y, data_in_z - the input data.
@param - data_out_x, data_out_y, data_out_z - the output data.
@param index, the array with all indicies.
@param dim, the dimension of the already sorted array. This array is skipped.
@param offset, an offset, if not all elements of the array should be sorted.
Note, if the number of elements is also limited, just start the right number of blocks / threads to make sure that only the required elements get sorted.
@param size, const int, the number of elements in the array. Since the data_in_'s keep the data in its original organization alive, size is used as a jump point
*/
__global__ void reorganize_all(uint32_t* data_in_x, uint32_t* data_in_y, uint32_t* data_in_z, uint32_t* data_out_x, uint32_t* data_out_y, uint32_t* data_out_z, int * index, int dim, const int N)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= N) { return; }
	int data_idx = index[idx];

	switch (dim)
	{
	case 0:
		data_out_y[idx] = data_in_y[N + data_idx];
		data_out_z[idx] = data_in_z[N + data_idx];
		break;
	case 1:
		data_out_x[idx] = data_in_x[N + data_idx];
		data_out_z[idx] = data_in_z[N + data_idx];
		break;
	case 2:
		data_out_x[idx] = data_in_x[N + data_idx];
		data_out_y[idx] = data_in_y[N + data_idx];
		break;
	}


}

__global__ void assign_chunks(int* segments, int width, int N) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= N) { return; }
	if ((idx + 1) % (width + 1) == 0) {
		segments[idx] = INT_MAX;
	}
	else {
		segments[idx] = (idx+1) / (width+1);
	}
}

__device__ void reorganize_func(int* data_in_x, int* data_in_y, int* data_in_z, int* data_out_x, int* data_out_y, int* data_out_z, int * index, int dim, int offset, const int size, int stop)
{

	for (int i = offset; i <= stop; i++) {

		int data_idx = index[i];

		switch (dim)
		{
		case 0:
			data_out_y[i] = data_in_y[size + data_idx];
			data_out_z[i] = data_in_z[size + data_idx];
			break;
		case 1:
			data_out_x[i] = data_in_x[size + data_idx];
			data_out_z[i] = data_in_z[size + data_idx];
			break;
		case 2:
			data_out_x[i] = data_in_x[size + data_idx];
			data_out_y[i] = data_in_y[size + data_idx];
			break;
		}
	}
	}



__global__ void create_nodes(int N, int n_chunks, uint32_t* data_x, uint32_t* data_y, uint32_t* data_z, float* mins, float* maxes, int dim, int* index, Cuda_KdNode* tree, int n_built, float chunk_width, bool final_level) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int tree_idx = idx + n_built;
	if (idx >= n_chunks || tree_idx >= MAX_NUM_POINTS) {return;}

	Cuda_KdNode* this_node = &tree[tree_idx];


	//int this_chunk_idx = 2*tree_idx;
	int l_bound = idx == 0 ? chunk_width * idx : (int)(chunk_width * idx) + 1;
	int r_bound = min(N, (int)(chunk_width * (idx + 1)));
	if (l_bound == r_bound) {
		return;
	}
	
	// Median. Computed in the same way as splitting points are elsewhere
	float next_chunk_width = chunk_width / 2;
	int median = floor(next_chunk_width * (2*idx + 1));

	int child_l_start = l_bound;
	int child_l_end = median;
	int child_r_start = median + 1;
	int child_r_end = r_bound;

	// Call constructor on existing Cuda_KdNode
	new (&tree[tree_idx]) Cuda_KdNode();
	this_node->_left = this_node->_right = NULL;

	// Fill node with data
	float x_range = maxes[0] - mins[0];
	float y_range = maxes[1] - mins[1];
	float z_range = maxes[2] - mins[2];
	// Scale back to real values
	this_node->_point[0] = (x_range * float(data_x[median]) / float(POINT_RES)) + mins[0];
	this_node->_point[1] = (y_range * float(data_y[median]) / float(POINT_RES)) + mins[1];
	this_node->_point[2] = (z_range * float(data_z[median]) / float(POINT_RES)) + mins[2];
	int point_id = index[median];
	
	this_node->_id = point_id;
	this_node->_point_index = point_id;
	this_node->_dim = dim;
	this_node->_point._id = point_id;

	if (child_l_start < child_l_end) {
		// Set pointer
		// TODO: Eliminate pointers
		int child_l_idx = tree_idx * 2 + 1;
		this_node->_left = &tree[child_l_idx];
	}
	// If there are more nodes to the right
	if (child_r_start < child_r_end) {
		// Set pointer
		// TODO: Eliminate pointers
		int child_r_idx = tree_idx * 2 + 2;
		this_node->_right = &tree[child_r_idx];
	}

}



/**
Prints the kd-tree into a terminal.
@param node - the root node of the tree
@param node_stack, memory for the breadth first search.
@param current_dimenstion, the start dimension / level of the tree
*/
__global__ void start_print(Cuda_KdNode* node, Cuda_KdNode** node_stack, int current_dimension, int _N)
{
	node_stack[0] = node;
	int stack_size = 1;
	int start = 0;
	int current_level = 0;

	int height = ceil(log2((double)_N+1));

	while (stack_size > 0)
	{

		current_level++;
		for (int i = 0; i < (1 << (height - current_level - 1)); i++) {
			printf("      ");
		}
		for (int i = 0; i < stack_size; i++)
		{
			Cuda_KdNode* current = node_stack[start + i];
			if (current != NULL) {
				//printf("  id:%d (%f, %f, %f)  ", current->_point_index, current->_point[0], current->_point[1], current->_point[2]);
				printf("  id:%d  ", current->_point_index);
			}
			else {
				printf("  NULL  ");
			}
			for (int j = 1; j < (1 << (height - current_level)); j++) {
				printf("      ");
			}
			//printf("\tid-le:%d\t\n", node_stack[start + i]->_left->_point_index);
			//printf("\tid-re:%d\t\n", node_stack[start + i]->_right->_point_index);
		}
		printf("\n");

		int new_stack_size = 0;
		for (int i = 0; i < stack_size; i++)
		{
			Cuda_KdNode* current = node_stack[start + i];
			if (current != NULL) {
				node_stack[start + stack_size + new_stack_size] = current->_left;
				new_stack_size++;
				node_stack[start + stack_size + new_stack_size] = current->_right;
				new_stack_size++;
			}
			//if (node_stack[start + i]->_left != NULL)
			//{
			//	node_stack[start + stack_size + new_stack_size] = node_stack[start + i]->_left;
			//	new_stack_size++;
			//}
			//if (node_stack[start + i]->_right != NULL)
			//{
			//	node_stack[start + stack_size + new_stack_size] = node_stack[start + i]->_right;
			//	new_stack_size++;
			//}
		}

		start += stack_size;
		stack_size = new_stack_size;
	}
}





//---------------------------------------------------------------------------------------------------------------------------------------------------------------------------
//---------------------------------------------------------------------------------------------------------------------------------------------------------------------------
// SEARCH


__device__ float dist2(const MyPoint a, const MyPoint b) {
	float sum = 0;
	for (int i = 0; i != MAX_KD_DIM; i++) {
		sum += (a[i] - b[i])*(a[i] - b[i]);
	}
	return sum;
}


/**
Searches the kd-tree for the nearest neighbors
@param root, the root node of the kd-tree
@param dev_search_points, an array with all device search points.
@param dev_matches, an array with all matches.
@param N, the total number of search points
*/
__global__ void knn_search(Cuda_KdNode* root, MyPoint* dev_search_points, MyMatches* dev_matches, const int N)
{
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	if (index >= N) {//out of range
		return;
	}

	MyPoint search_point = dev_search_points[index];
	//printf("search for point %d; %f, %f, %f\n", search_point._id, search_point[0], search_point[1], search_point[2]);

	PriorityQueue< Cuda_KdNode*, MAX_Q> q = PriorityQueue< Cuda_KdNode*, MAX_Q>();
	double nn_dist = INFINITY; // initial distance
	//int u_idx;
	Cuda_KdNode* u;
	Cuda_KdNode* u_match;
	q.insert(root, 0);
	//int u_match_idx = -1;
	double rd;  // distance to rectangle
	double old_off, new_rd;
	while (q.size != 0) {
		Q_Node< Cuda_KdNode*> closest = q.removeMin();
		rd = closest.priority;
		//u_idx = closest.idx; // closest node to query point
		//u = d_tree[u_idx];
		u = closest.node;
		//printf("Popped %d, prty: %f\n", u->_id, rd);
		if (rd >= nn_dist) { // further from nearest so far
			continue;
		}
		// TODO: Make handling nodes with a single child better
		while (u->_left != NULL || u->_right != NULL) { // descend until leaf found
			float new_dist = dist2(u->_point, search_point);
			//printf("distance for %d (%f, %f, %f):%f\n", u->_id, u->_point[0], u->_point[1], u->_point[2], new_dist);
			if (new_dist < nn_dist) {
				nn_dist = new_dist;
				u_match = u;
			}
			Cuda_KdNode* hi_child = u->_right;
			Cuda_KdNode* lo_child = u->_left;
			int cd = u->_dim; // cutting dimension
			float new_off = search_point[cd] - u->_point[cd]; // offset to further child
			new_rd = new_off*new_off;
			if (new_off < 0) { // search_point is below cutting plane
				if (new_rd < nn_dist && hi_child != NULL) {
					//printf("adding %d, oo:%f, no:%f, rd:%f\n", hi_child->_id, old_off, new_off, new_rd);
					q.insert(hi_child, new_rd); // enqueue hi_child for later
				}
				if (lo_child == NULL) {
					break;
				}
				u = lo_child;  // visit lo_child next
			}
			else { // q is above cutting plane
				if (new_rd < nn_dist && lo_child != NULL) {
					//printf("adding %d, oo:%f, no:%f, rd:%f\n", lo_child->_id, old_off, new_off, new_rd);
					q.insert(lo_child, new_rd); // enqueue lo_child for later
				}
				if (hi_child == NULL) {
					break;
				}
				u = hi_child;  // visit hi_child next
			}
			rd = new_rd;
		}
		float new_dist = dist2(u->_point, search_point);
		if (new_dist < nn_dist) {
			nn_dist = new_dist;
			u_match = u;
		}
	}
	dev_matches[index].matches[0].first = index;
	dev_matches[index].matches[0].second = u_match->_point._id;
	//dev_matches[index].matches[0].second = u_match->_point;
	dev_matches[index].matches[0].distance = nn_dist;

}


/**
Search for the largest value in dev_matches
@param dev_matches - input array with all found matches. 
@param k - number of total elements to consider. 
@param index - pointer to return the index
@parma value - pointer to return the value
*/
__device__ void find_largest_value(MyMatch* dev_matches, const int k, int* index, double* value)
{
	float largest_v = 0;
	int   largest_idx = -1;
	for (int i = 0; i < k; i++)
	{
		if (dev_matches[i].distance > largest_v)
		{
			largest_v = dev_matches[i].distance;
			largest_idx = i;
		}
	}

	(*index) = largest_idx;
	(*value) = largest_v;

}


/**
Searches the kd-tree for the nearest neighbors
@param root, the root node of the kd-tree
@param dev_search_points, an array with all device search points.
@param dev_matches, an array with all matches.
@param N, the total number of search points
param k, the number of nearest neighbors to find. 
*/
__global__ void knn_search(Cuda_KdNode* root, MyPoint* dev_search_points, MyMatches* dev_matches, const int N, const int k)
{
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	if (index >= N) {//out of range
		return;
	}

	int	output_index = 0;
	int stored_outputs = 0;

	MyPoint search_point = dev_search_points[index];
	//printf("search for point %d; %f, %f, %f\n", search_point._id, search_point[0], search_point[1], search_point[2]);

	PriorityQueue< Cuda_KdNode*, MAX_Q> q = PriorityQueue< Cuda_KdNode*, MAX_Q>();
	double nn_dist = INFINITY; // initial distance
	double nn_largest_dist = 0; // stores the largest value
	int nn_largest_idx = -1;
	//int u_idx;


	Cuda_KdNode* u;
	Cuda_KdNode* u_match;
	q.insert(root, 0);
	//int u_match_idx = -1;
	double rd;  // distance to rectangle
	double old_off, new_rd;
	while (q.size != 0) {
		Q_Node<Cuda_KdNode*> closest = q.removeMin();
		rd = closest.priority;
		//u_idx = closest.idx; // closest node to query point
		//u = d_tree[u_idx];
		u = closest.node;

		//printf("Popped %d, prty: %f\n", u->_id, rd);
		if (rd >= nn_largest_dist && stored_outputs == k) { // further from largest k and the list is full
			continue;
		}

		// TODO: Make handling nodes with a single child better
		while (u->_left != NULL || u->_right != NULL) { // descend until leaf found
			float new_dist = dist2(u->_point, search_point);
			//printf("distance for %d (%f, %f, %f):%f\n", u->_id, u->_point[0], u->_point[1], u->_point[2], new_dist);

			// just store
			if (stored_outputs < k) {

				// store node
				nn_dist = new_dist;
				u_match = u;

				dev_matches[index].matches[output_index].first = index;
				dev_matches[index].matches[output_index].second = u_match->_point._id;
				//dev_matches[index].matches[output_index].second = u_match->_point;
				dev_matches[index].matches[output_index].distance = nn_dist;

				if (new_dist > nn_largest_dist)
				{
					nn_largest_dist = new_dist;
					nn_largest_idx = output_index;
				}
				stored_outputs++;
				output_index++;

			}
			else if (new_dist < nn_largest_dist) { // replace largest value

				nn_dist = new_dist;
				u_match = u;

				dev_matches[index].matches[nn_largest_idx].first = index;
				dev_matches[index].matches[nn_largest_idx].second = u_match->_point._id;
				//dev_matches[index].matches[nn_largest_idx].second = u_match->_point;
				dev_matches[index].matches[nn_largest_idx].distance = nn_dist;

				// find new largest index and value
				find_largest_value(&dev_matches[index].matches[0], stored_outputs, &nn_largest_idx, &nn_largest_dist);
			}



			Cuda_KdNode* hi_child = u->_right;
			Cuda_KdNode* lo_child = u->_left;
			int cd = u->_dim; // cutting dimension
			float new_off = search_point[cd] - u->_point[cd]; // offset to further child
			new_rd = new_off*new_off;
			if (new_off < 0) { // search_point is below cutting plane
				if (new_rd < nn_dist && hi_child != NULL) {
					//printf("adding %d, oo:%f, no:%f, rd:%f\n", hi_child->_id, old_off, new_off, new_rd);
					q.insert(hi_child, new_rd); // enqueue hi_child for later
				}
				if (lo_child == NULL) {
					break;
				}
				u = lo_child;  // visit lo_child next
			}
			else { // q is above cutting plane
				if (new_rd < nn_dist && lo_child != NULL) {
					//printf("adding %d, oo:%f, no:%f, rd:%f\n", lo_child->_id, old_off, new_off, new_rd);
					q.insert(lo_child, new_rd); // enqueue lo_child for later
				}
				if (hi_child == NULL) {
					break;
				}
				u = hi_child;  // visit hi_child next
			}
			rd = new_rd;
		}
		float new_dist = dist2(u->_point, search_point);

		
		// just store
		if (stored_outputs < k) {

			// store node
			nn_dist = new_dist;
			u_match = u;

			dev_matches[index].matches[output_index].first = index;
			dev_matches[index].matches[output_index].second = u_match->_point._id;
			//dev_matches[index].matches[output_index].second = u_match->_point;
			dev_matches[index].matches[output_index].distance = nn_dist;

			if (new_dist > nn_largest_dist)
			{
				nn_largest_dist = new_dist;
				nn_largest_idx = output_index;
			}
			stored_outputs++;
			output_index++;

		}
		else if (new_dist < nn_largest_dist) { // replace largest value

			nn_dist = new_dist;
			u_match = u;

			dev_matches[index].matches[nn_largest_idx].first = index;
			dev_matches[index].matches[nn_largest_idx].second = u_match->_point._id;
			//dev_matches[index].matches[nn_largest_idx].second = u_match->_point;
			dev_matches[index].matches[nn_largest_idx].distance = nn_dist;

			// find new largest index and value
			find_largest_value(&dev_matches[index].matches[0], stored_outputs, &nn_largest_idx, &nn_largest_dist);
		}


		/*if (new_dist < nn_dist) {
			nn_dist = new_dist;
			u_match = u;
		}*/
	}
	//dev_matches[index].matches[output_index].first = index;
	//dev_matches[index].matches[output_index].second = u_match->_point;
	//dev_matches[index].matches[output_index].distance = nn_dist;

}





/**
Searches the kd-tree for the nearest neighbors that are closer than a given radius.
It finds the KNN_MATCHES_LENGTH closest one due to the fixed length of MyMatches.
@param root, the root node of the kd-tree
@param dev_search_points, an array with all device search points.
@param dev_matches, an array with all matches.
@param N, the total number of search points
param k, the number of nearest neighbors to find.
*/
__global__ void knn_radius_search(Cuda_KdNode* root, MyPoint* dev_search_points, MyMatches* dev_matches, const int N, const double radius)
{
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	if (index >= N) {//out of range
		return;
	}

	double radius2 = radius * radius;
	int	output_index = 0;
	int stored_outputs = 0;

	MyPoint search_point = dev_search_points[index];
	//printf("search for point %d; %f, %f, %f\n", search_point._id, search_point[0], search_point[1], search_point[2]);

	PriorityQueue< Cuda_KdNode*, MAX_Q> q = PriorityQueue< Cuda_KdNode*, MAX_Q>();
	double nn_dist = INFINITY; // initial distance
	double nn_largest_dist = 0; // stores the largest value
	int nn_largest_idx = -1;
	//int u_idx;


	Cuda_KdNode* u;
	Cuda_KdNode* u_match;
	q.insert(root, 0);
	//int u_match_idx = -1;
	double rd;  // distance to rectangle
	double old_off, new_rd;
	while (q.size != 0) {
		Q_Node< Cuda_KdNode*> closest = q.removeMin();
		rd = closest.priority;
		//u_idx = closest.idx; // closest node to query point
		//u = d_tree[u_idx];
		u = closest.node;

		//printf("Popped %d, prty: %f\n", u->_id, rd);
		if (rd >= nn_largest_dist && stored_outputs == KNN_MATCHES_LENGTH) { // further from largest k and the list is full
			continue;
		}

		// TODO: Make handling nodes with a single child better
		while (u->_left != NULL || u->_right != NULL) { // descend until leaf found
			float new_dist = dist2(u->_point, search_point);
			//printf("distance for %d (%f, %f, %f):%f\n", u->_id, u->_point[0], u->_point[1], u->_point[2], new_dist);

			// just store
			if (stored_outputs < KNN_MATCHES_LENGTH && new_dist <= radius2) {

				// store node
				nn_dist = new_dist;
				u_match = u;

				dev_matches[index].matches[output_index].first = index;
				dev_matches[index].matches[output_index].second = u_match->_point._id;
				//dev_matches[index].matches[output_index].second = u_match->_point;
				dev_matches[index].matches[output_index].distance = nn_dist;

				if (new_dist > nn_largest_dist)
				{
					nn_largest_dist = new_dist;
					nn_largest_idx = output_index;
				}
				stored_outputs++;
				output_index++;

			}
			else if (new_dist < nn_largest_dist && new_dist <= radius2) { // replace largest value

				nn_dist = new_dist;
				u_match = u;

				dev_matches[index].matches[nn_largest_idx].first = index;
				dev_matches[index].matches[nn_largest_idx].second = u_match->_point._id;
				//dev_matches[index].matches[nn_largest_idx].second = u_match->_point;
				dev_matches[index].matches[nn_largest_idx].distance = nn_dist;

				// find new largest index and value
				find_largest_value(&dev_matches[index].matches[0], stored_outputs, &nn_largest_idx, &nn_largest_dist);
			}



			Cuda_KdNode* hi_child = u->_right;
			Cuda_KdNode* lo_child = u->_left;
			int cd = u->_dim; // cutting dimension
			float new_off = search_point[cd] - u->_point[cd]; // offset to further child
			new_rd = new_off*new_off;
			if (new_off < 0) { // search_point is below cutting plane
				if (new_rd < nn_dist && hi_child != NULL) {
					//printf("adding %d, oo:%f, no:%f, rd:%f\n", hi_child->_id, old_off, new_off, new_rd);
					q.insert(hi_child, new_rd); // enqueue hi_child for later
				}
				if (lo_child == NULL) {
					break;
				}
				u = lo_child;  // visit lo_child next
			}
			else { // q is above cutting plane
				if (new_rd < nn_dist && lo_child != NULL) {
					//printf("adding %d, oo:%f, no:%f, rd:%f\n", lo_child->_id, old_off, new_off, new_rd);
					q.insert(lo_child, new_rd); // enqueue lo_child for later
				}
				if (hi_child == NULL) {
					break;
				}
				u = hi_child;  // visit hi_child next
			}
			rd = new_rd;
		}
		float new_dist = dist2(u->_point, search_point);


		// just store
		if (stored_outputs < KNN_MATCHES_LENGTH && new_dist <= radius2) {

			// store node
			nn_dist = new_dist;
			u_match = u;

			dev_matches[index].matches[output_index].first = index;
			dev_matches[index].matches[output_index].second = u_match->_point._id;
			//dev_matches[index].matches[output_index].second = u_match->_point;
			dev_matches[index].matches[output_index].distance = nn_dist;

			if (new_dist > nn_largest_dist)
			{
				nn_largest_dist = new_dist;
				nn_largest_idx = output_index;
			}
			stored_outputs++;
			output_index++;

		}
		else if (new_dist < nn_largest_dist && new_dist <= radius2 ) { // replace largest value

			nn_dist = new_dist;
			u_match = u;

			dev_matches[index].matches[nn_largest_idx].first = index;
			dev_matches[index].matches[nn_largest_idx].second = u_match->_point._id;
			//dev_matches[index].matches[nn_largest_idx].second = u_match->_point;
			dev_matches[index].matches[nn_largest_idx].distance = nn_dist;

			// find new largest index and value
			find_largest_value(&dev_matches[index].matches[0], stored_outputs, &nn_largest_idx, &nn_largest_dist);
		}


		/*if (new_dist < nn_dist) {
		nn_dist = new_dist;
		u_match = u;
		}*/
	}
	//dev_matches[index].matches[output_index].first = index;
	//dev_matches[index].matches[output_index].second = u_match->_point;
	//dev_matches[index].matches[output_index].distance = nn_dist;

}









//---------------------------------------------------------------------------------------------------------------------------------------------------------------------------
//---------------------------------------------------------------------------------------------------------------------------------------------------------------------------
// Class members Cuda_KdNode

Cuda_KdNode::Cuda_KdNode()
{
	_left = NULL;
	_right = NULL;
	_id = -1;
	_N = 1;
	_dim = -1;
	_point_index = -1;



}


Cuda_KdNode::~Cuda_KdNode()
{

	if (_left != NULL) delete _left;
	if (_right != NULL) delete _right;
	
}


/**
Return the size in the tree
*/
int& Cuda_KdNode::size(void)
{
	return _N;
}


//--------------------------------------------------------------------------------------
// Class members Cuda_KdTree




Cuda_KdTree::Cuda_KdTree()
{
	_host_root = NULL;
	_d_tree = NULL;
	_N = 0;
	
	sorter = new ChunkedSorter(MAX_NUM_POINTS, 1 << MIN_SORT_BASE_EXP);
	// allocate memory
	allocateMemory();
}


Cuda_KdTree::~Cuda_KdTree()
{
	CudaSafeCall(cudaFree(_dev_data_arr));
	CudaSafeCall(cudaFree(d_limits_tmp));
	CudaSafeCall(cudaFree(d_maxes));
	CudaSafeCall(cudaFree(d_mins));
	CudaSafeCall(cudaFree(_d_tree));
	CudaSafeCall(cudaFree(_dev_data_arr_x));
	CudaSafeCall(cudaFree(_dev_data_arr_y));
	CudaSafeCall(cudaFree(_dev_data_arr_z));
	CudaSafeCall(cudaFree(_dev_temp_memory));
	CudaSafeCall(cudaFree(_dev_index));
	cudaStreamDestroy(streams[0]);
	cudaStreamDestroy(streams[1]);
	cudaStreamDestroy(streams[2]);
	delete sorter;
	
#ifdef _DEBUG
	free(_host_data_arr_x);
	free(_host_data_arr_y);
	free(_host_data_arr_z);
	free(_host_index);
#endif
}


/**
Create the kd-tree
@param points, a vector with all points
@return, the number of points that have been added to the tree
*/
void Cuda_KdTree::initialize(vector<MyPoint>& points) {
	if (SORT_TPB < 16) SORT_TPB = 16;
	return initialize_full(points, SORT_TPB, 5, SORT_TPB);
}

struct sort_functor {
	thrust::device_ptr<int> arr_ptr;
	thrust::device_ptr<int> index_ptr;
	int width;
	int N;
	//template <typename Tuple>
	__host__ __device__
	void operator()(int i) {
		//int i = thrust::get<0>(t);
		int l_bound = i * (width + 1);
		int r_bound = (i + 1) * (width + 1) - 1;
		r_bound = min(r_bound, N);
		thrust::sort_by_key(thrust::device, arr_ptr + l_bound, arr_ptr + r_bound, index_ptr + l_bound);
	}
};

void Cuda_KdTree::initialize_full(vector<MyPoint>& points, int sort_tpb, int sort_base_exp, int sort_cutoff)
{



	_data = points;
	_N = _data.size();
	assert(_N <= MAX_NUM_POINTS);
	//struct cudaDeviceProp info;
	//cudaGetDeviceProperties(&info, 0);
	//printf("Shader clock freq: %d\n", info.clockRate);
	int needed_depth = log2(_N);
	// copy data to the device:
	Cuda_Helpers::HostToDevice<MyPoint>(&_data[0], _dev_data_arr, _data.size());
	CudaCheckError();

	// split the data into single arrays
	split_data << < (_N + TPB - 1) / TPB, TPB >> >(_dev_data_arr, (float*)_dev_data_arr_x, (float*)_dev_data_arr_y, (float*)_dev_data_arr_z, _N);
	cudaDeviceSynchronize();
	CudaCheckError();



	size_t orig_limits_tmp_size = limits_tmp_size; // In case cub::max modifies the value, would mess up the offset on d_limits_tmp
	// Launch all the min/max reductions
	cub::DeviceReduce::Max((char*)(d_limits_tmp) + (0 * limits_tmp_size), orig_limits_tmp_size, (float*)_dev_data_arr_x, &(d_maxes[0]), _N, streams[0]);
	cub::DeviceReduce::Max((char*)(d_limits_tmp) + (1 * limits_tmp_size), orig_limits_tmp_size, (float*)_dev_data_arr_y, &(d_maxes[1]), _N, streams[1]);
	cub::DeviceReduce::Max((char*)(d_limits_tmp) + (2 * limits_tmp_size), orig_limits_tmp_size, (float*)_dev_data_arr_z, &(d_maxes[2]), _N, streams[2]);
	cub::DeviceReduce::Min((char*)(d_limits_tmp) + (0 * limits_tmp_size), orig_limits_tmp_size, (float*)_dev_data_arr_x, &(d_mins[0]),  _N, streams[0]);
	cub::DeviceReduce::Min((char*)(d_limits_tmp) + (1 * limits_tmp_size), orig_limits_tmp_size, (float*)_dev_data_arr_y, &(d_mins[1]),  _N, streams[1]);
	cub::DeviceReduce::Min((char*)(d_limits_tmp) + (2 * limits_tmp_size), orig_limits_tmp_size, (float*)_dev_data_arr_z, &(d_mins[2]),  _N, streams[2]);
	cudaDeviceSynchronize();
	CudaCheckError();

	uint32_t* x_transformed = (uint32_t*)_dev_data_arr_x;
	uint32_t* y_transformed = (uint32_t*)_dev_data_arr_y;
	uint32_t* z_transformed = (uint32_t*)_dev_data_arr_z;
	transform_data << < (_N + TPB - 1) / TPB, TPB >> >(_dev_data_arr, x_transformed, y_transformed, z_transformed, _dev_index, _N, d_mins, d_maxes);
	CudaCheckError();



#ifdef T_DEBUG
	using   debug_type = uint32_t;

	debug_type* _host_data_x = (debug_type*)malloc(MAX_NUM_POINTS * sizeof(debug_type));
	debug_type* _host_data_y = (debug_type*)malloc(MAX_NUM_POINTS * sizeof(debug_type));
	debug_type* _host_data_z = (debug_type*)malloc(MAX_NUM_POINTS * sizeof(debug_type));
	int*	_host_index = (int*)malloc(MAX_NUM_POINTS * sizeof(int));


	Cuda_Helpers::DeviceToHost<debug_type>((debug_type*)x_transformed, _host_data_x, _N);
	Cuda_Helpers::DeviceToHost<debug_type>((debug_type*)y_transformed, _host_data_y, _N);
	Cuda_Helpers::DeviceToHost<debug_type>((debug_type*)z_transformed, _host_data_z, _N);
	Cuda_Helpers::DeviceToHost<int>((int*)_dev_index, _host_index, _N);

	for (int i = 0; i<_N; i++)
	{
		//printf("%" PRIu32  PRIu32  PRIu32 "\n", _host_data_x[i], _host_data_y[i], _host_data_z[i]);
		cout << _host_index[i] << " : " <<(float)_host_data_x[i] << " : " << (float)_host_data_y[i] << " : " << (float)_host_data_z[i]  << endl;
	}

#endif



	sorter->setBase(sort_base_exp);

	int cur_level = 0;
	while (cur_level <= needed_depth) {
		int dim = cur_level % 3;
		uint32_t* d_arr;
		switch (dim) {
			case 0:
				d_arr = x_transformed;
				break;
			case 1:
				d_arr = y_transformed;
				break;
			case 2:
				d_arr = z_transformed;
				break;
		}

		int n_built = (1 << cur_level) - 1;
		int n_chunks = 1 << cur_level;
		double chunk_width = (double)_N / (1 << cur_level);





		// Sort
		if (chunk_width >= sort_cutoff) {
			sorter->sort(d_arr, _dev_index, _dev_temp_memory, _N, cur_level, POINT_RES, sort_tpb);
		}
		else {
			if (chunk_width < 16) {
				serial_insertionsort << <(n_chunks + TPB - 1) / TPB, TPB >> > (d_arr, POINT_RES, n_chunks, chunk_width, _dev_index, _N);
			}
			else {
				serial_radixsort << <(n_chunks + TPB - 1) / TPB, TPB >> > (d_arr, POINT_RES, n_chunks, chunk_width, _dev_index, _dev_temp_memory, _N);
			}
			//cudaDeviceSynchronize();
			CudaCheckError();
		}
	

		// Reorganize all points
		reorganize_all<<<(_N + TPB - 1) / TPB, TPB>>> (x_transformed, y_transformed, z_transformed, x_transformed, y_transformed, z_transformed, _dev_index, dim, _N);
		//cudaDeviceSynchronize();
		CudaCheckError();

		// create nodes
		create_nodes<<<(n_chunks + TPB - 1) / TPB, TPB>>> (_N, n_chunks, x_transformed, y_transformed, z_transformed, d_mins, d_maxes, dim, _dev_index, _d_tree, n_built, chunk_width, cur_level == needed_depth);
		CudaCheckError();
		//cudaDeviceSynchronize();

		cur_level++;
	}
	cudaDeviceSynchronize();



#ifdef T_DEBUG
	Cuda_Helpers::DeviceToHost<int>((int*)_dev_data_arr_x, _host_data_arr_x, _N);
	Cuda_Helpers::DeviceToHost<int>((int*)_dev_data_arr_y, _host_data_arr_y, _N);
	Cuda_Helpers::DeviceToHost<int>((int*)_dev_data_arr_z, _host_data_arr_z, _N);
	Cuda_Helpers::DeviceToHost<int>(_dev_index, _host_index, _N);

	
	for(int i=0; i<_N; i++)
	{
		cout << _host_data_arr_x[i] << " : " << _host_data_arr_y[i] << " : " << _host_data_arr_z[i] << endl;
	}

#endif

}


/**
Clears the tree memory.
*/
bool Cuda_KdTree::resetDevTree(void)
{
	cudaError_t cudaStatus = cudaFree((void**)_d_tree);CudaCheckError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaFree failed!");
		return false;
	}

	// Allocate memory
	cudaStatus = cudaMalloc((void**)&_d_tree, 1 * sizeof(Cuda_KdNode));CudaCheckError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		return false;
	}

	// resetting the node with new data. 
	if (_host_root) {
		_host_root->_dim = -1;
		_host_root->_id = -1;
		_host_root->_left = NULL;
		_host_root->_right = NULL;
	}


	return true;
}



/**
Prints the tree structure
*/
void Cuda_KdTree::print(void)
{

	// search stack for breadth first search.
	Cuda_KdNode** node_stack;
	size_t stack_size = 2*pow(2, ceil(log2(_N)));
	cudaMalloc((void**)&node_stack, stack_size * sizeof(Cuda_KdNode)); CudaCheckError();

	cout << "----- Start print tree -----" << endl;

	// start the kernel
	start_print << <1, 1 >> >(_d_tree, node_stack, 0, _N);

	cudaDeviceSynchronize();


	cout << "----- Stop print tree -----\n" << endl;


	//cudaFree((void**)&node_stack); CudaCheckError();
}

/**
Returns the number of nodes in this tree.
*/
int Cuda_KdTree::size(void)
{
	return _N;
}



/**
Allocate memory
*/
void Cuda_KdTree::allocateMemory(void)
{

	size_t size = _data.size();
	cudaError_t cudaStatus;

	// Allocate memory
	cudaStatus = cudaMalloc((void**)&_dev_data_arr, MAX_NUM_POINTS * sizeof(MyPoint));CudaCheckError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		return;
	}

	// Memory for storing the maximum and minimum values of the input, per dimension
	cudaMalloc(&d_maxes, 3 * sizeof(float));
	cudaMalloc(&d_mins, 3 * sizeof(float));
	CudaCheckError();

	// Memory for CUB min/max reduce
	cub::DeviceReduce::Max(d_limits_tmp, limits_tmp_size, d_maxes, d_maxes, MAX_NUM_POINTS);
	size_t for_max = limits_tmp_size;
	cub::DeviceReduce::Min(d_limits_tmp, limits_tmp_size, d_maxes, d_maxes, MAX_NUM_POINTS);
	limits_tmp_size = limits_tmp_size > for_max ? limits_tmp_size : for_max; // max(size_for_max, size_for_min)
	cudaMalloc((void**)&d_limits_tmp, 3 * limits_tmp_size);
	CudaCheckError();

	// Need to store a complete level, so store 2 * 2^(floor(log2(MAX_POINTS)))
	size_t tree_size = 2 * (1 << ((int) log2(MAX_NUM_POINTS))) * sizeof(Cuda_KdNode);
	cudaStatus = cudaMalloc((void**)&_d_tree, tree_size);CudaCheckError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		return;
	}


	//------------------------------------------
	// Memory for the sort process

	cudaStatus = cudaMalloc((void**)&_dev_data_arr_x, 2 * MAX_NUM_POINTS * sizeof(int));CudaCheckError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		return;
	}
	cudaStatus = cudaMalloc((void**)&_dev_data_arr_y, 2 * MAX_NUM_POINTS * sizeof(int));CudaCheckError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		return;
	}
	cudaStatus = cudaMalloc((void**)&_dev_data_arr_z, 2 * MAX_NUM_POINTS * sizeof(int));CudaCheckError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		return;
	}
	cudaStatus = cudaMalloc((void**)&_dev_temp_memory, 3 * MAX_NUM_POINTS * sizeof(int));CudaCheckError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		return;
	}
	cudaStatus = cudaMalloc((void**)&_dev_index, MAX_NUM_POINTS * sizeof(int));CudaCheckError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		return;
	}

	// Memory for searching
	cudaStatus = cudaMalloc((void**)&d_query_points, MAX_SEARCH_POINTS * sizeof(MyPoint));CudaCheckError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		return;
	}

	cudaStatus = cudaMalloc((void**)&d_query_results, MAX_OUTPUT_POINTS * sizeof(MyMatches));
	CudaCheckError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		return;
	}


	// Create streams
	cudaStreamCreate(&streams[0]);
	cudaStreamCreate(&streams[1]);
	cudaStreamCreate(&streams[2]);

#ifdef _DEBUG
	_host_data_arr_x = (int*)malloc(MAX_NUM_POINTS * sizeof(int));
	_host_data_arr_y = (int*)malloc(MAX_NUM_POINTS * sizeof(int));
	_host_data_arr_z = (int*)malloc(MAX_NUM_POINTS * sizeof(int));
	_host_index = (int*)malloc(MAX_NUM_POINTS * sizeof(int));
#endif



}



/**
Nearset neighbor search.
This function looks only for ONE nearest neighbor in the tree.
@param search_points, vector with the search points
@param matches, vector with the matches.
*/
void Cuda_KdTree::knn(std::vector<MyPoint>& search_points, std::vector<MyMatches>& output)
{
	size_t n = search_points.size();
	assert(n <= MAX_SEARCH_POINTS);
	output.resize(n);

	Cuda_Helpers::HostToDevice<MyPoint>(&search_points.front(), d_query_points, n);  CudaCheckError();

	// start the search
	knn_search << < (n + TPB - 1) / TPB, TPB >> > (_d_tree, d_query_points, d_query_results, n);
	cudaDeviceSynchronize();
	CudaCheckError();

	Cuda_Helpers::DeviceToHost<MyMatches>(d_query_results, &output.front(), n);
	CudaCheckError();
}



/*
Searches for k nearest neighbors.
@param search_points, vector with the search points
@param matches, vector with the matches.
@param k - the number of neaarest neighbors to be fouund.
*/
void Cuda_KdTree::knn(std::vector<MyPoint>& search_points, std::vector<MyMatches>& output, int k)
{

	size_t n = search_points.size();
	assert(n <= MAX_SEARCH_POINTS);
	assert(n  <= MAX_OUTPUT_POINTS);
	assert(k <= KNN_MATCHES_LENGTH);
	output.resize(n);

	Cuda_Helpers::HostToDevice<MyPoint>(&search_points.front(), d_query_points, n); 
	CudaCheckError();

	// start the search
	knn_search << < (n + TPB - 1) / TPB, TPB >> > (_d_tree, d_query_points, d_query_results, n, k);
	cudaDeviceSynchronize();
	CudaCheckError();

	Cuda_Helpers::DeviceToHost<MyMatches>(d_query_results, &output.front(),  n);
	CudaCheckError();

}



/*
Searches for the points within a given radius. Note that the output memory is constant (check in types.h).
The function can only return the KNN_MATCHES_LENGTH closests matches to the search point.
@param search_points, vector with the search points
@param matches, vector with the matches.
@param radius, the maximum search radius.
*/
void Cuda_KdTree::radius_search(std::vector<MyPoint>& search_points, std::vector<MyMatches>& output, double radius)
{
	size_t n = search_points.size();
	assert(n <= MAX_SEARCH_POINTS);
	assert(n <= MAX_OUTPUT_POINTS);
	output.resize(n);

	Cuda_Helpers::HostToDevice<MyPoint>(&search_points.front(), d_query_points, n);  CudaCheckError();

	// start the search
	knn_radius_search << < (n + TPB - 1) / TPB, TPB >> > (_d_tree, d_query_points, d_query_results, n, radius);
	cudaDeviceSynchronize();
	CudaCheckError();

	Cuda_Helpers::DeviceToHost<MyMatches>(d_query_results, &output.front(), n);
	CudaCheckError();
}



