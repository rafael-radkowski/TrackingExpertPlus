#pragma once
/*
class Cuda_KdTree

Rafael Radkowski
Iowa State University
rafael@iastate.edu
MIT License
---------------------------------------------------------------
*/


// stl
#include <stdio.h>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <algorithm>
#include <chrono>

// cuda
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <thrust/sort.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/system/cuda/execution_policy.h>
#include <thrust/extrema.h>

// local
#include "Cuda_Helpers.h"
#include "Cuda_Common.h"
#include "Cuda_Types.h"

#define _WITH_PERFORMANCE

#define MAX_NUM_POINTS 800000
#define MAX_SEARCH_POINTS 100000
#define MAX_OUTPUT_POINTS 450000
#define RELOC_TPB 64
using namespace std;
using namespace std::chrono;


// Forward declare ChunkedSorter so sort.h can include CUB library
class ChunkedSorter;

//--------------------------------------------------------------------------------------
// KD-tree node


class Cuda_KdNode
{
public:

	CUDA_MEMBER Cuda_KdNode();
	CUDA_MEMBER ~Cuda_KdNode();

	/**
	Return the size in the tree
	*/
	CUDA_MEMBER int& size(void);


	//-----------------------------------------
	// Data

	// child nodes
	Cuda_KdNode*			_left;
	Cuda_KdNode*			_right;

	// node id >= 0; -1 - no id assigned
	int						_id;

	// separator dimension
	// both variables are necessary for construction. 
	int						_dim;
	int						_point_index; // This is the node id. Node id and point_index got mixed up


	// the point that is stored in this node
	MyPoint					_point;

private:

	// size;
	int						_N;


};



//--------------------------------------------------------------------------------------
// KD-tree 

class Cuda_KdTree // : public mycv::KdTree<mycv::dPoint>
{
public:



	Cuda_KdTree();
	~Cuda_KdTree();

	
	/**
	Create the kd-tree
	@param points, a vector with all points
	@return, the number of points that have been added to the tree
	*/
	void initialize(std::vector<MyPoint>& points);
	void initialize_full(vector<MyPoint>& points, int sort_tpb, int sort_base_exp, int cutoff);



	/**
	Clears the tree memory and create a new tree root node.
	This is necessary, if the tree should be re-used with new data. 
	It cleans the device, temporary data. 
	*/
	bool resetDevTree(void);

	/**
	Prints the tree structure
	*/
	void print(void);

	/**
	Returns the number of nodes in this tree. 
	*/
	int size(void);

	/**
	Nearset neighbor search. 
	This function looks only for ONE nearest neighbor in the tree. 
	@param search_points, vector with the search points
	@param matches, vector with the matches. 
	*/
	void knn(std::vector<MyPoint>& search_points, std::vector<MyMatches>& output);


	/*
	Searches for k nearest neighbors. 
	@param search_points, vector with the search points
	@param matches, vector with the matches.
	@param k - the number of neaarest neighbors to be fouund. 
	*/
	void knn(std::vector<MyPoint>& search_points, std::vector<MyMatches>& output, int k);


	/*
	Searches for the points within a given radius. Note that the output memory is constant (check in types.h).
	The function can only return the KNN_MATCHES_LENGTH closests matches to the search point. 
	@param search_points, vector with the search points
	@param matches, vector with the matches.
	@param radius, the maximum search radius. 
	*/
	void radius_search(std::vector<MyPoint>& search_points, std::vector<MyMatches>& output, double radius);

private:
	/**
	Allocate memory
	*/
	void allocateMemory(void);

	ChunkedSorter* sorter;

	// Streams for running multiple kernels in parallel. Online created once, then reused
	cudaStream_t streams[3];

	//----------------------------
	// Host data

	// root node of the tree
	Cuda_KdNode*		_host_root;

	// the kd-tree data
	vector<MyPoint>	_data;

	// the id's for each data point
	vector<MyPoint>	_data_index;

	// The [start, end) bounds for each node in the tree. Inclusive start, exclusive end
	//int* _d_chunk_bounds;

	// Array to hold the entire tree
	Cuda_KdNode* _d_tree;

	// number of points in the tree
	int					_N;

#ifdef _DEBUG
	int*				_host_data_arr_x;
	int*				_host_data_arr_y;
	int*				_host_data_arr_z;



	// the index array for this data 
	int*				_host_index;
#endif



private:

	//----------------------------
	// Device data

	// the data array, pointer to the first element
	MyPoint*			_dev_data_arr;

	// array for data of 1 dimension, after cuda split the data into arrays
	// Large enough to hold MAX_NUM_POINTS of 4 bytes each (Used for both floats and ints)
	void*			_dev_data_arr_x;
	void*			_dev_data_arr_y;
	void*			_dev_data_arr_z;

	// temporary memory to sort the arrays. 
	std::uint32_t*	_dev_temp_memory;

	// the index array for this data 
	int*				_dev_index;

	// Size of temporary memory needed for a single max or min scan
	size_t limits_tmp_size = 0;
	// Temp storage for CUB scan; Of size (3 * limits_tmp_size)
	void* d_limits_tmp = NULL;
	// min/max values for X, Y, and Z
	float *d_maxes, *d_mins;
	// Search query points
	MyPoint* d_query_points;
	// Search results
	MyMatches* d_query_results;
};
