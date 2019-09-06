#pragma once




// stl
#include <iostream>
#include <vector>
#include <numeric>
#include <cassert>

// Eigen 3
#include <Eigen/Dense>

// local
#include "Cuda_KdTree.h"
#include "Types.h"

using namespace std;


namespace texpert {


using Matches = MyMatches;

class NearestNeighbors
{
public:

	NearestNeighbors();
	~NearestNeighbors();


	/*
	Set the reference point cloud. 
	This one goes into the kd-tree as soon as it is set. 
	@param pc - reference to the point cloud model
	*/
	bool setReferenceModel(PointCloud& pc);

	/*
	Set the test model, this is tested agains the 
	reference model in the kd-tree
	@param pc - reference to the point cloud model
	*/
	bool setTestModel(PointCloud& pc);


	/*
	Start the knn search and return matches.
	@param k - the number of matches to return
	@param matches - reference to the matches
	*/
	int knn(int k, vector<Matches>& matches);




	/*
	Reset the tree
	*/
	int reset(void);
	


private:

	/*
	Check if this class is ready to run.
	The kd-tree and the test points - both need to have points
	@return - true, if it can run. 
	*/
	bool ready(void);

	///////////////////////////////////////////////////////
	// Members

	// reference to the test point cloud and the 
	// reference point cloud (environment)
	PointCloud*		_refPoint;
	PointCloud*		_testPoint;

	// the kd-tree
	Cuda_KdTree*		_kdtree;

	// reference points
	vector<Cuda_Point>	_rpoints;

	// test points
	vector<Cuda_Point>	_tpoints;

	// the matches
	Matches				_matches;


	bool					_ready;

};


}//namespace texpert 