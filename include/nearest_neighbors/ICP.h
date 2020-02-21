#pragma once


// stl
#include <iostream>
#include <vector>
#include <numeric>
#include <algorithm>
#include <functional>

// Eigen 3
#include <Eigen/Dense>

// local
#include "FDTypes.h"
#include "ICPTransform.h"
#include "KNN.h"
#include "Types.h"
#include "MatrixUtils.h"

namespace texpert{


class ICP
{
public:

	ICP();
	~ICP();

		/*
	Set the reference point cloud. 
	This one goes into the kd-tree as soon as it is set. 
	@param pc - reference to the point cloud model
	*/
	bool setCameraData(PointCloud& pc);

	/*
	Set the test model, this is tested agains the 
	reference model in the kd-tree
	@param pc - reference to the point cloud model
	*/
	bool compute(PointCloud& pc, Pose initial_pose, Eigen::Matrix4f& result_pose, float& rms);


	/*
	Set the amount of output plotted to the dialot. 
	@param verbose - true enables outputs. 
	@param int - level 1 or 2 sets the amount of outputs.
			1 gives just basis data
			2 prints data per ICP iterations
	*/
	bool setVerbose(bool verbose, int verbose_level = 1);


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
	PointCloud				_cameraPoints;
	PointCloud				_testPoints;
	PointCloud				_testPointsProcessing;

	// k-nearest neighbors implementation
	KNN*					_knn;		


	// icp params
	float					_max_error;
	int						_max_iterations;

	bool					_verbose;
	int						_verbose_level;
};


}//namespace texpert{