 #pragma once
/*
class ICP
files: ICP.h/.cpp

@brief: The class implements the Iterative Closest Point algorithm, 
its point-to-point version according to 
Besl, McKay, A method for registration of 3D shapes, PAMI, 1992.

	arg min -> sum ( R m + t - p)

with m, the model points and p, the camera points. The matrix R and the translation t 
align the point set m with p. 

Note that the class's apis are labels for camera data. 

The class keeps an instance of a CUDA-based kd-tree for knn search. 
Thus, the class does not work if CUDA capabilities are not present. 

Also, this implementation has two termination criteria. 
First, a minimum error that has to be reached. The error can be set with setMinError().
Second, a maximum number of iterations. This number can be set with setMaxIterations().

The second termination criterium is helpful when working with real-time camera data and  augmented reality. 
It may yield imperfect registration when an object is moving. But the user does not notice this. 

Usage:
	1) set the camera data using setCameraData(PointCloud& pc);
	2) register a 3D model, its poitns using compute(PointCloud& pc, Pose initial_pose, Eigen::Matrix4f& result_pose, float& rms);

ICP does not run if no points are given.

Tests:
July 23, 2019, RR: the class was tested with static stanford bunny models and worked just fine. 

Dependencies:
- ICPTransform.h/.cpp
- KNN.h/.cpp
- ICPReject.h/.cpp


Rafael Radkowski
Iowa State University
rafael@iastate.edu
July 2019

MIT License
-----------------------------------------------------------------------------------------------
Last edited:

Feb 22, 2020, RR
- Fixed a bug that set the verbose level incorrectly. 

March 3, 2020, RR
- Added two apis to set the inlier outlier rejection criteria params.

March 16, 2020, RR
- Added a api that returns the last nearest neighbors vectors. This is a debug 
 api that allows one to understand the nn better. It is not meant for general use. 

 June 3rd, 2020, RR
 - Added an api to set the outlier rejection method. 

 June 11, 2020, RR
 - Added two variables (R, t) to keep track of the rotation and translation individually. 
 - Fixed a bug: the ICP code rotated an object around the wrong centroid. 
 - Added a function to return the overall translation and rotation. The transformation can be applied on 
	a reference object to register it with its counterpart. 

Aug 06, 2020, RR
- Incorporated the initial pose into the final pose so that the result of this class is 
  a complete pose. 
- Set default values for all parameters. 

Aug 07, 2020, RR
- Added code to check the parameters ranges. 

Mar 08, 2021, WB
- Fixed ICP Rt to return a non-transposed matrix
- Transferred most Rt calculations to the PointCloudTrans class

*/


// stl
#include <iostream>
#include <vector>
#include <numeric>
#include <algorithm>
#include <functional>
#include <chrono>

// Eigen 3
#include <Eigen/Dense>

// local
#include "FDTypes.h"
#include "ICPTransform.h"
#include "ICPReject.h"
#include "KNN.h"
#include "Types.h"
#include "MatrixUtils.h"
#include "PointCloudUtils.h"
#include "cuICP.h"
#include "MatrixConv.h"
#include "PointCloudTrans.h"


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


	/*
	Set a minimum error as termination criteria.
	@param error - a float value > 0.0
	*/
	void setMinError(float error);


	/*
	Set the number of maximum iterations. 
	@param max_iterations - integer number with the max. number of iterations.
		The number must be in the range [1, 1000] 
	*/
	void setMaxIterations(int max_iterations);


	/*
	Set the maximum angle delta for two points to be considered
	as inliers. All other points will be rejected. 
	@param max_angle - the maximum angle in degrees. 
		The value must be between 0 and 180 degrees. 
	*/
	void setRejectMaxAngle(float max_angle);


	/*
	Set the maximum value for two point sets to be considered
	as inliers. 
	@param max_distance - a float value larger than 0.01;
	*/
	void setRejectMaxDistance(float max_distance);



	/*
	Set the ICP outlier rejection mechanism. 
	@param method - NONE, DIST, ANG, DIST_ANG
	*/
	void setRejectionMethod(ICPReject::Testcase method);


	/*!
	Return the overall rotation after ICP terminates. 
	@return a 3x3 matrix with the last rotation. 
	*/
	Matrix3f R(void) {return _R_all;}		

	/*!
	Return the overall translation after ICP terminates. 
	@return - a vector 3 with x, y, z coordinates. 
	*/
	Vector3f t(void) {return _t_all;} 


	/*!
	Return a 4x4 matrix (homogenous space) with the rotation and translation. 
	Note the the matrix rotates the reference object around its centroid. 
	The function internally computes 
			T = t * centroid * R * centroid_inverse;

	@return - 4x4 matrix with the object transformation to register the reference point set with 
	its counterpart in the camera point set. 
	*/
	Matrix4f Rt(void);


	Matrix4f Rt2(void);


	//-------------------------------------------------------------------------------

	/* DEBUG FUNCTION
	   Return the last set of nearest neighbors from the knn search. 
	   @return vector containing the nn pairs as indices pointing from the reference point set
	   to the envrionment point set. 
	   Note that this functionality is just for debugging. It is performance consuming and should not be used
	   under normal operations. 
	*/
	std::vector<std::pair<int, int> >& getNN(void);


private:


	// this is a tests function to test the functionality of the translation and orientation calculation. 
	// Use it by defining #define ICPTRANSTEST.
	// Note that this function does not use any nearest neighbors function and assumes that 
	// _testPoints and _cameraPoints are equal and index aligned. 
	bool test_transformation(PointCloud& pc, Pose initial_pose, Eigen::Matrix4f& result_pose, float& rms);
	bool test_rejection(PointCloud& pc, Pose initial_pose, Eigen::Matrix4f& result_pose, float& rms);

	

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
	Vector3f				_testPoint_centroid;

	// translation and rotation for all iterations. 
	Matrix3f				_R_all;
	Vector3f				_t_all; 
	Matrix4f				_Rt_initial;//inintial pose
	Affine3f				_Rt_affine; // initial pose

	Matrix4f				_Rt_final;

	std::vector<Matches>	_local_matches;

	// k-nearest neighbors implementation
	KNN*					_knn;		

	// cuda class to run icp
	cuICP*					_cuicp;


	// tests for outlier rejection
	ICPReject				_outlier_reject;
	ICPReject::Testcase		_outlier_rejectmethod;

	// icp params
	float					_max_error;
	int						_max_iterations;

	bool					_verbose;
	int						_verbose_level;

	// helper to debug knn hits
	std::vector<std::pair<int, int>> _verbose_matches;

	// Matrix conversion singleton class
	MatrixConv* _conv;
};


}//namespace texpert{