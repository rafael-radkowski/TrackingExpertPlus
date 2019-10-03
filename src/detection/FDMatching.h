#pragma once
/*
class FDMatching

Extended or improved version of the point pair feature tracking method. 
It does not match all points in the piont set.

It works only with points with a curvature difference > x degree and also only with points further away from the 
indicated goal. 

The class implements a version of the pair-point feature, 
Bertram Drost et al., Model Globally, Match Locally: Efficient and Robust 3D Object Recognition
http://campar.in.tum.de/pub/drost2010CVPR/drost2010CVPR.pdf

Rafael Radkowski
Iowa State University
rafael@iastate.edu
Mit License
---------------------------------------------------------------
Last Changes:

March 20, 2019, RR
- Changed the variable g_dlg_ppf_nn = 9, to 9 since the kd-tree radius search is limited to 10 hits. 
- Inverted the final transformation pose.t.data()[12] , elements 12-14
- Added a verbose variable to enable or surpress console outputs
*/

// stl
#include <vector>
#include <iostream>
#include <unordered_map>
#include <conio.h>

// Eigen
#include <Eigen/Dense>
#include <Eigen/Geometry>

// local
#include "FDTypes.h"
#include "FDTools.h"
#include "Cuda_KdTree.h"
#include "FDClustering.h"

using namespace std;


namespace isu_ar {

	class FDMatching
	{
	public:
		FDMatching();
		~FDMatching();


		/*
		Set the distance step
		@param distance_step -  a value larger than 0.0;
		*/
		bool setDistanceStep(float distance_step);


		/*
		Set the angle step
		@param angle_step -  a value larger than 0.0;
		*/
		bool setAngleStep(float angele_step);


		/*
		Extract a ppf feature map for an object of interest
		@param points - point as {x, y, z}
		@param normals - normal vectors, points and normal vectors are index aligned.
		@param map_results - the map in which all the results should be inserted to.
		*/
		bool extract_feature_map(vector<Eigen::Vector3f>* points, vector<Eigen::Vector3f>* normals);



		/*
		Detect the feature map in the environment model.
		@param points - point as {x, y, z}
		@param normals - normal vectors, points and normal vectors are index aligned.
		@param map_results - the map in which all the results should be inserted to.
		*/
		bool detect(vector<Eigen::Vector3f>* points, vector<Eigen::Vector3f>* normals, std::vector<Pose>& poses);


		/*
		Set true to invert the pose. The standard pose transforms the
		reference object coord. to the test object coord, e.g. for a regular camera setting. The inverted pose
		translates the test object to the reference object.
		@param invert - set true to invert the pose. Default is false.
		*/
		bool invertPose(bool value);


		/*
		Set the cluster threshold for pose clusterin algorithm.
		Pose clustering considers all poses as equal (similar) if their
		center-distance and their angle delta are under a threshold.
		@param distance_th - the distance threshold.
		@param angle_th - the angle threshold in degress.
		BOTH VALUES NEED TO BE LARGER THAN 1
		*/
		void setClusteringThreshold(float distance_th, float angle_th);


		/*
		Enable console outputs.
		*/
		void setVerbose(bool verbose = true);

	private:

		// test object
		vector<Eigen::Vector3f>*                 _points_test;
		vector<Eigen::Vector3f>*				 _normals_test;


		// The reference object, the environment
		vector<Eigen::Vector3f>*                 _ref_points;
		vector<Eigen::Vector3f>*				 _ref_normals;

		// size
		int										_N;


		// the discretization steps.
		// Those discretize the angles and link between points into certain discrete steps. 
		float									_distance_step;
		float									_angle_step;
		int										_angle_bins;

		// The ppf map for the test object
		PPFMap									_map_test_points;

		// clustering 
		FDClustering							_cluster;

		// kdtree tofind the nearest matches
		Cuda_KdTree*							_kdtree;
		int										_k;

		bool									_verbose;
	};

}