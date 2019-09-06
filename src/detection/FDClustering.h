#pragma once
/*
Class FDClustering

The class clusters poses depending on their similarity. 
Two poses are considered as similar if their distance is within a certain threshold margin
and their coordinate frame alignment is similar.

A translation and a rotation threshold need to be set for similarity. 

It implements a version of the pair-point feature clustering, 
Bertram Drost et al., Model Globally, Match Locally: Efficient and Robust 3D Object Recognition
http://campar.in.tum.de/pub/drost2010CVPR/drost2010CVPR.pdf

Rafael Radkowski
Iowa State University
rafael@iastate.edu
All copyrights reserved.
---------------------------------------------------------------
Latest edits

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

using namespace std;

namespace isu_ar {

	class FDClustering
	{
	public:
		FDClustering();
		~FDClustering();

		/*
		Cluster the poses. 
		The function clusters the poses into groups depending on the similarity. Similarity is defined as the angle delta and 
		the distance. 
		@param poses - the input poses of type Pose. 
		@param pose_clusters - a location to store the pose clusters. 
		@param votes - a location to store the number of votes. 
		@return true - if poses were successfully clusterd
		*/
		bool clusterPoses(const std::vector<Pose>& poses, vector<Pose>& pose_clusters, vector<int>& votes);
	
		/*
		Set the translation threshold for two poses to be considered as similar. 
		The number should match the dimensions of the object of interest. 
		@param value - the threshold. 
		*/
		void setTranslationThreshold(float value);

		/*
		Set the max. rotation angle for two poses to be considered as equal
		@param value - the rotation threshold in degress. 
		*/
		void setRotationThreshold(float value);

		/*
		Set true to invert the pose. The standard pose transforms the 
		reference object coord. to the test object coord, e.g. for a regular camera setting. The inverted pose
		translates the test object to the reference object. 
		@param invert - set true to invert the pose. Default is false. 
		*/
		void invertPose(bool invert);

	private:

		/*
		Calculates the similarity between a and b. 
		a and b are 4x4 matrices with affine transformations. 
		*/
		bool similar(Eigen::Affine3f a, Eigen::Affine3f b);


		//------------------------------------------------------------
		// Members 


		// translation and rotation threshold 
		// so that poses are considered as similar.
		float	_translation_threshold;
		float	_rotation_threshold;


		std::vector<std::pair<int, int> >		_temp_cluster_votes;
		std::vector<std::vector<Pose>>			_temp_clustered_poses;

		// to invert the pose if the test object should be moved to the
		// reference object
		bool									_invert;
	};
};