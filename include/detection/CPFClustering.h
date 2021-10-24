#pragma once
/*
@class CPFClustering

The class clusters multiple pose results by identifying aligning poses. 

Process:

Responsibilties


Rafael Radkowski
Aug 2021

MIT License
--------------------------------------------------------------------------------
Last edited:


*/

//stl 
#include <iostream>
#include <string>
#include <algorithm>
#include <vector>
#include <iostream>
#include <unordered_map>
#include <algorithm>

// Eigen
#include <Eigen/Dense>
#include <Eigen/Geometry>

// tracking expert
#include "CPFData.h"



namespace texpert_experimental
{

	typedef struct CPFClusteringParam {

		float custering_threshold_R;
		float custering_threshold_t;

		bool verbose;


		CPFClusteringParam() {

			custering_threshold_R = 0.1f;
			custering_threshold_t = 0.01f;

			verbose = false;

		};


	}CPFClusteringParam;


	class CPFClustering
	{
	public:

		/*
		Cluster the poses
		@param data CPFMatchingData, the matching results
		*/
		static int Clustering(CPFMatchingData& src_data, CPFClusteringData& dst_clustering, CPFClusteringParam& params);


		static int GetBest(CPFClusteringData& dst_clustering, CPFClusteringParam& params);

	private:


		static bool similarPose(Eigen::Affine3f a, Eigen::Affine3f b, CPFClusteringParam& param);


		static bool combinePoseCluster(std::vector<Eigen::Affine3f>& pose_clustser, int votes, CPFClusteringData& dst_data, CPFClusteringParam& param, bool invert = false);


	};

}