#pragma once
/*
@class CPFDetect

The class is the main class for the feature detector. 
It manages the control flow and the process the detector takes. 

Process:
1) add a reference mode with addModel()
2) add a scene with setScene()
3) call match to obtain a pose. 

Note that the class works currently with one object only (future ToDo).

Responsibilties
- Save the reference models and control the descriptor extraction process
- Save the scene model and control the descritptor extraction process
- Match scene and reference model and return a pose. 

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

// local
#include "CPFRenderHelpers.h"
#include "Types.h"
#include "CPFTypes.h"
#include "CPFTools.h"
#include "KNN.h"


#include "CPFData.h"
#include "CPFMatching.h"
#include "CPFClustering.h"

namespace texpert_experimental {

	class CPFDetect
	{
	public:

		/*
		Constructor
		*/
		CPFDetect();

		/*
		Destructor
		*/
		~CPFDetect();


		/*!
		Add the reference point set, a point cloud model of an object to detect and track.
		Note that adding the model will also start the descriptor extraction process.
		@param points - a reference to a point cloud model containing points and normal vectors.
		@param label - an object label.
		@return an model id as integer or -1 if the model was rejected.
		*/
		int addModel(PointCloud& points, std::string label);


		/*!
		Set a scene model or a point cloud from a camera as PointCloud object.
		Note that adding the model will also start the descriptor extraction process.
		@param points - point and normal vectors.
		@return true if the scene was accepted.
		*/
		bool setScene(PointCloud& points);


		/*!
		Start the detection and pose estimation process for a model with the id model_id.
		Invoking this function will start descriptor matching for model_id, voting, and pose clustering.
		@para model_id - the id of the model to be detected (the int returned from addModel().
		*/
		bool match(int model_id);


	private:

		// enable text output
		bool				m_verbose;
		int					m_verbose_level;

		//TODO: extend to a vector. 
		// data for one model
		CPFModelData* m_model_data;

		// dataset to store the scene data. 
		CPFSceneData* m_scene_data;

		// parameters for descriptor and curvature extraction and matching. 
		CPFMatchingParams	m_params;


		// stores the matching and voting results per object. 
		CPFMatchingData* m_matching_results;


		CPFClusteringData* m_clustering_results;
		CPFClusteringParam	m_cluster_param;
	};

}