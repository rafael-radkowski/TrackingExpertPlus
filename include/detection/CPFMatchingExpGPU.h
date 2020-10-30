#pragma once
/*
@class CPFMatchingExp(erimental)GPU

@brief The class is an implementation of CPFMatchingExp that utilizes CPFToolsGPU instead of its CPU
counterpart, CPFTools.

Note that this class implements a naive version of this code, which acts as as experimental platform.
Use CPFMatching as a deliverable implementation.

William Blanchard
Iowa State University
wsb@iastate.edu
(847) 707-1421
27 Oct 2020

MIT License
-------------------------------------------------------------------------------------------------------
Last edits:
28 October 2020
- Added better allocation/deallocation of memory for CPFToolsGPU
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
#include "CPFToolsGPU.h"
#include "KNN.h"

namespace texpert {

	class CPFMatchingExpGPU
	{
	private:

		/*!
		Struct to keep the results and temporary data for one object.
		*/
		typedef struct CPFMatchingData {

			//----------------------------------------------------------------
			// Pose candidates and vote pairs for one model
			std::vector<Eigen::Affine3f >			pose_candidates; // stores the pose candidates as Eigen affine matrix. 
			vector<int>								pose_candidates_votes; // stores the votes each pose candidates get. Votes and poses are index aligned. 

			std::vector<std::pair<float, int>>		vote_pair;	// vote pairs, stores an angle and the id of an obect

			//----------------------------------------------------------------
			// pose clustering for one model

			std::vector< std::vector<Eigen::Affine3f> >		pose_clusters;   // the pose cluster or potential pose candidates of an object. 
			std::vector< std::pair<int, int> >				pose_cluster_votes; // < votes, cluster index> Stores the votes for each cluster. THe list is sorted so that the highest votes win. 
			std::vector< std::vector<int> >					debug_pose_candidates_id; // [cluster] -> pose candidate id>  Links the individual pose candidates to the cluster for debugging. 

			//--------------------------------------------------------------------
			// the winner poses
			std::vector< Eigen::Affine3f >					poses;
			std::vector<int >								poses_votes;

			// function to clear all votes
			void voting_clear(void) {
				pose_candidates.clear();
				vote_pair.clear();
				pose_candidates_votes.clear();
			}

			// function to clear all clusters
			void cluster_clear(void) {
				pose_clusters.clear();
				pose_cluster_votes.clear();
				debug_pose_candidates_id.clear();
				poses.clear();
				poses_votes.clear();
			}

		}CPFMatchingData;

	public:

		CPFMatchingExpGPU();
		~CPFMatchingExpGPU();


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


		/*!
		Return the poses for a particular model.
		The function returns the 12 best hits by default. Note that this can be reduced to only 1 or so.
		@param model_id - the model id of the object to track as int.
		@param poses - vector with the poses
		@param pose_votes - vector with the pose votes.
		*/
		bool getPose(const int model_id, std::vector<Eigen::Affine3f >& poses, std::vector<int >& pose_votes);

		/*!
		Return the number of pose clusters for a particular model.
		@param model_id - the model id of the object to track as int.
		*/
		int getNumPoseClusters(int model_id);

		/*!
		Enable extra debug messages.
		@param verbose - output extra debug messages if true.
		@param level - the verbose level 0, 1, 2. The higher the number, the more detailed the info.
		*/
		bool setVerbose(bool verbose, int level = 0);


		/*!
		Set the parameters for the extraction tool.
		@param params - a struct containing some parameters.
		*/
		bool setParams(CPFParams params);


		//-----------------------------------------------------------------------------------------------------------------
		// Render and debug helpers

		/*!
		The class contains a tool that generates extra data for rendering and visual debugging.
		However, this requires time and memory, thus, can be disabled or enabled on demand.
		@param enable - enables the render helper tools.
		*/
		bool enableRenderHelpers(bool enable);

		/*!
		Return the RenderHelper object to access the data.
		Note that no data is produced if the helper is empty.
		*/
		CPFRenderHelpers& getRenderHelper(void);

		/*!
		For visual debugging.
		Return the point pairs of a pose cluster. The point pairs indicate the descriptor locations used to
		calculate the final pose.
		@param cluster_id - the cluster id. Note that the cluster are sorted. Cluster 0 has the highest number of votes.
		@param cluster_point_pairs - reference to store the cluster point pairs as <reference model point, scene point>.
		*/
		bool getPoseClusterPairs(const int cluster_id, std::vector< std::pair<int, int> >& cluster_point_pairs);


		/*!
		For visual debugging.
		Return the discretized curvature values for the model.
		*/
		bool getModelCurvature(const int model_id, std::vector<uint32_t>& model_cu);

		/*!
		For visual debugging.
		Return the discretized curvature values for the scene.
		*/
		bool getSceneCurvature(std::vector<uint32_t>& scene_cu);


	private:


		/*
		Descriptor based on curvature pairs and the direction vector
		*/
		void calculateDescriptors(PointCloud& pc, float radius, std::vector<CPFDiscreet>& descriptors, std::vector<uint32_t>& curvatures);


		/*
		Match the model and scene descriptors.
		@param src_model - the descriptors of the reference model
		@param src_scene - the scene descriptors
		@param pc_model - reference to the model point cloud data.
		@param pc_scene - reference to the scene point cloud data.
		@param dst_data - location for all destination data.
		*/
		void matchDescriptors(std::vector<CPFDiscreet>& src_model, std::vector<CPFDiscreet>& src_scene, PointCloud& pc_model, PointCloud& pc_scene,
			CPFMatchingData& dst_data);


		/*
		Invoke the pose candidate clustering process.
		Clusters the poses depending on their difference. Similar poses form a similar cluster.
		@param data - the internal model data struct.
		*/
		bool clustering(CPFMatchingData& data);


		/*
		Calculate the similarity between two poses.
		*/
		bool similarPose(Eigen::Affine3f a, Eigen::Affine3f b);


		/*
		Combine the pose clusters to one pose.
		*/
		bool combinePoseCluster(std::vector<Eigen::Affine3f>& pose_clustser, int votes, CPFMatchingData& dst_data, bool invert);

		//-------------------------------------------------------------

		// vector with addresses to reference point sets
		std::vector<PointCloud>		m_ref;
		std::vector<std::string>	m_ref_labels;

		// reference to the scene point cloud
		PointCloud					m_scene;

		// descriptor parameterss
		CPFParams					m_params;


		KNN* m_knn;

		//--------------------------------------------------------------
		// the model
		// descriptors and curvatures
		std::vector< std::vector<CPFDiscreet> >	m_model_descriptors;
		std::vector< std::vector<uint32_t> >    m_model_curvatures;

		// scene descriptors and curvaturs
		std::vector<CPFDiscreet>				m_scene_descriptors;
		std::vector<uint32_t>					m_scene_curvatures;

		// stores the matching, voting, and clustering results per object. 
		std::vector<CPFMatchingData>			m_matching_results;

		// for debuging and to render descriptor content.
		CPFRenderHelpers						m_helpers;

		// params
		int										m_angle_bins;
		float									m_multiplier;

		bool									m_verbose;
		bool									m_render_helpers;
		int										m_verbose_level;

		int										m_max_points;
	};

}
