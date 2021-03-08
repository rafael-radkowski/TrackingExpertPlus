#pragma once
/*
@class CPFMatchingWrapper

@brief The pure virtual class provides an interface for the CPFMatchingExp
and CPFMatchingExpGPU classes.

William Blanchard
Iowa State University
wsb@iastate.edu
(847) 707-1421
1 November 2020

MIT License
-------------------------------------------------------------------------------------------------------
Last edits:
1 November 2020
- Added class opening documentation
*/
#include <vector>

#include "Types.h"
#include "CPFTypes.h"

namespace texpert {
	class CPFMatchingWrapper
	{
	public:
		/*!
		Add the reference point set, a point cloud model of an object to detect and track.
		Note that adding the model will also start the descriptor extraction process.
		@param points - a reference to a point cloud model containing points and normal vectors.
		@param label - an object label.
		@return an model id as integer or -1 if the model was rejected.
		*/
		virtual int addModel(PointCloud& points, std::string label) = 0;


		/*!
		Set a scene model or a point cloud from a camera as PointCloud object.
		Note that adding the model will also start the descriptor extraction process.
		@param points - point and normal vectors.
		@return true if the scene was accepted.
		*/
		virtual bool setScene(PointCloud& points) = 0;


		/*!
		Start the detection and pose estimation process for a model with the id model_id.
		Invoking this function will start descriptor matching for model_id, voting, and pose clustering.
		@para model_id - the id of the model to be detected (the int returned from addModel().
		*/
		virtual bool match(int model_id) = 0;


		/*!
		Return the poses for a particular model.
		The function returns the 12 best hits by default. Note that this can be reduced to only 1 or so.
		@param model_id - the model id of the object to track as int.
		@param poses - vector with the poses
		@param pose_votes - vector with the pose votes.
		*/
		virtual bool getPose(const int model_id, std::vector<Eigen::Affine3f >& poses, std::vector<int >& pose_votes) = 0;

		/*!
		Return the number of pose clusters for a particular model.
		@param model_id - the model id of the object to track as int.
		*/
		virtual int getNumPoseClusters(int model_id) = 0;

		/*!
		Enable extra debug messages.
		@param verbose - output extra debug messages if true.
		@param level - the verbose level 0, 1, 2. The higher the number, the more detailed the info.
		*/
		virtual bool setVerbose(bool verbose, int level = 0) = 0;


		/*!
		Set the parameters for the extraction tool.
		@param params - a struct containing some parameters.
		*/
		virtual bool setParams(CPFParams params) = 0;


		//-----------------------------------------------------------------------------------------------------------------
		// Render and debug helpers

		/*!
		The class contains a tool that generates extra data for rendering and visual debugging.
		However, this requires time and memory, thus, can be disabled or enabled on demand.
		@param enable - enables the render helper tools.
		*/
		virtual bool enableRenderHelpers(bool enable) = 0;

		/*!
		Return the RenderHelper object to access the data.
		Note that no data is produced if the helper is empty.
		*/
		virtual CPFRenderHelpers& getRenderHelper(void) = 0;

		/*!
		For visual debugging.
		Return the point pairs of a pose cluster. The point pairs indicate the descriptor locations used to
		calculate the final pose.
		@param cluster_id - the cluster id. Note that the cluster are sorted. Cluster 0 has the highest number of votes.
		@param cluster_point_pairs - reference to store the cluster point pairs as <reference model point, scene point>.
		*/
		virtual bool getPoseClusterPairs(const int cluster_id, std::vector< std::pair<int, int> >& cluster_point_pairs) = 0;


		/*!
		For visual debugging.
		Return the discretized curvature values for the model.
		*/
		virtual bool getModelCurvature(const int model_id, std::vector<uint32_t>& model_cu) = 0;

		/*!
		For visual debugging.
		Return the discretized curvature values for the scene.
		*/
		virtual bool getSceneCurvature(std::vector<uint32_t>& scene_cu) = 0;
	};
}