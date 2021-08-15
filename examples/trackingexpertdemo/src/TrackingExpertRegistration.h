#pragma once

/*

----------------------------------------------------------------------------------
Last edits:

Aug 6, 2020, RR
- Added two pointers for the point cloud addresses (The current version of the code has too many
  point cloud copies. 
*/

// STL
#include <iostream>
#include <string>
#include <Windows.h>
#include <fstream>
#include <thread>
#include <mutex>
#include <vector>

// GLM include files
#define GLM_FORCE_INLINE
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtx/transform.hpp>  // transformation
#include <glm/gtx/quaternion.hpp> // quaternions

// eigen
#include <Eigen\Dense>

// TrackingExpert
#include "CPFMatchingExp.h"
#include "CPFMatchingExpGPU.h"
#include "ICP.h"
#include "TrackingExpertParams.h"

class TrackingExpertRegistration
{
public:
	TrackingExpertRegistration();
	~TrackingExpertRegistration();

	/*!
	Add the reference point set, a point cloud model of an object to detect and track. 
	Note that adding the model will also start the descriptor extraction process. 
	@param points - a reference to a point cloud model containing points and normal vectors. 
	@param label - an object label. 
	@return an model id as integer or -1 if the model was rejected. 
	*/
	int addReferenceModel(PointCloud& points, std::string label);



	/*!
	Set or update the scene point cloud from a camera as PointCloud object. 
	Note that adding the model will also start the descriptor extraction process. 
	@param points - point and normal vectors. 
	@return true if the scene was accepted. 
	*/
	bool updateScene(PointCloud& points);



	/*!
	Start the detection and pose estimation process for a model with the id model_id.
	Invoking this function will start descriptor matching for model_id, voting, and pose clustering. 
	@para model_id - the id of the model to be detected (the int returned from addModel().
	*/
	bool process(void);


	/*!
	Return the poses for a particular model.
	The function returns the 12 best hits by default. Note that this can be reduced to only 1 or so. 
	@param model_id - the model id of the object to track as int. 
	@param poses - vector with the poses
	@param pose_votes - vector with the pose votes. 
	*/
	bool getPose( std::vector<Eigen::Affine3f >& poses);


	/*!
	Reset registration and move the reference model to its original start position.
	*/
	bool reset(void);


	/*!
	Returns the pose after ICP was applied. 
	@return a 4x4 matrix in homogenous coordinates with the pose. 
	*/
	Eigen::Matrix4f  getICPPose(void);



	/*!
	Enable extra debug messages. 
	@param verbose - output extra debug messages if true. 
	*/
	bool setVerbose(bool verbose);


	/*!
	Set the parameters for the extraction tool.
	@param params - a struct containing some parameters. 
	*/
	bool setParams(TEParams	m_params);



	//---------------------------------------------------------------------------------
	// Render helpser

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


	/*
	Return the number of pose clusters for the current model
	*/
	int getNumPoseClusters(void);

	/*!
	For visual debugging. 
	Return the discretized curvature values for the model.
	*/
	bool getModelCurvature( std::vector<uint32_t>& model_cu );

	/*!
	For visual debugging. 
	Return the discretized curvature values for the scene.
	*/
	bool getSceneCurvature(std::vector<uint32_t>& scene_cu);



private:

	void init(void);

	//------------------------------------------------------------------
	// Params

	// feature detector and matching
	CPFMatchingWrapper*	m_fd;
	CPFParams			m_fd_params;
	bool				m_working;

	// ICP
	ICP*				m_icp;

	// 
	int					m_model_id;

	std::vector<int>	m_pose_votes;
	float				m_rms;

	//------------------------------------------------------------------
	// Point clouds
	PointCloud			m_model_pc;
	PointCloud			m_scene_pc;

	PointCloud*			m_ptr_model_pc;
	PointCloud*			m_ptr_scene_pc;

	// the global pose of the model 
	Eigen::Matrix4f		m_model_pose;

	//------------------------------------------------------------------
	// Parameters
	TEParams			m_params;
};