/*
@class CPFModelData

The class holds all data essential for feature matching. 

Rafael Radkowski
Aug 2021

MIT License
--------------------------------------------------------------------------------
Last edited:


*/
#pragma once

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



class CPFModelData
{
protected:
	
	friend class CPFDataDB;

	/*
	Constructor
	*/
	CPFModelData();


	/*
	Constructor
	*/
	CPFModelData(PointCloud& points, std::string label);

public:
	/*
	Destructor
	*/
	~CPFModelData();



	/*
	Return the size of the point cloud. 
	@return integer with the number of points. 
	*/
	int size(void);


	/*
	Clear the descriptors and curvatures. 
	*/
	void clear(void);

	/*
	Return a reference to the point cloud
	@return reference to the point cloud of type PointCloud.
	*/
	PointCloud& getPointCloud(void);


	/*
	Return a read-write reference to the descriptor array;
	@return descriptors saved in a vector<> of type CPFDiscreet
	*/
	CPFDiscreetVec& getDescriptor(void);

	/*
	Return a refereence to the Curvature location. 
	*/
	CPFCurvatureVec& getCurvature(void);									


private:

	PointCloud&										m_points; 
	std::string										m_label;



	//--------------------------------------------------------------
	// the model
	// descriptors and curvatures
	CPFDiscreetVec									m_model_descriptors;
	CPFCurvatureVec									m_model_curvatures;

};


// alias for the scene data. 
using CPFSceneData = CPFModelData;






/*
@class CPFMatchingData

The class holds all data essential for feature matching.
These are temporary objects required to yield results and the final pose results. 

*/
class  CPFMatchingData {

public:
	
	CPFMatchingData();
	~CPFMatchingData();

	/*
	Clear all cluster data
	*/
	void cluster_clear(void);

	/*
	Clear all voting data
	*/
	void voting_clear(void);


	/*
	Return a reference to the vote pair vector. 
	*/
	std::vector<std::pair<float, int>>& getVotePairVec(void);

	std::vector<Eigen::Affine3f >& getPoseCandidatesPose(void);


	vector<int>& getPoseCandidatesVotes(void);

private:
	//----------------------------------------------------------------
	// Pose candidates and vote pairs for one model
	std::vector<Eigen::Affine3f >					m_pose_candidates; // stores the pose candidates as Eigen affine matrix. 
	vector<int>										m_pose_candidates_votes; // stores the votes each pose candidates get. Votes and poses are index aligned. 

	std::vector<std::pair<float, int>>				m_vote_pair;	// vote pairs, stores an angle and the id of an obect

	//----------------------------------------------------------------
	// pose clustering for one model

	std::vector< std::vector<Eigen::Affine3f> >		m_pose_clusters;   // the pose cluster or potential pose candidates of an object. 
	std::vector< std::pair<int, int> >				m_pose_cluster_votes; // < votes, cluster index> Stores the votes for each cluster. THe list is sorted so that the highest votes win. 
	std::vector< std::vector<int> >					m_debug_pose_candidates_id; // [cluster] -> pose candidate id>  Links the individual pose candidates to the cluster for debugging. 

	//--------------------------------------------------------------------
	// the winner poses
	std::vector< Eigen::Affine3f >					m_poses;
	std::vector<int >								m_poses_votes;

};




class CPFDataDB{
private:

	// private constructor
	CPFDataDB();

public:

	/*
	Get an instance of the class
	*/
	static CPFDataDB* GetInstance(void);


	/*
	Destructor	
	*/
	~CPFDataDB();


	/*
	Create a model instance.
	Note that the function returns an existing instance if one with the same name already exists. 
	@param points - point cloud reference of type PointCloud
	@param label - name for this point cloud.
	@param instance of CPFModelData
	*/
	CPFModelData* createModelInstance(PointCloud& points, std::string label);

	/*
	Create a sceene instance
	@param points - point cloud reference of type PointCloud
	@param instance of CPFModelData
	*/
	CPFSceneData* createSceneInstance(PointCloud& points);


	/*
	Return model data by name
	@param label - string containing the label of the model
	@return pointer to the model data. 
	*/
	CPFModelData* getModelData(std::string label);
	
	/*
	Return model data by name
	@return pointer to the scene data.
	*/
	CPFModelData* getSceneData(void);

private:

	// storage for all model data <model label, data>
	std::unordered_map< std::string, CPFModelData*>	model_data;

	// storage for the scene data
	CPFSceneData*		scene_data;


	static CPFDataDB*	m_instance;

};