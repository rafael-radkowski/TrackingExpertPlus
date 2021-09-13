#include "CPFData.h"


/*
Constructor
*/
CPFModelData::CPFModelData():
	m_points(PointCloud()), m_label("")
{

}

/*
Constructor
*/
CPFModelData::CPFModelData(PointCloud& points, std::string label):
	m_points(points), m_label(label)
{

}

/*
Destructor
*/
CPFModelData::~CPFModelData()
{

}


/*
Return the size of the point cloud.
@return integer with the number of points.
*/
int CPFModelData::size(void)
{
	return m_points.size();
}


/*
Clear the descriptors and curvatures.
*/
void CPFModelData::clear(void)
{
	m_model_curvatures.clear();
	m_model_descriptors.clear();
}

/*
Return a reference to the point cloud
@return reference to the point cloud of type PointCloud.
*/
PointCloud& CPFModelData::getPointCloud(void)
{
	return m_points;
}


/*
Return a read-write reference to the descriptor array;
@return descriptors saved in a vector<> of type CPFDiscreet
*/
CPFDiscreetVec& CPFModelData::getDescriptor(void)
{
	return m_model_descriptors;
}

/*
Return a refereence to the Curvature location.
*/
CPFCurvatureVec& CPFModelData::getCurvature(void)
{
	return m_model_curvatures;
}



///////////////////////////////////////////////////////////////////////////////////////////////////
//




CPFMatchingData::CPFMatchingData()
{

}


CPFMatchingData::~CPFMatchingData()
{

}


// function to clear all votes
void CPFMatchingData::voting_clear(void) {
	m_pose_candidates.clear();
	m_vote_pair.clear();
	m_pose_candidates_votes.clear();



}

// function to clear all clusters
void CPFMatchingData::cluster_clear(void) {
	m_pose_clusters.clear();
	m_pose_cluster_votes.clear();
	m_debug_pose_candidates_id.clear();
	m_poses.clear();
	m_poses_votes.clear();

}


/*
Return a reference to the vote pair vector.
*/
std::vector<std::pair<float, int>>& CPFMatchingData::getVotePairVec(void)
{
	return m_vote_pair;
}

std::vector<Eigen::Affine3f >& CPFMatchingData::getPoseCandidatesPose(void)
{
	return m_pose_candidates;
}


vector<int>& CPFMatchingData::getPoseCandidatesVotes(void)
{
	return m_pose_candidates_votes;
}


std::vector<std::pair<int, int>>& CPFMatchingData::getMatchingPairs(void)
{
	return m_matching_pairs;
}


////////////////////////////////////////////////////////////////////////////////////////////////////////
//
// CPFDataDB 

CPFDataDB * CPFDataDB::m_instance = nullptr;

// private constructor
CPFDataDB::CPFDataDB()
{
	scene_data = NULL;
	matching_data = NULL;
}



/*
Get an instance of the class
*/
//static 
CPFDataDB* CPFDataDB::GetInstance(void)
{
	if(m_instance == nullptr){
		m_instance = new CPFDataDB();
	}
	return m_instance;
}


/*
Destructor
*/
CPFDataDB::~CPFDataDB()
{
	// Delete all instance 
	for(auto m : model_data)
		delete m.second;

	if(scene_data != NULL)
		delete scene_data;
}


/*
Create a model instance
@param points - point cloud reference of type PointCloud
@param label - name for this point cloud.
@param instance of CPFModelData
*/
CPFModelData* CPFDataDB::createModelInstance(PointCloud& points, std::string label)
{
	// Key is not present
	if (model_data.find(label) == model_data.end()){
		CPFModelData* model = new CPFModelData(points, label);
		model_data[label] = model;
	}else
	{
		cout << "[WARNING] - CPFDataDB: model " << label << " already exist. Cannot create a new model with the same name." << endl;
	}
	return model_data[label];
}

/*
Create a sceene instance
@param points - point cloud reference of type PointCloud
@param instance of CPFModelData
*/
CPFSceneData* CPFDataDB::createSceneInstance(PointCloud& points)
{
	if(scene_data == NULL){
		scene_data = new CPFSceneData(points, "scene");
	}else{
		cout << "[WARNING] - CPFDataDB: scene data was already initiated. Did not create a new instance." << endl;
	}

	return scene_data;
}


/*
Return model data by name
@param label - string containing the label of the model
@return pointer to the model data.
*/
CPFModelData* CPFDataDB::getModelData(std::string label)
{
	return model_data[label];
}

/*
Return model data by name
@return pointer to the scene data.
*/
CPFModelData* CPFDataDB::getSceneData(void)
{
	if(scene_data == NULL){
		cout << "[ALARM] - CPFDataDB::getSceneData(void) - scene_data is NULL" << endl;
	}
	return scene_data;
}




/*
	Return a matching data instance.
	*/
CPFMatchingData* CPFDataDB::GetMatchingData(void)
{
	if(matching_data == NULL){
		matching_data = new CPFMatchingData();
	} 
	return matching_data;
	
}





//-----------------------------------------------------------------------------------------------------
//
