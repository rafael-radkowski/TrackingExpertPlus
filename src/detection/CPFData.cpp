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