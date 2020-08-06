#include "TrackingExpertRegistration.h"


TrackingExpertRegistration::TrackingExpertRegistration()
{
	init();
}


TrackingExpertRegistration::~TrackingExpertRegistration()
{
	delete m_fd;
	delete m_icp;
}


void TrackingExpertRegistration::init(void)
{
	m_model_id = -1;

	// create an ICP and feature descriptor instance. 
	m_fd = new CPFMatchingExp();
	m_icp = new ICP();

	// set the default params.
	m_fd_params.search_radius = 0.1;
	m_fd_params.angle_step = 12.0;
	m_fd->setParams(m_fd_params);


	m_icp->setMinError(0.00000001);
	m_icp->setMaxIterations(200);
	m_icp->setVerbose(true, 2);
	m_icp->setRejectMaxAngle(45.0);
	m_icp->setRejectMaxDistance(0.1);
	m_icp->setRejectionMethod(ICPReject::DIST_ANG);

}


/*!
Add the reference point set, a point cloud model of an object to detect and track. 
Note that adding the model will also start the descriptor extraction process. 
@param points - a reference to a point cloud model containing points and normal vectors. 
@param label - an object label. 
@return an model id as integer or -1 if the model was rejected. 
*/
int TrackingExpertRegistration::addReferenceModel(PointCloud& points, std::string label)
{
	assert(m_fd != NULL);

	if(m_model_id != -1) return -1;

	m_ptr_model_pc = &points;

	// copies the point cloud (too much copies in this code).
	m_model_pc = points;

	m_model_id = m_fd->addModel(points, label);

	return m_model_id;
}


/*!
Set or update the scene point cloud from a camera as PointCloud object. 
Note that adding the model will also start the descriptor extraction process. 
@param points - point and normal vectors. 
@return true if the scene was accepted. 
*/
bool TrackingExpertRegistration::updateScene(PointCloud& points)
{
	assert(m_fd != NULL);
	assert(m_icp != NULL);

	// copies the point cloud (too much copies in this code).
	m_scene_pc = points;

	m_ptr_model_pc = &points;

	return m_fd->setScene(points);
}


/*!
Start the detection and pose estimation process for a model with the id model_id.
Invoking this function will start descriptor matching for model_id, voting, and pose clustering. 
@para model_id - the id of the model to be detected (the int returned from addModel().
*/
bool TrackingExpertRegistration::process(void)
{
	assert(m_fd != NULL);
	assert(m_icp != NULL);

	if(m_model_id == -1) return false;

	bool ret = true;

	ret = m_fd->match(m_model_id);

	// positive match
	if (ret) {

		// Do not set the scene data earlier. 
		// icp and fd share the same knn. 
		// m_fd->setScene(points) overwrites the data. 
		m_icp->setCameraData(m_scene_pc);

		std::vector<Eigen::Affine3f > pose;
		getPose(pose);
		
		if (pose.size() <= 0) {
			return false;
		}	

		texpert::Pose Rt;
		Rt.t = pose[0];

		Eigen::Matrix4f out_Rt;
		float rms = 0.0;
		m_icp->compute(m_model_pc, Rt, out_Rt, rms);

		std::cout << rms << std::endl;
	}

	return true;
}


/*!
Return the poses for a particular model.
The function returns the 12 best hits by default. Note that this can be reduced to only 1 or so. 
@param model_id - the model id of the object to track as int. 
@param poses - vector with the poses
@param pose_votes - vector with the pose votes. 
*/
bool TrackingExpertRegistration::getPose( std::vector<Eigen::Affine3f >& poses)
{
	assert(m_fd != NULL);
	
	m_pose_votes.clear();

	m_fd->getPose(m_model_id, poses, m_pose_votes);

	return true;
}


/*!
Returns the pose after ICP was applied. 
@return a 4x4 matrix in homogenous coordinates with the pose. 
*/
Eigen::Matrix4f  TrackingExpertRegistration::getICPPose(void)
{
	assert(m_icp != NULL);

	return m_icp->Rt();

}


/*!
Enable extra debug messages. 
@param verbose - output extra debug messages if true. 
*/
bool TrackingExpertRegistration::setVerbose(bool verbose)
{
	assert(m_fd != NULL);
//	assert(m_icp != NULL);

	return m_fd->setVerbose(verbose);// & m_icp->setVerbose(verbose);
}


/*!
Set the parameters for the extraction tool.
@param params - a struct containing some parameters. 
*/
bool TrackingExpertRegistration::setParams(CPFParams params)
{
	assert(m_fd != NULL);

	m_fd->setParams(params);

	return true;
}


/*!
Return the RenderHelper object to access the data. 
Note that no data is produced if the helper is empty. 
*/
CPFRenderHelpers& TrackingExpertRegistration::getRenderHelper(void)
{
	assert(m_fd != NULL);

	return m_fd->getRenderHelper();
}

/*!
For visual debugging. 
Return the point pairs of a pose cluster. The point pairs indicate the descriptor locations used to 
calculate the final pose. 
@param cluster_id - the cluster id. Note that the cluster are sorted. Cluster 0 has the highest number of votes. 
@param cluster_point_pairs - reference to store the cluster point pairs as <reference model point, scene point>.
*/
bool TrackingExpertRegistration::getPoseClusterPairs(const int cluster_id, std::vector< std::pair<int, int> >& cluster_point_pairs)
{
	assert(m_fd != NULL);

	return m_fd->getPoseClusterPairs(cluster_id, cluster_point_pairs);
}


/*
Return the number of pose clusters for the current model
*/
int  TrackingExpertRegistration::getNumPoseClusters(void)
{
	if(m_model_id == -1) return 0;

	return m_fd->getNumPoseClusters(m_model_id);
}



/*!
For visual debugging. 
Return the discretized curvature values for the model.
*/
bool TrackingExpertRegistration::getModelCurvature( std::vector<uint32_t>& model_cu )
{
	assert(m_fd != NULL);

	if(m_model_id == -1 ) return false;

	return m_fd->getModelCurvature(m_model_id, model_cu);
}

/*!
For visual debugging. 
Return the discretized curvature values for the scene.
*/
bool TrackingExpertRegistration::getSceneCurvature(std::vector<uint32_t>& scene_cu)
{
	assert(m_fd != NULL);

	return m_fd->getSceneCurvature(scene_cu);
}