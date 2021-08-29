#include "CPFDetect.h"


/*
Constructor
*/
CPFDetect::CPFDetect()
{
	m_model_data = NULL;
	m_scene_data = NULL;
	m_verbose = true;
	m_verbose_level = 1;
}

/*
Destructor
*/
CPFDetect::~CPFDetect()
{
	delete m_model_data;
	delete m_scene_data;
}


/*!
Add the reference point set, a point cloud model of an object to detect and track.
Note that adding the model will also start the descriptor extraction process.
@param points - a reference to a point cloud model containing points and normal vectors.
@param label - an object label.
@return an model id as integer or -1 if the model was rejected.
*/
int CPFDetect::addModel(PointCloud& points, std::string label)
{
	if (points.size() == 0) return -1;

	if (points.points.size() != points.normals.size()) {
		std::cout << "[ERROR] - CPFDetect: points size " << points.points.size() << " != " << points.normals.size() << " for " << label << ".\n" << std::endl;
		return -1;
	}

	// create a new model instance. 
	m_model_data =  CPFDataDB::GetInstance()->createModelInstance(points, label);

	if (m_verbose) {
		std::cout << "[INFO] - CPFMatchingExp: start extracting descriptors from " << label << " with " << points.size() << " points." << std::endl;
	}


	//--------------------------------------------------------
	// Start calculating descriptors

	CPFMatching::CalculateDescriptors(*m_model_data, m_params);


	if (m_verbose) {
		std::cout << "[INFO] - CPFMatchingExp: finished extraction of " << m_model_data->getDescriptor().size() << " descriptors for  " << label << "." << std::endl;
	}

	return 0;
}


/*!
Set a scene model or a point cloud from a camera as PointCloud object.
Note that adding the model will also start the descriptor extraction process.
@param points - point and normal vectors.
@return true if the scene was accepted.
*/
bool CPFDetect::setScene(PointCloud& points)
{
	if (points.size() == 0) return false;

	if (points.points.size() != points.normals.size()) {
		std::cout << "[ERROR] - CPFMatchingExp: scene point size != normals size: " << points.points.size() << " != " << points.normals.size() << "." << std::endl;
	}

	// dataset to store the scene data. 
	// note that the scene data instance stores the reference to the point cloud and is automatically "up-to-date"
	// since it points to the same memory for points. 
	if(m_scene_data == NULL)
		m_scene_data = CPFDataDB::GetInstance()->createSceneInstance(points); // create a new instance if none is available


	if (m_verbose && m_verbose_level == 2) {
		std::cout << "[INFO] - CPFMatchingExp: start extracting scene descriptors for " << points.size() << " points." << std::endl;
	}

	//--------------------------------------------------------
	// Start calculating descriptors

	CPFMatching::CalculateDescriptors(*m_scene_data, m_params);
	

	if (m_verbose && m_verbose_level == 2) {
		std::cout << "[INFO] - CPFMatchingExp: finished extraction of " << m_scene_data->getDescriptor().size() << " scene descriptors for." << std::endl;
	}
	return true;
}


/*!
Start the detection and pose estimation process for a model with the id model_id.
Invoking this function will start descriptor matching for model_id, voting, and pose clustering.
@para model_id - the id of the model to be detected (the int returned from addModel().
*/
bool CPFDetect::match(int model_id)
{
	if(m_scene_data == NULL || m_model_data == NULL)
		return false;
	if (model_id != 0) {
		std::cout << "[ERROR] - Selected model id " << model_id << " for matching does not exist.";
		return false;
	}
	if (m_model_data->size() <= 0) {
		std::cout << "[ERROR] - No model data set. Add a model first." << std::endl;
		return false;
	}
	if (m_scene_data->size() <= 0) {
		std::cout << "[ERROR] - No scene set. Set a scene model first." << std::endl;
		return false;
	}


	// matching
	CPFMatching::MatchDescriptors(*m_model_data, *m_scene_data, m_matching_results, m_params);


	// clustering


	// align and update pose


	return true;
}