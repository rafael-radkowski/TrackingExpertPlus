#include "PointCloudManager.h"


PointCloudManager* PointCloudManager::m_instance = nullptr;

//static 
PointCloudManager* PointCloudManager::getInstance()
{
	if (m_instance == nullptr) {
		m_instance = new PointCloudManager();
	}
	return m_instance;
}

PointCloudManager::~PointCloudManager()
{

}


/*
Public constructor
*/
PointCloudManager::PointCloudManager()
{
	pc_update = false;
}



/*
Return a reference to the camera point cloud
@return reference to a object of type PointCloud.
*/
PointCloud& PointCloudManager::getCameraPC(void)
{
	std::lock_guard<std::mutex> lock(m_pc_camera_mutex);

	return m_pc_camera;
}

/*
Return a reference to the camera point cloud
@return reference to a object of type PointCloud.
*/
PointCloud& PointCloudManager::getCameraRawPC(void)
{
	std::lock_guard<std::mutex> lock(m_pc_camera_raw_mutex);

	return m_pc_camera_raw;
}

/*
Return a reference to the reference point cloud as loaded
@return reference to a object of type PointCloud.
*/
PointCloud& PointCloudManager::getRefereceRawPC(void)
{
	std::lock_guard<std::mutex> lock(pc_ref_as_loaded_mutex);

	return pc_ref_as_loaded;
}

/*
Return a reference to the camera point cloud as processed.
@return reference to a object of type PointCloud.
*/
PointCloud& PointCloudManager::getReferecePC(void)
{
	std::lock_guard<std::mutex> lock(pc_ref_mutex);

	return pc_ref;
}

/*
Clear the reference point clouds
*/
void PointCloudManager::clearReferencePC(void)
{
	std::lock_guard<std::mutex> lock(pc_ref_mutex);

	if (pc_ref_as_loaded.size() > 0) {
		pc_ref_as_loaded.points.clear();
		pc_ref_as_loaded.normals.clear();
	}
}


/*
Indicate a pose update
*/
void PointCloudManager::updatePose(void)
{
	pc_update = true;
}


/*
Query the pose update variable
*/
bool PointCloudManager::getUpdatePose(void)
{
	bool ret = pc_update;
	if(pc_update)
		pc_update = false;

	return ret;
}


