/*
The class PointCloudManager maintains and manages access to the point cloud objects that
are used as part of the application. 
The most relevant objects are:

- m_pc_camera_raw: the raw point cloud when loaded from a data file. The object is not used
	when the data comes from a camera.
- m_pc_camera: the projected point cloud as a result of the depth image projection + sampling. 
- pc_ref_as_loaded: the reference point cloud as loaded from a file
- pc_ref: the reference point cloud during processing (sampling, tracking, et.)

The class gives access to the point clouds via its apis and protects access via mutexes. 

ToDo: Need to verify the exact mutex functionality. Perhaps the lock is unlocked to early at this point since it is unlocked when out of scope. 

Rafael Radkowski
Aug 2021
radkowski.dev@gmail.com
MIT License
--------------------------------------------------------------------------------
Last edits:


*/
#pragma once
// STL
#include <iostream>
#include <string>
#include <Windows.h>
#include <fstream>
#include <algorithm>
#include <mutex>
#include <thread>

// data
#include "trackingx.h"

class PointCloudManager
{
public:

	/*
	Get an instance of the data manager
	*/
	static PointCloudManager* getInstance();
	~PointCloudManager();

	/*
	Return a reference to the camera point cloud
	@return reference to a object of type PointCloud.
	*/
	PointCloud& getCameraPC(void);

	/*
	Return a reference to the camera point cloud
	@return reference to a object of type PointCloud.
	*/
	PointCloud& getCameraRawPC(void);

	/*
	Return a reference to the reference point cloud as loaded
	@return reference to a object of type PointCloud.
	*/
	PointCloud& getRefereceRawPC(void);

	/*
	Return a reference to the camera point cloud as processed.
	@return reference to a object of type PointCloud.
	*/
	PointCloud& getReferecePC(void);

	/*
	Clear the reference point clouds
	*/
	void clearReferencePC(void);


	//ToDo: updatePose and getUpdatePose are not necessary.
	// Both are here to debug some code and can be remove later. 

	/*
	Indicate a pose update
	*/
	void updatePose(void);


	/*
	Query the pose update variable
	*/
	bool getUpdatePose(void);


	
private:

	/*
	Public constructor
	*/
	PointCloudManager();


	// the instance
	static PointCloudManager*	 m_instance;



	//-------------- Point cloud data ----------

	// camera point cloud
	std::mutex					m_pc_camera_mutex;
	PointCloud					m_pc_camera;
	std::mutex					m_pc_camera_raw_mutex;
	PointCloud					m_pc_camera_raw;

	// The reference point cloud.
	// The first one is the point cloud for all processing purposes.
	// The second one is the raw point cloud as loaded. 
	std::mutex					pc_ref_mutex;
	PointCloud					pc_ref;
	std::mutex					pc_ref_as_loaded_mutex;
	PointCloud					pc_ref_as_loaded;


	bool						pc_update;


};

