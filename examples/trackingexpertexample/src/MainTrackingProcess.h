/*
The class MainTrackingProcess is responsible for managing and running the tracking process. 
The example class works with one object only since it uses the TE+ descriptor CPU implementation. 
For more object, use the GPU implementation, which lacks some functional testing. 

Input: 
- A point cloud describing the scene.
- A reference point cloud model representing the object of interest. 

The abridged version:
1) The class detects the object of interest using a feature descriptor-based detector. Here, 
detection means detection by registration. If a subset of a reference point set registers properly with 
the scene point cloud (= up to a user-determined "convidence" value, the object is considered detected).

2) A pose is estimated using matching point pairs and a clustering algorithm. 

3) The pose is refined using IPC.

4) The reference point set position and orientation (the values of all points) are updated. 

And the process restarts at 1. 

Responsibilitities
- Detect object in a point cloud
- Register a reference object to the point cloud.
- Estimate the pose of the object. 
- Track the pose of the object. 
- Maintain different states (idle, lost, detected, tracking).
- Notify the renderer about all changes of the model (the task should be delegated to somebody else in the future).
- Maintain and trigger the PointCloudProducer instance, which is the starting point for all tracking processes. 
	The trigger fetches a new image from the camera. 


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
#include <functional>

// GLM include files
#define GLM_FORCE_INLINE
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtx/transform.hpp>  // transformation
#include <glm/gtx/quaternion.hpp> // quaternions

// local
#include "trackingx.h"
#include "PointCloudManager.h"
#include "PointCloudProducer.h"

class MainTrackingProcess
{
public:

	/*
	Get an instance of the class.
	@return Instance of the class
	*/
	static MainTrackingProcess* getInstance();
	~MainTrackingProcess();


	/*
	Init the tracking procss and set the camera object
	*/
	void init(texpert::ICaptureDevice* camera = NULL);

	/*
	Grab a new frame and process the frame
	including all tracking steps. 
	*/
	void process(void);


	/*
	Set the sampling params for the point cloud producer
	@params params - parameter varialb.s
	*/
	void setSamplingParams(SamplingParam params);

	/*
	Set the parameters for the tracking filter
	*/
	void setFilterParams(FilterMethod method, FilterParams params);


private:

	/*
	Private constructor
	*/
	MainTrackingProcess();


	static MainTrackingProcess* m_instance;


	//---------------------------------------------------

		// Point cloud producer
	texpert::PointCloudProducer*		m_producer;
	SamplingParam						m_producer_param;
	FilterParams						m_filter_param;
	FilterMethod						m_filter_method;


};

MainTrackingProcess* MainTrackingProcess::m_instance = nullptr;
