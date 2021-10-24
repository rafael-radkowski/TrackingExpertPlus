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
#include <cassert>

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
#include "ICP.h"
#include "CPFMatchingExp.h"
#include "CPFMatchingExpGPU.h"
#include "TrackingExpertParams.h"
#include "CPFDetect.h"

#include "DebugSwitches.h"


using namespace texpert;
using namespace texpert_experimental;

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
		Add a reference model to be detected and tracked.
		@param model - a point cloud reference model of type PointCloud
		@param label - a string containing the label. 
		@return true, if the model was set correctly. 
	*/
	bool addReferenceModel(PointCloud& model, std::string label);

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


	/*
	Enable or disable object tracking functionality.
	Note that "tracking" refers here to a single-shot function. 
	Tracking via multiple frames is realized via a Kalman filter. 
	@param enable - true enables tracking and false disables tracking. 
	*/
	void enableTracking(bool enable = true);


	/*
	Process one step only
	*/
	void step(void);

	/*
	Return the current pose. 
	This is the pose for the current frame after running
	process() once. 
	*/
	Eigen::Matrix4f  getCurrentPose(void);



private:

	/*!
	Return the poses for a particular model.
	The function returns the 12 best hits by default. Note that this can be reduced to only 1 or so.
	@param model_id - the model id of the object to track as int.
	@param poses - vector with the poses
	@param pose_votes - vector with the pose votes.
	*/
	bool getPose(std::vector<Eigen::Affine3f >& poses);



	/*!
	Run the idel state operations;
	*/
	void runIdle(void);

	/*!
	Run the detect state operations;
	*/
	void runDetect(void);

	/*!
	Run the registration state operations;
	*/
	void runRegistration(void);

	/*!
	Run the tracking state operations;
	*/
	void runTracking(void);


	/*
	Update all debug helpers if those are enabled. 
	*/
	void updateHelpers(void);


	/*
	Private constructor
	*/
	MainTrackingProcess();


	static MainTrackingProcess* m_instance;
	
	//---------------------------------------------------

	// data manager
	PointCloudManager*					_dm;


	// the tracking states
	typedef enum {
		IDLE,
		DETECT,
		REGISTRATION,
		TRACKING
	}State;

	State								m_tracking_state;

	//---------------------------------------------------

	// Point cloud producer
	texpert::PointCloudProducer*		m_producer;
	SamplingParam						m_producer_param;
	FilterParams						m_filter_param;
	FilterMethod						m_filter_method;

	
	//---------------------------------------------------
	// Tracking and registration.

	bool								m_enable_tracking;
	// ICP
	ICP*								m_icp;
	PointCloud							m_model_pc;



	// feature detector and matching
	//ICPFMatching*						m_fd;
	//CPFParams							m_fd_params;

	CPFDetect*							m_detect;

	int									m_model_id;
	std::vector<int>					m_pose_votes;
	float								m_rms;


	// the global pose of the model 
	Eigen::Matrix4f						m_model_pose;
};


