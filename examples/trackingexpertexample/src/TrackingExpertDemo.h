#pragma once

/*
The class implements the main file for a TrackingExpert+ demo. 

It is the primary interface to the outside classes and is responsible for organizing the 
data exchange between the graphical interface and the tracking thread. 

Class responsibilities:
- Organizing the interface between graphics and rendering
- Providing access to the primary graphic and tracking features and functions. 
- Process control: starts, pauses, and stops the tracking process. 


Rafael Radkowski
Aug 2021
radkowski.dev@gmail.com
MIT License
--------------------------------------------------------------------------------
Last edits:


*/

// STL
#include <iostream>
#include <string>
#include <Windows.h>
#include <fstream>
#include <algorithm>
#include <thread>

// GLM include files
#define GLM_FORCE_INLINE
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtx/transform.hpp>  // transformation
#include <glm/gtx/quaternion.hpp> // quaternions

// eigen
#include <Eigen\Dense>

// TrackingExpert
#include "trackingx.h"
#include "graphicsx.h"
#include "ICaptureDeviceTypes.h"
#include "ReaderWriterUtil.h"
#include "TrackingExpertParams.h"
#include "KinectAzureCaptureDevice.h"
#include "AzureKinectFromMKV.h"
#include "PointCloudManager.h"
#include "MainRenderProcess.h"
#include "MainTrackingProcess.h"



namespace texpert{


class TrackingExpertDemo
{
	

public:

	/*!
	Constructor
	*/
	TrackingExpertDemo();
	~TrackingExpertDemo();

	/*!
	Set a camera type to use or none, if the data comes from a file.
	@param type - camera type of type CaptureDeviceType.
	@return true 
	*/
	bool setSourceCamera(CaptureDeviceType type = CaptureDeviceType::None);


	/*!
	Load a scene model from a file instead of from a camera or video.
	Note that the file needs to be a point cloud file. 
	@param path_and_filename - path and file to the scene model.
	@return true - if the scene model was loaded. 
	*/
	bool setSourceScene(std::string path_and_filename);


	/*!
	Load and label a model file. Needs to be a point cloud file.
	@param pc_path_and_filename - path and file to the model.
	@param label - label the model with a string. 
	@return true - if the model could be found and loaded. 
	*/
	bool loadReferenceModel(std::string pc_path_and_filename, std::string label);

	/*!
	Start the application. This is the last thing one should do 
	since the function is blocking and will only return after the window closes.
	*/
	bool run(void);


	/*
	Enable more outputs
	@param verbose - true enables more debug outputs. 
	*/
	bool setVerbose(bool verbose);



	/*!
	Set the application parameters
	@param params - struct params of type TEParams. 
	*/
	bool setParams(TEParams params);


	/*
	Reset the reference model to the state as loaded
	*/
	void resetReferenceModel(void);

		

private:

	/*
	Init the class
	*/
	void init(void);


	/*
	Keyboard callback for the renderer
	*/
	void keyboard_cb(int key, int action);



	/*
	Allows one to enable or disable the tracking functionality.
	@param enable, true starts detection and registration
	*/
	void enableTracking(bool enable = true);


	/*
	Update camera data.
	This function runs in a thread and updates the 
	camera if required. 
	*/
	void autoProcessFrame(void);


	/*
	Init the Azure Kinect video as source. 
	Note that the camera file name needs to be set vis setParams(), the parameter
	TEParams::input_mkv needs to hold the path and file. 
	*/
	bool initAzureKinectMKV(void);




	//--------------------------------------------------------------------
	// Graphics stuff

	MainRenderProcess*			_renderer;


	//--------------------------------------------------------------------
	// Input

	CaptureDeviceType	m_camera_type;
	std::string			m_camera_file;  // in case the input data comes from a file. 
	std::string			m_model_file;

	// Helper variables to set the point cloud sampling. 
	SamplingParam		sampling_param;
	SamplingMethod		sampling_method;


	SamplingParam		m_producer_param;
	FilterParams		m_filter_param;
	FilterMethod		m_filter_method;

	// instance of a structure core camera 
	texpert::ICaptureDevice* m_camera;

	// camera thread
	std::thread				m_camera_updates;


	// Data manager class
	PointCloudManager*		_dm;

	//--------------------------------------------------------------------
	// Detetction and registration
	MainTrackingProcess*	_tracking;


	//--------------------------------------------------------------------
	// Helper params

	bool				m_verbose;
	bool				m_update_camera;
	bool				m_is_running;

	//--------------------------------------------------------------------
	// keyboard helper params
	bool				m_render_scene_normals;
	bool				m_render_ref_normals;
	bool				m_enable_tracking;

};

}//namespace texpert{