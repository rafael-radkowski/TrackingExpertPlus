#pragma once

/*
--------------------------------------------------------------------------------
Last edits:

Mar 08, 2021, WB
- Fixed conversion of ICP matrix from Matrix4f to Mat4
*/

// STL
#include <iostream>
#include <string>
#include <Windows.h>
#include <fstream>
#include <algorithm>

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
#include "GLLineRenderer.h"
#include "TrackingExpertRegistration.h"
#include "ColorCoder.h"
#include "TrackingExpertParams.h"
#include "FilterTypes.h"
#include "MatrixConv.h"

#ifdef _WITH_AZURE_KINECT // set via cmake
#include "KinectAzureCaptureDevice.h"  // the camera
#include "PointCloudProducer.h"
#define _WITH_PRODUCER
#endif

namespace texpert{


class TrackingExpertDemo
{
private:
	
	typedef enum {
		AR,
		PC
	}SceneType;

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
	bool setCamera(CaptureDeviceType type = CaptureDeviceType::None);

	/*!
	Load a scene model from a file instead of from a camera.
	Note that the file needs to be a point cloud file. 
	@param path_and_filename - path and file to the scene model.
	@return true - if the scene model was loaded. 
	*/
	bool loadScene(std::string path_and_filename);


	/*!
	Load and label a model file. Needs to be a point cloud file.
	@param pc_path_and_filename - path and file to the model.
	@param label - label the model with a string. 
	@return true - if the model could be found and loaded. 
	*/
	bool loadModel(std::string pc_path_and_filename, std::string label);

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

private:

	/*
	Init the class
	*/
	void init(void);

	/*
	To be passed to the renderer to draw the content. 
	*/
	void render_fcn(glm::mat4 proj_matrix, glm::mat4 view_matrix);

	/*
	Keyboard callback for the renderer
	*/
	void keyboard_cb(int key, int action);

	/*
	Inits all the graphical content. 
	*/
	void initGfx(void);

	/*
	Render the point cloud sceen and show the point cloud content
	*/
	void renderPointCloudScene(glm::mat4 pm, glm::mat4 vm);

	/*
	Render the AR scene and show AR content. 
	*/
	void renderARScene(glm::mat4 pm, glm::mat4 vm);

	/*
	Track the reference object in the scene. 
	*/
	void trackObject(void);


	/*
	Allows one to enable or disable the tracking functionality.
	@param enable, true starts detection and registration
	*/
	void enableTracking(bool enable = true);


	/*
	Update camera data
	*/
	void updateCamera(void);

	/*
	Get a single frame from the camera.
	*/
	void grabSingleFrame(void);


	// debug rendering functions
	void renderMatches(void);
	void renderVotes(void);
	void renderPoseCluster(void);
	void updateRenderData(void);
	void updateRenderCluster(void);
	void renderCurvatures(void);
	void updateCurvatures(void);
	void renderNormalVectors(void);
	void upderRenderPose(void);


	//--------------------------------------------------------------------
	// Graphics stuff

	isu_ar::GLViewer*	m_window;

	// OpenGL point cloud objects showing the point cloud
	isu_ar::GLPointCloudRenderer* gl_camera_point_cloud;
	isu_ar::GLPointCloudRenderer* gl_reference_point_cloud;
	isu_ar::GLPointCloudRenderer* gl_reference_eval; // evaluation point cloud for visual evaluation

	// OpenGL line renderer to show matches, etc. 
	isu_ar::GLLineRenderer*  gl_matches;
	isu_ar::GLLineRenderer*  gl_best_votes;
	isu_ar::GLLineRenderer*  gl_best_pose;

	bool					m_enable_matching_renderer;
	bool					m_enable_best_votes_renderer;
	bool					m_enable_best_pose_renderer;
	bool					m_render_curvatures;
	bool					m_render_normals;
	bool					m_enable_filter;
	int						m_current_debug_point;
	int						m_current_debug_cluster;


	// vectors to indicate the matching paris.
	std::vector<std::pair<int, int>>		 match_pair_ids;
	std::vector<std::pair<int, int>>		 vote_pair_ids;
	std::vector<std::pair<int, int>>		 pose_ids;

	int					m_window_width;
	int					m_window_height;


	//--------------------------------------------------------------------
	// Input

	CaptureDeviceType	m_camera_type;
	std::string			m_camera_file;
	std::string			m_model_file;

	// Helper variables to set the point cloud sampling. 
	SamplingParam		sampling_param;
	SamplingMethod		sampling_method;
	FilterParams		m_filter_param;
	FilterMethod		m_filter_method;

	// switch between AR and point cloud scene. 
	SceneType			m_scene_type;


	// instance of a structure core camera 
	texpert::ICaptureDevice* m_camera;

#ifdef _WITH_PRODUCER
	PointCloudProducer*		 m_producer;
#endif
	SamplingParam		m_producer_param;
	//--------------------------------------------------------------------
	// Detetction and registration

	// point cloud data
	PointCloud			m_pc_camera_raw;
	PointCloud			m_pc_camera;

	// The reference point cloud.
	// The first one is the point cloud for all ICP purposes.
	// The second one is the raw point cloud as loaded. 
	PointCloud			pc_ref;
	PointCloud			pc_ref_as_loaded;

	// object detection and registration 
	TrackingExpertRegistration*	m_reg;

	// Matrix conversion helper
	MatrixConv*			m_conv;

	bool				m_new_scene;
	bool				m_enable_tracking;
	bool				m_update_camera;

	//--------------------------------------------------------------------
	// Helper params

	bool				m_verbose;
};

}//namespace texpert{