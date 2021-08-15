/*

The class MainRenderProcess implements the graphics processes, graphics updates, and keeps an instance of the OpenGL
Window. 

Note that the OpenGL window comes with its own thread. So this class does not start or run a thread on its own. 

The class implements a singleton pattern.

Class responsibilities:
- Keeping an instance of the opengl window
- Maintaining the window update loop.
- Managing the graphics states/ 


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
#include "graphicsx.h"
#include "GLLineRenderer.h"
#include "PointCloudManager.h"

// ToDo: Remove when ready
#include "DebugSwitches.h"

class MainRenderProcess
{
public:

	typedef enum
	{
		PointsScene,
		PointsRef,
		NormalsScene,
		NormalsRef
	}RenderFeature;

	/*
	Get an instance of the class.
	@return Instance of the class
	*/
	static MainRenderProcess* getInstance();
	~MainRenderProcess();


	/*
	initialize the grapics system and processes. 
	*/
	void initGfx(void);


	/*
	Start the rendern loop
	*/
	void start(void);


	/*
	Update the data for the renderer.
	The function triggers the point cloud to read new data 
	before the next render loops starts. 
	*/
	void setUpdate(void);


	/*
	Add a keyboard function to the existing window.
	@param fc -  function pointer to the keyboard function. 
	*/
	void setKeyboardFcn(std::function<void(int, int)> fc);


	/*
	Enable or disable a render feature such as normal rendering, etc
	@param f - the feature of type RenderFeature (see the enum for details.)
	@param enable - true enables the feature, false disables it. 
	*/
	void setRenderFeature(RenderFeature f, bool enable);


private:

	/*
	Private constructor
	*/
	MainRenderProcess();


	static MainRenderProcess* m_instance;



	/*
	The render function
	To be passed to the renderer to draw the content.
	*/
	void render_fcn(glm::mat4 pm, glm::mat4 vm);


	/*
	Render the point cloud sceen and show the point cloud content
	*/
	void renderPointCloudScene(glm::mat4 pm, glm::mat4 vm);


	//--------------------------------------------------------------------
	// Graphics stuff

	isu_ar::GLViewer* m_window;

	// OpenGL point cloud objects showing the point cloud
	isu_ar::GLPointCloudRenderer* gl_camera_point_cloud;
	isu_ar::GLPointCloudRenderer* gl_reference_point_cloud;
	isu_ar::GLPointCloudRenderer* gl_reference_eval; // evaluation point cloud for visual evaluation

	// OpenGL line renderer to show matches, etc. 
	isu_ar::GLLineRenderer*		gl_matches;
	isu_ar::GLLineRenderer*		gl_best_votes;
	isu_ar::GLLineRenderer*		gl_best_pose;

	// vectors to indicate the matching paris.
	std::vector<std::pair<int, int>>		 match_pair_ids;
	std::vector<std::pair<int, int>>		 vote_pair_ids;
	std::vector<std::pair<int, int>>		 pose_ids;

	// update the camera point cloud
	bool					m_update_camera_pc;


	int						m_window_width;
	int						m_window_height;

};

MainRenderProcess* MainRenderProcess::m_instance = nullptr;