#pragma once
/*
@class DebugHelpers


The class is responsible for rendering visual helpers or updating relevant objects so 
that those visualize data that supports debugging. 

Rafael Radkowski
Aug 27, 2021

MIT License
-------------------------------------------------------------------------------------------------
Last edited:


*/

// stl
#include <iostream>
#include <string>



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
#include "MatrixConv.h"

// feature detector data
#include "CPFData.h"

// ToDo: Remove when ready
#include "DebugSwitches.h"

// curvature color encoder
#include "GLColorCoder.h"


class DebugHelpers
{
public:

	typedef enum{
		CAM_CURVATURE,
		MODEL_CURVATURE
	}DebugHelperType;

	/*
	Constructor
	*/
	DebugHelpers();

	/*
	Destructor
	*/
	~DebugHelpers();



	/*
	Enable or disabel a particular render helper function.
	@param type - the visual widget type.
	@param enable - true enables the renderer, false disables the renderer. Default is true
	*/
	void enableRenderer( DebugHelperType type, bool enable = true);


	/*
	Render the curvature values as false colors. 
	The function access the curvature values stored in the point cloud manager and renders those. 
	@paraem gl_camera_point_cloud - pointer to the camera point cloud. 
	*/
	bool renderCameraCurvatures(isu_ar::GLPointCloudRenderer* gl_camera_point_cloud);

private:


	bool	_render_cam_curvatures;
	bool	_render_model_curvature;
};