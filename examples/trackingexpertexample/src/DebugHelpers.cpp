

#include "DebugHelpers.h"


/*
Constructor
*/
DebugHelpers::DebugHelpers()
{
	_render_cam_curvatures = false;
	_render_model_curvature = false;
}

/*
Destructor
*/
DebugHelpers::~DebugHelpers()
{

}



/*
Enable or disabel a particular render helper function.
@param type - the visual widget type.
@param enable - true enables the renderer, false disables the renderer. Default is true
*/
void DebugHelpers::enableRenderer(DebugHelperType type, bool enable)
{
	switch(type){
	case CAM_CURVATURE:
		if(enable) _render_cam_curvatures = true;
		else _render_cam_curvatures = false;
	break;
	case MODEL_CURVATURE:
		if (enable) _render_model_curvature = true;
		else _render_model_curvature = false;
	break;
	}
}


/*
Render the curvature values as false colors.
The function access the curvature values stored in the point cloud manager and renders those.
@paraem gl_camera_point_cloud - pointer to the camera point cloud.
*/
bool DebugHelpers::renderCameraCurvatures(isu_ar::GLPointCloudRenderer* gl_camera_point_cloud)
{
	if(gl_camera_point_cloud == NULL)
		return false;

	if(_render_cam_curvatures)
		return true;

	PointCloudManager* pcm = PointCloudManager::getInstance();
	
	if(pcm->getCameraCurvatures().size() == 0)
		return false;

	std::vector<glm::vec3> colors;

	GLColorCoder::CPF2Color(pcm->getCameraCurvatures(), colors);

	gl_camera_point_cloud->setPointColors(colors);


	return true;
}