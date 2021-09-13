

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

	if(!_render_cam_curvatures){
		gl_camera_point_cloud->setPointColor(glm::vec3(1.0,0.0,0.0));
		return true;
	}

	// get the data db
	CPFDataDB* db = CPFDataDB::GetInstance();

	// grab the scene data instance
	CPFSceneData* sd =  db->getSceneData();
	
	// check that valid. 
	if(sd != NULL){
		if (db->getSceneData()->size() == 0)
			return false;

		std::vector<glm::vec3> colors;

		// convert the curvatures to colors 
		GLColorCoder::CPF2Color(sd->getCurvature(), colors);

		// set the colors for all points. 
		gl_camera_point_cloud->setPointColors(colors);
	}
	
	


	return true;
}



/*
Render the curvatures of the reference model.
@param gl_model_point_cloud - a pointer to the gl render model.
*/
bool DebugHelpers::renderModelCurvatures(isu_ar::GLPointCloudRenderer* gl_model_point_cloud)
{

	if (gl_model_point_cloud == NULL)
		return false;

	if (!_render_model_curvature) {
		gl_model_point_cloud->setPointColor(glm::vec3(0.0, 1.0, 0.0));
		return true;
	}

	// get the data db
	CPFDataDB* db = CPFDataDB::GetInstance();

	// grab the scene data instance
	CPFModelData* sd = db->getModelData("ref_model");

	// check that valid. 
	if (sd != NULL) {
		if (sd->size() == 0)
			return false;

		std::vector<glm::vec3> colors;

		// convert the curvatures to colors 
		GLColorCoder::CPF2Color(sd->getCurvature(), colors);

		// set the colors for all points. 
		gl_model_point_cloud->setPointColors(colors);
	}




	return true;
}


/*
Update lines between the individual points.
*/
bool DebugHelpers::renderMatches(isu_ar::GLLineRenderer* gl_line_renderer)
{
	if (gl_line_renderer == NULL)
		return false;

	// get the data db
	CPFDataDB* db = CPFDataDB::GetInstance();


	gl_line_renderer->updateMatches(db->GetMatchingData()->getMatchingPairs());

}