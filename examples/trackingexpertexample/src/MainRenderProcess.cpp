#include "MainRenderProcess.h"


MainRenderProcess* MainRenderProcess::m_instance = nullptr;


#ifdef _WITH_AZURE_OUTPUT
// crude and ugly debug helper. Remove!
extern texpert::ICaptureDevice* g_camera;
int g_counter = 0;
#endif

//static 
MainRenderProcess* MainRenderProcess::getInstance()
{
	if (m_instance == nullptr) {
		
		m_instance = new MainRenderProcess();
	}
	return m_instance;
}

MainRenderProcess::~MainRenderProcess()
{
	delete _helper;
	delete gl_camera_point_cloud;
	delete gl_reference_point_cloud;
	delete gl_reference_eval;

	// render lines between matches
	delete gl_matches;
	delete gl_best_votes;
	delete gl_best_pose;

}

MainRenderProcess::MainRenderProcess()
{
	
	// init the opengl window
	glm::mat4 vm = glm::lookAt(glm::vec3(0.0f, 0.0f, -0.5f), glm::vec3(0.0f, 0.0f, 0.5), glm::vec3(0.0f, 1.0f, 0.0f));
	m_window = new isu_ar::GLViewer();
	m_window->create(1280, 1280, "TrackingExpert+ Demo");
	m_window->addRenderFcn(std::bind(&MainRenderProcess::render_fcn, this, _1, _2));
	//m_window->addKeyboardCallback(std::bind(&TrackingExpertDemo::keyboard_cb, this, _1, _2));
	m_window->setViewMatrix(vm);
	m_window->setClearColor(glm::vec4(0, 0, 0, 1));

	m_update_camera_pc = false;


	_dm = PointCloudManager::getInstance();
	_helper = new DebugHelpers();
}


/*
initialize the grapics system and processes.
*/
void MainRenderProcess::initGfx(void)
{
	// fetch an instance of the data manager. 
	PointCloudManager* dm = PointCloudManager::getInstance();
	if (dm == NULL) {
		std::cout << "[ERROR] -  Did not obtain DataManager pointer." << std::endl;
	}

	// camera point cloud
	gl_camera_point_cloud = new	isu_ar::GLPointCloudRenderer(dm->getCameraPC().points, dm->getCameraPC().normals); 
	gl_camera_point_cloud->setPointColor(glm::vec3(1.0, 0.0, 0.0));
	gl_camera_point_cloud->setNormalColor(glm::vec3(0.8, 0.5, 0.0));
	gl_camera_point_cloud->setNormalGfxLength(0.05f);
	gl_camera_point_cloud->enableAutoUpdate(false);
	gl_camera_point_cloud->enablePointRendering(true);

	// reference point cloud
	gl_reference_point_cloud = new	isu_ar::GLPointCloudRenderer(dm->getReferecePC().points, dm->getReferecePC().normals); 
	gl_reference_point_cloud->setPointColor(glm::vec3(0.0, 1.0, 0.0));
	gl_reference_point_cloud->setNormalColor(glm::vec3(0.0, 0.8, 0.8));
	gl_reference_point_cloud->setNormalGfxLength(0.05f);

	// point cloud for evaluation.
	gl_reference_eval = new	isu_ar::GLPointCloudRenderer(dm->getReferecePC().points, dm->getReferecePC().normals); 
	gl_reference_eval->setPointColor(glm::vec3(1.0, 1.0, 0.0));
	gl_reference_eval->setNormalColor(glm::vec3(0.5, 0.8, 0.8));
	gl_reference_eval->setNormalGfxLength(0.05f);


	//-------------------------------------------------------------------
	// Render lines

	gl_matches = new isu_ar::GLLineRenderer(dm->getReferecePC().points, dm->getCameraPC().points, match_pair_ids);  //pc_ref.points, m_pc_camera.points
	gl_matches->setLineColor(glm::vec3(1.0, 0.0, 1.0));
	gl_matches->enableRenderer(false);

	gl_best_votes = new isu_ar::GLLineRenderer(dm->getReferecePC().points, dm->getCameraPC().points, vote_pair_ids);//pc_ref.points, m_pc_camera.points
	gl_best_votes->setLineWidth(5.0);
	gl_best_votes->setLineColor(glm::vec3(1.0, 1.0, 0.0));
	gl_best_votes->enableRenderer(false);

	gl_best_pose = new isu_ar::GLLineRenderer(dm->getReferecePC().points, dm->getCameraPC().points, pose_ids); // pc_ref.points, m_pc_camera.points
	gl_best_pose->setLineWidth(5.0);
	gl_best_pose->setLineColor(glm::vec3(0.0, 1.0, 0.2));
	gl_best_votes->enableRenderer(false);

}


/*
Start the rendern loop
*/
void MainRenderProcess::start(void)
{
	// start the viewer
	m_window->start();
}


/*
To be passed to the renderer to draw the content.
*/
void MainRenderProcess::render_fcn(glm::mat4 pm, glm::mat4 vm)
{
	// update the camera data
	//updateCamera();

	if (m_update_camera_pc) {
		m_update_camera_pc = false;
		gl_camera_point_cloud->updatePoints();
	}


	if(_dm->getUpdatePose()){
		Eigen::Matrix4f pose = _dm->getReferecePC().pose;
		glm::mat4 glm_out;
		MatrixConv* mc = MatrixConv::getInstance();
		mc->Matrix4f2Mat4(pose, glm_out);
		//gl_reference_eval->setModelmatrix(glm_out);

		gl_reference_point_cloud->updatePoints();
	}


	// update the poses if a new scene model is available.

	renderAndUpdateHelpers( pm, vm);
	renderPointCloudScene(pm, vm);



#ifdef _WITH_AZURE_OUTPUT
	// crude and ugly debug helper. Remove!
	if (g_camera != NULL)
	{
		if (g_counter < 30) {
			g_counter++;
			return;
		}
		cv::Mat rgb_frame;
		cv::Mat depth_frame;
		g_camera->getDepthFrame(depth_frame);
		cv::Mat img_depth_col = depth_frame.clone();
		img_depth_col.convertTo(img_depth_col, CV_8U, 255.0 / 5000.0, 0.0);
		cv::imshow("Depth", img_depth_col);
		

		g_camera->getRGBFrame(rgb_frame);
		cv::imshow("RGB", rgb_frame);
		
		
		cv::waitKey(1);
	}
#endif
}


/*
Render the point cloud sceen and show the point cloud content
*/
void MainRenderProcess::renderPointCloudScene(glm::mat4 pm, glm::mat4 vm)
{
#ifdef _WITH_SEQUENTIAL
	for (auto a : _render_callbacks) {
		a();
	}
#endif

	// draw the camera point cloud
	gl_camera_point_cloud->draw(pm, vm);
	gl_reference_point_cloud->draw(pm, vm);
	gl_reference_eval->draw(pm, vm);

	// render lines between matches
	//gl_matches->draw(pm, vm);
	//gl_best_votes->draw(pm, vm);
	gl_best_pose->draw(pm, vm);
//
}



/*
The function renders all graphical helpers in the scene.
*/
void MainRenderProcess::renderAndUpdateHelpers(glm::mat4 pm, glm::mat4 vm)
{
	assert(_helper != NULL);

	_helper->renderCameraCurvatures(gl_camera_point_cloud);
}



/*
Update the data for the renderer
*/
void MainRenderProcess::setUpdate(void)
{
	m_update_camera_pc = true;
}



/*
Add a keyboard function to the existing window.
@param fc -  function pointer to the keyboard function.
*/
void MainRenderProcess::setKeyboardFcn(std::function<void(int, int)> fc)
{
	m_window->addKeyboardCallback(fc);
}


/*
Enable or disable a render feature such as normal rendering, etc
@param f - the feature of type RenderFeature (see the enum for details.)
@param enable - true enables the feature, false disables it.
*/
void MainRenderProcess::setRenderFeature(RenderFeature f, bool enable)
{

	switch (f) {
		case PointsScene:
			if(gl_camera_point_cloud) gl_camera_point_cloud->enablePointRendering(enable);
			break;
		case PointsRef:
			if (gl_reference_point_cloud) gl_reference_point_cloud->enablePointRendering(enable);
			break;
		case NormalsScene:
			if (gl_camera_point_cloud) gl_camera_point_cloud->enableNormalRendering(enable);
			break;
		case NormalsRef:
			if (gl_reference_point_cloud) gl_reference_point_cloud->enableNormalRendering(enable);
			break;
		case CurvScene:
			if (gl_reference_point_cloud) _helper->enableRenderer(DebugHelpers::CAM_CURVATURE, enable);
			break;
		default:
			break;
	}
	
}

#ifdef _WITH_SEQUENTIAL
void MainRenderProcess::addRenderFunction(std::function<void(void)> f)
{
	_render_callbacks.push_back(f);
}
#endif