#include "TrackingExpertDemo.h"
#include "CamPose.h"

using namespace texpert;

TrackingExpertDemo::TrackingExpertDemo()
{
	init();
}


TrackingExpertDemo::~TrackingExpertDemo()
{
	delete gl_camera_point_cloud;
	delete gl_reference_point_cloud;
	delete gl_reference_eval;

	// render lines between matches
	delete gl_matches;
	delete gl_best_votes;
	delete gl_best_pose;

#ifdef _WITH_PRODUCER
	m_producers.clear();
#endif
	delete m_reg;
}

void TrackingExpertDemo::init(void)
{

	m_window_width = 1280;
	m_window_height = 1280;
	m_camera_type = None;
	m_camera_file = "";
	m_model_file = "";
	m_verbose = true;
	m_scene_type = PC;
	m_new_scene = false;
	m_enable_matching_renderer = false;
	m_enable_best_votes_renderer = false;
	m_enable_best_pose_renderer = false;
	m_render_curvatures = false;
	m_render_normals = false;
	m_current_debug_point = 0;
	m_current_debug_cluster = 0;
	m_producers = std::vector<PointCloudProducer>();
	m_enable_tracking = true;
	m_producer_param.uniform_step = 8;
	m_update_camera = true;
	m_enable_filter = false;

	m_filter_method = BILATERAL;

	m_voxel = GPUvoxelDownsample();

	// sampling parameters
	sampling_method = SamplingMethod::UNIFORM;
	sampling_param.grid_x = 0.015f;
	sampling_param.grid_y = 0.015f;
	sampling_param.grid_z = 0.015f;
	Sampling::SetMethod(sampling_method, sampling_param);
	
	// init the opengl window
	glm::mat4 vm = glm::lookAt(glm::vec3(0.0f, 0.0f, -0.5f), glm::vec3(0.0f, 0.0f, 0.5), glm::vec3(0.0f, 1.0f, 0.0f));
	m_window = new isu_ar::GLViewer();
	m_window->create(1280, 1280, "TrackingExpert+ Demo");
	m_window->addRenderFcn(std::bind(&TrackingExpertDemo::render_fcn, this, _1, _2));
	m_window->addKeyboardCallback(std::bind(&TrackingExpertDemo::keyboard_cb, this, _1, _2));
	m_window->setViewMatrix(vm);
	m_window->setClearColor(glm::vec4(0, 0, 0, 1));

	// registration 
	m_reg = new TrackingExpertRegistration();
}



/*!
Set a camera type to use or none, if the data comes from a file.
@param type - camera type of type CaptureDeviceType.
@return true 
*/
bool TrackingExpertDemo::setCamera(CaptureDeviceType type)
{
	m_camera_type = type;

	if(m_camera_type == None) return false;

	switch (m_camera_type) {
		case CaptureDeviceType::KinectAzure:
		{
#ifdef _WITH_AZURE_KINECT
			//set up pose based on given file
			CamPose poseFile = CamPose(2, 4, 7, 0.028f);
			poseFile.readFile("C:/Users/Tyler/Documents/TrackingExpertPlus/pose2.txt"); //Hard coded for pose file TODO -Tyler

			for (int i = KinectAzureCaptureDevice::getNumberConnectedCameras()-1; i >= 0; i--)
			{ 
				m_cameras.emplace(m_cameras.begin(), new KinectAzureCaptureDevice(i, KinectAzureCaptureDevice::Mode::RGBIRD, false));
				m_pc_camerasIndividual.push_back(new PointCloud());
			}

			//set pose for each PC
			for (int i = 0; i < m_cameras.size(); i++)
			{
				vector<vector<float>> pose = poseFile.getPose(i);
				m_pc_camerasIndividual[i]->pose = Eigen::Matrix4f();
				m_pc_camerasIndividual[i]->pose << // Eigen is column major be default
					pose.at(0).at(0), pose.at(1).at(0), pose.at(2).at(0), pose.at(3).at(0),
					pose.at(0).at(1), pose.at(1).at(1), pose.at(2).at(1), pose.at(3).at(1),
					pose.at(0).at(2), pose.at(1).at(2), pose.at(2).at(2), pose.at(3).at(2),
					pose.at(0).at(3), pose.at(1).at(3), pose.at(2).at(3), pose.at(3).at(3);
			}

#else
			std::cout << "[ERROR] - Camera Azure Kinect selected but no such camera is present." << std::endl;
#endif
		}
	}


	if (m_cameras.size() == 0) {
		std::cout << "[ERROR] - Could not create a camera " << std::endl;
		return false;
	}
	for (int i = 0; i < m_cameras.size(); i++)
	{
		if (!m_cameras[i]->isOpen()) {
			std::cout << "[ERROR] - Could not open camera at index " << i << std::endl;
		}
	}
	
	/*
	Create a point cloud producer. It creates a point cloud 
	from a depth map. Assign a camera and the location to store the point cloud data. 
	*/
#ifdef _WITH_PRODUCER
	
	for (int i = 0; i < m_cameras.size(); i++)
	{
		m_producers.emplace_back(texpert::PointCloudProducer(*m_cameras[i], *m_pc_camerasIndividual[i]));
		m_producers[i].setSampingMode(SamplingMethod::UNIFORM, m_producer_param);
		m_producers[i].setFilterMethod(m_filter_method, m_filter_param);
	}

	//voxel stuff
	float3 minBoandaries = make_float3(-2.0f, -2.0f, 0.0f);
	float3 maxBoundaries = make_float3(2.0f, 2.0f, 5.0f);
	m_voxel = GPUvoxelDownsample((int)m_cameras.size(), m_cameras[0]->getRows(CaptureDeviceComponent::DEPTH), m_cameras[0]->getCols(CaptureDeviceComponent::DEPTH));
	m_voxel.setBoundaries(minBoandaries, maxBoundaries, .01f);



	
#else
	m_producer = NULL;
#endif
	
	if(m_verbose){
		std::cout << "[INFO] - Open camera device successfull." << std::endl;
		std::cout << "[INFO] - RGB cols: " << m_cameras[0]->getCols(CaptureDeviceComponent::COLOR) << std::endl;
		std::cout << "[INFO] - RGB rows: " << m_cameras[0]->getRows(CaptureDeviceComponent::COLOR) << std::endl;
		std::cout << "[INFO] - Depth cols: " << m_cameras[0]->getCols(CaptureDeviceComponent::DEPTH) << std::endl;
		std::cout << "[INFO] - Depth rows: " << m_cameras[0]->getRows(CaptureDeviceComponent::DEPTH) << std::endl;
	}

	return true;
}


/*!
Load a scene model from a file instead of from a camers. 
@param path_and_filename - path and file to the scene model.
@return true - if the scene model was loaded. 
*/
bool TrackingExpertDemo::loadScene(std::string path_and_filename)
{
	if (m_camera_type != None) {
		std::cout << "[INFO] - Cannot load a scene file since the camera is not set to None." << std::endl;
		return false;
	}


	/*------------------------------------------------------------------------
	Load the second object for the test. 
	*/
	m_camera_file = "";

	FileUtils::Search(path_and_filename, m_camera_file);
	if (m_camera_file.empty()) {
		std::cout << "[ERROR] - Did not find file " << path_and_filename << "." << std::endl;
		return false;
	}

	// Load the file. 

	bool ret = ReaderWriterUtil::Read(m_camera_file, m_pc_camera_raw.points, m_pc_camera_raw.normals, false, false);
	
	if(!ret) {
		if(m_verbose) std::cout << "[ERROR] - Could not load file " << path_and_filename << "." << std::endl;
		return false;
	}

	// Sample the file

 	Sampling::Run(m_pc_camera_raw, m_pc_camera_raw, m_verbose);
	
	m_pc_camera_raw.centroid0 = PointCloudUtils::CalcCentroid(&m_pc_camera_raw);
	m_pc_camera = m_pc_camera_raw;


	// add file as a reference model.
	// set the new_scene flag to true so that the new scene gets processd
	m_new_scene =  m_reg->updateScene(m_pc_camera);


	return true;
}


/*!
Load and label a model file. Needs to be a point cloud file.
@param pc_path_and_filename - path and file to the model.
@param label - label the model with a string. 
@return true - if the model could be found and loaded. 
*/
bool TrackingExpertDemo::loadModel(std::string pc_path_and_filename, std::string label)
{

	/*------------------------------------------------------------------------
	Load the second object for the test. 
	*/
	m_model_file = "";

	FileUtils::Search(pc_path_and_filename, m_model_file);
	if (m_model_file.empty()) {
		std::cout << "[ERROR] - Did not find file " << pc_path_and_filename << "." << std::endl;
		return false;
	}

	if(pc_ref_as_loaded.size() > 0) {
		pc_ref_as_loaded.points.clear();
		pc_ref_as_loaded.normals.clear();
	}

	// Load the file. 

	bool ret = ReaderWriterUtil::Read(m_model_file, pc_ref_as_loaded.points, pc_ref_as_loaded.normals, false, false);
	
	if(!ret) {
		if(m_verbose) std::cout << "[ERROR] - Could not load file " << pc_path_and_filename << "." << std::endl;
		return false;
	}

	// Sample the file

 	Sampling::Run(pc_ref_as_loaded, pc_ref_as_loaded, m_verbose);
	
	pc_ref_as_loaded.centroid0 = PointCloudUtils::CalcCentroid(&pc_ref_as_loaded);
	pc_ref = pc_ref_as_loaded;


	// add file as a scene model.
	m_reg->addReferenceModel(pc_ref, label);


	return true;

}


/*
Inits all the graphical content. 
*/
void TrackingExpertDemo::initGfx(void)
{
	// camera point cloud
	gl_camera_point_cloud = new	isu_ar::GLPointCloudRenderer(m_pc_camera.points, m_pc_camera.normals);
	gl_camera_point_cloud->setPointColor(glm::vec3(1.0,0.0,0.0));
	gl_camera_point_cloud->setNormalColor(glm::vec3(0.8,0.5,0.0));
	gl_camera_point_cloud->setNormalGfxLength(0.05f);
	gl_camera_point_cloud->enableAutoUpdate(false);

	// reference point cloud
	gl_reference_point_cloud = new	isu_ar::GLPointCloudRenderer(pc_ref.points, pc_ref.normals);
	gl_reference_point_cloud->setPointColor(glm::vec3(0.0,1.0,0.0));
	gl_reference_point_cloud->setNormalColor(glm::vec3(0.0,0.8,0.8));
	gl_reference_point_cloud->setNormalGfxLength(0.005f);

	// point cloud for evaluation.
	gl_reference_eval = new	isu_ar::GLPointCloudRenderer(pc_ref.points, pc_ref.normals);
	gl_reference_eval->setPointColor(glm::vec3(1.0,1.0,0.0));
	gl_reference_eval->setNormalColor(glm::vec3(0.5,0.8,0.8));
	gl_reference_eval->setNormalGfxLength(0.05f);


	//-------------------------------------------------------------------
	// Render lines

	gl_matches =  new isu_ar::GLLineRenderer(pc_ref.points, m_pc_camera.points, match_pair_ids);
	gl_matches->setLineColor(glm::vec3(1.0,0.0,1.0));
	gl_matches->enableRenderer(false);

	gl_best_votes =  new isu_ar::GLLineRenderer(pc_ref.points, m_pc_camera.points, vote_pair_ids);
	gl_best_votes->setLineWidth(5.0);
	gl_best_votes->setLineColor(glm::vec3(1.0,1.0,0.0));
	gl_best_votes->enableRenderer(false);

	gl_best_pose = new isu_ar::GLLineRenderer(pc_ref.points, m_pc_camera.points, pose_ids);
	gl_best_pose->setLineWidth(5.0);
	gl_best_pose->setLineColor(glm::vec3(0.0,1.0, 0.2));
	gl_best_votes->enableRenderer(false);

}



/*!
Start the application. This is the last thing one should do 
since the function is blocking and will only return after the window closes.
*/
bool TrackingExpertDemo::run(void)
{
	enableTracking(false);

	// init graphics
	initGfx();

	Sleep(100);

	// start the viewer
	m_window->start();

	return true;
}	

/*
Render the point cloud sceen and show the point cloud content
*/
void TrackingExpertDemo::renderPointCloudScene(glm::mat4 pm, glm::mat4 vm)
{
	// draw the camera point cloud
	gl_camera_point_cloud->draw(pm, vm);
	gl_reference_point_cloud->draw(pm, vm);
	gl_reference_eval->draw(pm, vm);

	// render lines between matches
	gl_matches->draw(pm, vm);
	gl_best_votes->draw(pm, vm);
	gl_best_pose->draw(pm, vm);

}

/*
Render the AR scene and show AR content. 
*/
void TrackingExpertDemo::renderARScene(glm::mat4 pm, glm::mat4 vm)
{

}


/*
Track the reference object in the scene. 
*/
void TrackingExpertDemo::trackObject(void)
{
	assert(m_reg != NULL);

	if(!m_new_scene || !m_enable_tracking) return;

	// process the latest scene model. 
	m_reg->process();

	// set to false after processing, so that the scene does not get re-processed. 
	m_new_scene = false;

	// update the final pose
	upderRenderPose();

	// update the renderer
	updateRenderData();
	updateRenderCluster();
	
}

/*
Allows one to enable or disable the tracking functionality.
@param enable, true starts detection and registration
*/
void TrackingExpertDemo::enableTracking(bool enable)
{
	if (m_verbose) {
		if(enable){std::cout << "[INFO] - TRACKING IS ENABLED" << std::endl; }
		else{std::cout << "[INFO] - WARNING, TRACKING IS DISABLED" << std::endl; }
	}
	m_enable_tracking = enable;
}


/*
Update camera data
*/
void TrackingExpertDemo::updateCamera(void)
{
	if(m_camera_type == None || m_update_camera == false) return;
	if(m_producers.size() == 0) return;
	if (!m_voxel.properConstructorUsed()) return;

	// fetches a new camera image and update the data for all cameras. 
	for(int i = 0; i< m_producers.size(); i++)
		m_producers[i].process();

	std::vector<PointCloud*> temp = std::vector<PointCloud*>();
	for (int i = 0; i < m_pc_camerasIndividual.size(); i++)
		temp.emplace_back(new PointCloud());


	//voxel downsample into m_pc_camera
	m_voxel.voxelDownSample(m_pc_camerasIndividual);
	m_voxel.copyBackToHost(temp);
	m_voxel.removeDuplicates(temp, &m_pc_camera);

	//Sampling::Run(m_pc_camera, m_pc_camera, m_verbose);

	// camera point cloud, m_pc_camera needs to be the voxel downsample
	m_reg->updateScene(m_pc_camera);

	// Update the opengl points and draw the points. 
	gl_camera_point_cloud->updatePoints();

	// update the curvature values
	updateCurvatures();

	// the registratin call only starts working if this is set to true. 
	m_new_scene = true;

}


/*
Get a single frame from the camera.
*/
void TrackingExpertDemo::grabSingleFrame(void)
{

	if(m_camera_type == None ) return;
	if(m_producers.size() == 0) return;
	if (!m_voxel.properConstructorUsed()) return;

	// fetches a new camera image and update the data
	for(int i = 0; i < m_cameras.size(); i++)
		m_producers[i].process();

	//voxel downsample pc into m_pc_camera
	m_voxel.voxelDownSample(m_pc_camerasIndividual);
	m_voxel.copyBackToHost(m_pc_camerasIndividual);
	m_voxel.removeDuplicates(m_pc_camerasIndividual, &m_pc_camera);

	// camera point cloud
	m_reg->updateScene(m_pc_camera);

	//Sampling::Run(m_pc_camera, m_pc_camera, m_verbose);

	// Update the opengl points and draw the points. 
	gl_camera_point_cloud->updatePoints();

	// the registratin call only starts working if this is set to true. 
	m_new_scene = true;
	
}


/*
To be passed to the renderer to draw the content. 
*/
void TrackingExpertDemo::render_fcn(glm::mat4 pm, glm::mat4 vm)
{
	// update the camera data
	updateCamera();
	
	// update the poses if a new scene model is available.
	trackObject();

	switch (m_scene_type) {
		case PC:
			renderPointCloudScene(pm, vm);
			break;
		case AR:
			renderARScene(pm, vm);
			break;
	}
}


/*!
Set the application parameters
@param params - struct params of type TEParams. 
*/
bool TrackingExpertDemo::setParams(TEParams params)
{
	assert(m_reg != NULL);
	bool err =  m_reg->setParams(params);

	m_producer_param.uniform_step = params.camera_sampling_offset;
	m_filter_param.kernel_size = params.filter_kernel;
	m_filter_param.sigmaI = params.filter_sigmaI;
	m_filter_param.sigmaS = params.filter_sigmaS;
	m_enable_filter = true;

	if(params.filter_enabled) m_filter_method = BILATERAL;
	else{ 
		m_filter_method = NONE;
		m_enable_filter = false;
	}

	// Note that the next lines have no effect if the camera has not been initialized.
	// Go to setCamera to change default params. 
	// This method is only useful for changed during runtime. 
#ifdef _WITH_PRODUCER
	if (m_producers.size() != 0) {
		for (int i = 0; i < m_producers.size(); i++)
		{
			m_producers[i].setSampingMode(SamplingMethod::UNIFORM, m_producer_param);
			m_producers[i].setFilterMethod(m_filter_method, m_filter_param);
		}
	}
#endif

	sampling_param.grid_x = params.sampling_grid_size;
	sampling_param.grid_y = params.sampling_grid_size;
	sampling_param.grid_z = params.sampling_grid_size;
	Sampling::SetMethod(sampling_method, sampling_param);
	return err;
}


//-------------------------------------------------------------------------------------------------------------------------------
// render helpers

void TrackingExpertDemo::upderRenderPose(void) {

	assert(m_reg != NULL);

	std::vector<Eigen::Affine3f > poses;
	std::vector<int > pose_votes;

	m_reg->getPose(poses);

	if(poses.size() <= 0) return;
	
	// pose 0 is the one with the highest votes
	Eigen::Affine3f mat = poses[0];

	glm::mat4 m;
	for (int i = 0; i < 16; i++) {
		m[i/4][i%4] =  mat.data()[i];
	}

	gl_reference_eval->setModelmatrix(MatrixUtils::ICPRt3Mat4(m_reg->getICPPose()));
	//gl_reference_eval->setModelmatrix(m);

	gl_reference_eval->enablePointRendering(true);

//	std::cout << "RENDER POSE" << std::endl;
}


void TrackingExpertDemo::updateRenderData(void)
{
	assert(m_reg != NULL);

	m_current_debug_point = (std::max)((int)0, (std::min)( (int)pc_ref.points.size(), (int)m_current_debug_point ) );

	if(m_reg->getRenderHelper().getMatchingPairs(m_current_debug_point, match_pair_ids))
		gl_matches->updatePoints(pc_ref.points, m_pc_camera.points, match_pair_ids);

	if( m_reg->getRenderHelper().getVotePairs(m_current_debug_point, vote_pair_ids))
		gl_best_votes->updatePoints(pc_ref.points, m_pc_camera.points, vote_pair_ids);

	
}


void TrackingExpertDemo::updateRenderCluster(void)
{
	assert(m_reg != NULL);

	m_current_debug_cluster = (std::max)((int)0, (std::min)( (int)m_reg->getNumPoseClusters(), (int)m_current_debug_cluster ) );

	m_reg->getPoseClusterPairs(m_current_debug_cluster, pose_ids);
	gl_best_pose->updatePoints(pc_ref.points, m_pc_camera.points, pose_ids);
	
}



// debug rendering functions
void TrackingExpertDemo::renderMatches(void)
{
	if(m_enable_matching_renderer) m_enable_matching_renderer = false;
	else m_enable_matching_renderer = true;

	gl_matches->enableRenderer(m_enable_matching_renderer);

}

void TrackingExpertDemo::renderVotes(void)
{
	if(m_enable_best_votes_renderer) m_enable_best_votes_renderer = false;
	else m_enable_best_votes_renderer = true;

	gl_best_votes->enableRenderer(m_enable_best_votes_renderer);
}

void TrackingExpertDemo::renderPoseCluster(void)
{
	if(m_enable_best_pose_renderer) m_enable_best_pose_renderer = false;
	else m_enable_best_pose_renderer = true;

	gl_best_pose->enableRenderer(m_enable_best_pose_renderer);
}

void TrackingExpertDemo::renderCurvatures(void)
{
	assert(m_reg != NULL);

	if(m_render_curvatures){
		m_render_curvatures = false;
		gl_reference_point_cloud->setPointColor(glm::vec3(0.0, 1.0, 0.0));
		gl_camera_point_cloud->setPointColor(glm::vec3(1.0, 0.0, 0.0));
	}
	else{
		m_render_curvatures = true;
		updateCurvatures();
	}
}

void TrackingExpertDemo::updateCurvatures(void)
{
	if(!m_render_curvatures) return;

	std::vector<uint32_t> cu0, cu1;
	std::vector<glm::vec3> colors;
	std::vector<glm::vec3> colors2;

	m_reg->getModelCurvature(cu0);
	m_reg->getSceneCurvature(cu1);

	ColorCoder::CPF2Color(cu0, colors);
	gl_reference_point_cloud->setPointColors(colors);

	ColorCoder::CPF2Color(cu1, colors2);
	gl_camera_point_cloud->setPointColors(colors2);
}


void TrackingExpertDemo::renderNormalVectors(void)
{
	if(m_render_normals) m_render_normals = false;
	else m_render_normals = true;


	gl_camera_point_cloud->enableNormalRendering(m_render_normals);
	gl_reference_point_cloud->enableNormalRendering(m_render_normals);
	
}

/*
Keyboard callback for the renderer
*/
void TrackingExpertDemo::keyboard_cb(int key, int action)
{
	//cout << key << " : " << action << endl;
	switch (action) {
	case 0:  // key up
	
		switch (key) {
		case 87: // w
		{
			m_current_debug_point++;
			updateRenderData();
			break;
		} 
		case 81: // q
		{
			m_current_debug_point--;
			updateRenderData();
			break;
		} 
		case 78: // n
			{
			

			break;
			}
		case 32: // space
			{

				break;
			}
		case 65: // a
			{
				m_current_debug_cluster--;
				updateRenderCluster();
				break;
			}
		case 83: // s
			{
				m_current_debug_cluster++;
				updateRenderCluster();
				break;
			}
		case 49: // 1
			{
				renderMatches();
				updateRenderData();
				break;
			}
		case 50: // 2
			{
				renderVotes();
				updateRenderData();
				break;
			}
		case 51: // 3
			{
				renderPoseCluster();
				updateRenderCluster();
				break;
			}
		case 52: // 4
			{
				renderCurvatures();
				break;
			}
		case 53: // 5
			{
				
				break;
			}
		case 54: // 6
			{
				renderNormalVectors();
				break;
			}
		case 88: // x
			{
				if(m_enable_tracking) m_enable_tracking = false;
				else m_enable_tracking = true;

				enableTracking(m_enable_tracking);
				break;
			}
		case 66://b
			{
				if(m_enable_filter){ 
					m_enable_filter = false;
					m_filter_method = NONE;
				}
				else {
					m_enable_filter = true;
					m_filter_method = BILATERAL;
				}

#ifdef _WITH_PRODUCER
				if(m_producers.size() != 0)
				{
					for(int i = 0; i < m_producers.size(); i++)
						m_producers[i].setFilterMethod (m_filter_method, m_filter_param);
				}
#endif
				break;
			}	
		case 67://c
			{
				if(m_update_camera) m_update_camera = false;
				else m_update_camera = true;
				break;
			}	
		case 70:// f
			{
				grabSingleFrame();
				break;
			}	

		}
		break;
		

		

		case 1: // key down

			break;
	}
}


/*
Enable more outputs
@param verbose - true enables more debug outputs. 
*/
bool TrackingExpertDemo::setVerbose(bool verbose)
{
	assert(m_reg != NULL);

	m_verbose = verbose;

	m_reg->setVerbose(m_verbose);

	return m_verbose;
}

void TrackingExpertDemo::generatePoseData(string poseFolderlocation, string fileName)
{
	int numCameras = KinectAzureCaptureDevice::getNumberConnectedCameras();
	if (numCameras <= 0)
	{
		printf("No cameras connected. File not created.\n");
		return;
	}

	//initiliaze cameras
	std::vector<KinectAzureCaptureDevice*> cameras = std::vector<KinectAzureCaptureDevice*>();
	for (int i = 0; i < numCameras; i++)
	{
		cameras.push_back(new KinectAzureCaptureDevice(i, KinectAzureCaptureDevice::Mode::RGBIRD, false));
	}

	cv::String window = "Generate Pose";

	//declare opencv windows, move them
	cv::namedWindow(window);
	cv::resizeWindow(window, 1080 * numCameras, 720);
	cv::moveWindow(window, 0, 0);
	cv::waitKey(1);

	//Test if the cameras are ready to run.
	for (int i = 0; i < numCameras; i++)
	{
		if (cameras[i]->isOpen() == false)
		{
			std::cout << "Camera at index " << i << " could not connect." << endl;
			return;
		}
	}

	vector<cv::Mat> colors;
	cv::Mat ColorConcat;

	//start rendering
	printf("Place the checkerboard in clear view of all cameras. The squares must be visible and clear in each frame. Make sure it is in a static position (not being held by a person).\n");
	printf("Press ESC when your cameras meet the above criteria.\n");
	while (1)
	{
		colors.clear();
		for (int i = 0; i < numCameras; i++)
		{
			if (cameras[i]->isOpen() == false)
				break;
			colors.emplace_back(cv::Mat());
			cameras[i]->getRGBFrame(colors[i]);
		}
		cv::hconcat(colors, ColorConcat);
		cv::imshow(window, ColorConcat);


		// Press  ESC on keyboard to  exit
		if ((char)cv::waitKey(10) == 27) break;
	}

	cv::destroyAllWindows();
	cv::waitKey(1);
	printf("Generating pose file. This may take a few minutes depending on the image quality, size, and checkerboard layout.\n");

	//Cameras are in position
	//Using 2 cameras, a 4x7 checkerboard with square sides of 32mm
	CamPose pose = CamPose(numCameras, 4, 7, 0.032f);
	//set calibration files
	for (int i = 0; i < numCameras; i++)
	{
		pose.addCameraCalibrations(i, cameras[i]->getCalibration(texpert::CaptureDeviceComponent::COLOR));
	}
	pose.start(colors, poseFolderlocation.c_str(), true, fileName.c_str());

	// delete all instances.
	cv::destroyAllWindows();
}