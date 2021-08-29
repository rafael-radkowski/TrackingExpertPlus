#include "TrackingExpertDemo.h"


using namespace texpert;

#ifdef _WITH_AZURE_OUTPUT
// crude and ugly debug helper. Remove!
texpert::ICaptureDevice* g_camera = NULL;
#endif

TrackingExpertDemo::TrackingExpertDemo()
{
	m_camera_type = None;
	m_camera_file = "";
	m_model_file = "";
	m_verbose = true;
	m_is_running = false;

	m_enable_tracking = true;
	m_update_camera = true;


	m_render_scene_normals = false;
	m_render_ref_normals = false;
	m_enable_tracking = false;
	m_render_ref_points = true;
	m_render_scene_points = true;
	m_render_eval_points = false;
	m_enable_curvature_renderer  = false;

	init();
}


TrackingExpertDemo::~TrackingExpertDemo()
{
	if(m_camera)
		delete m_camera;

	//delete m_reg;
}

void TrackingExpertDemo::init(void)
{
	

	// fetch an instance of the data manager. 
	_dm = PointCloudManager::getInstance();
	if(_dm == NULL){
		std::cout<< "[ERROR] -  Did not obtain a DataManager instance." << std::endl;
	}

	_renderer = MainRenderProcess::getInstance();
	if (_renderer == NULL) {
		std::cout << "[ERROR] -  Did not obtain a MainRendererProcess instance." << std::endl;
	}

	// add the keyboard callback
	_renderer->setKeyboardFcn(std::bind(&TrackingExpertDemo::keyboard_cb, this, _1, _2));
	

	_tracking =  MainTrackingProcess::getInstance();
	if (_tracking == NULL) {
		std::cout << "[ERROR] -  Did not obtain a MainTrackingProcess instance." << std::endl;
	}


	// sampling 
	// The loaded reference model gets sampled with those parameters. 
	sampling_method = SamplingMethod::UNIFORM;
	sampling_param.grid_x = 0.0015f;
	sampling_param.grid_y = 0.0015f;
	sampling_param.grid_z = 0.0015f;
	Sampling::SetMethod(sampling_method, sampling_param);
	

}



/*!
Set a camera type to use or none, if the data comes from a file.
@param type - camera type of type CaptureDeviceType.
@return true 
*/
bool TrackingExpertDemo::setSourceCamera(CaptureDeviceType type)
{
	m_camera_type = type;

	if(m_camera_type == None) return false;

	switch (m_camera_type) {
		case CaptureDeviceType::KinectAzure:
		{
#ifdef _WITH_AZURE_KINECT

			m_camera = new KinectAzureCaptureDevice();
#else
			std::cout << "[ERROR] - Camera Azure Kinect selected but no such camera is present." << std::endl;
			system("pause");
#endif	
			break;
		}
		case CaptureDeviceType::AzureKinectMKV:
		{
			initAzureKinectMKV();
			break;
		}
		default:
		{
			std::cout << "[ERROR] - No other camera is supported at this time (press a key to exit)." << std::endl;
			system("pause");
			exit(1);
			break;
		}
	}

	// check if camera is open. 
	if (!m_camera->isOpen()) {
		std::cout << "[ERROR] - Could not open camera (press a key to exit)." << std::endl;
		system("pause");
		exit(1);
	}

#ifdef _WITH_AZURE_OUTPUT
	// crude and ugly debug helper. Remove!
	g_camera = m_camera;
#endif


	if (m_verbose) {
		std::cout << "[INFO] - Open camera device successfull." << std::endl;
		std::cout << "[INFO] - RGB cols: " << m_camera->getCols(CaptureDeviceComponent::COLOR) << std::endl;
		std::cout << "[INFO] - RGB rows: " << m_camera->getRows(CaptureDeviceComponent::COLOR) << std::endl;
		std::cout << "[INFO] - Depth cols: " << m_camera->getCols(CaptureDeviceComponent::DEPTH) << std::endl;
		std::cout << "[INFO] - Depth rows: " << m_camera->getRows(CaptureDeviceComponent::DEPTH) << std::endl;
	}


	// Initialize the tracking class. 
	_tracking->init(m_camera);

	
	return true;
}



/*
Init the Azure Kinect video as source.
Note that the camera file name needs to be set vis setParams(), the parameter
TEParams::input_mkv needs to hold the path and file.
*/
bool TrackingExpertDemo::initAzureKinectMKV(void)
{
	FileUtils::Search(m_camera_file, m_camera_file);
	if (m_camera_file.empty()) {
		std::cout << "[ERROR] - Did not find file " << m_camera_file << "." << std::endl;
		return false;
	}

	m_camera = new AzureKinectFromMKV(m_camera_file);

	return true;
}



/*!
Load a scene model from a file instead of from a camers. 
@param path_and_filename - path and file to the scene model.
@return true - if the scene model was loaded. 
*/
bool TrackingExpertDemo::setSourceScene(std::string path_and_filename)
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

	bool ret = ReaderWriterUtil::Read(m_camera_file, _dm->getCameraRawPC().points, _dm->getCameraRawPC().normals, false, false); 
	
	if(!ret) {
		if(m_verbose) std::cout << "[ERROR] - Could not load file " << path_and_filename << "." << std::endl;
		return false;
	}

	// Sample the file
	Sampling::SetMethod(sampling_method, sampling_param);
 	Sampling::Run(_dm->getCameraRawPC(), _dm->getCameraPC(), m_verbose); 
	
	// Update the centroid
	_dm->getCameraPC().centroid0 = PointCloudUtils::CalcCentroid(&_dm->getCameraPC()); 


	return true;
}


/*!
Load and label a model file. Needs to be a point cloud file.
@param pc_path_and_filename - path and file to the model.
@param label - label the model with a string. 
@return true - if the model could be found and loaded. 
*/
bool TrackingExpertDemo::loadReferenceModel(std::string pc_path_and_filename, std::string label)
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

	_dm->clearReferencePC();

	// Load the file. 
	bool ret = ReaderWriterUtil::Read(m_model_file, _dm->getRefereceRawPC().points, _dm->getRefereceRawPC().normals, false, false); //  pc_ref_as_loaded.points, pc_ref_as_loaded.normals
	
	if(!ret) {
		if(m_verbose) std::cout << "[ERROR] - Could not load file " << pc_path_and_filename << "." << std::endl;
		return false;
	}

	// Sample the file
 	Sampling::Run(_dm->getRefereceRawPC(), _dm->getReferecePC(), m_verbose);

	_dm->getReferecePC().centroid0 = PointCloudUtils::CalcCentroid(&_dm->getReferecePC());


	//pass model to tracking process
	_tracking->addReferenceModel(_dm->getReferecePC(), "ref_model");

	return true;

}


/*!
Start the application. This is the last thing one should do 
since the function is blocking and will only return after the window closes.
*/
bool TrackingExpertDemo::run(void)
{
	enableTracking(false);

	// init graphics
	_renderer->initGfx();

	Sleep(100);

#ifdef _WITH_SEQUENTIAL
	_renderer->addRenderFunction(std::bind(&TrackingExpertDemo::autoProcessFrame, this));
#else
	// assign the camera update to a new thread
	m_camera_updates = std::thread(std::bind(&TrackingExpertDemo::autoProcessFrame, this));
#endif

	// blocking function
	_renderer->start();

	// end thread
	m_camera_updates.detach();

	return true;
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

	_tracking->enableTracking(m_enable_tracking);
}


/*
Update camera data
*/
void TrackingExpertDemo::autoProcessFrame(void)
{
	if(m_camera_type == None || m_update_camera == false) return;
	if(_tracking == NULL) return;

	m_is_running = true;

#ifdef _WITH_SEQUENTIAL
	if (m_is_running) {
#else
	while(m_is_running){
#endif
		// fetches a new camera image and update the data for all cameras. 
		// process this frame
		_tracking->process();

		// set the renderer to update
		_renderer->setUpdate();


		

		Sleep(30);

	}

}




/*!
Set the application parameters
@param params - struct params of type TEParams. 
*/
bool TrackingExpertDemo::setParams(TEParams params)
{
	//assert(m_reg != NULL);
	//bool err =  m_reg->setParams(params);

	m_producer_param.uniform_step = params.camera_sampling_offset;
	m_filter_param.kernel_size = params.filter_kernel;
	m_filter_param.sigmaI = params.filter_sigmaI;
	m_filter_param.sigmaS = params.filter_sigmaS;


	// camera file to load
	m_camera_file = params.input_mkv;
	

	if(params.filter_enabled) m_filter_method = BILATERAL;
	else{ 
		m_filter_method = NONE;
	}

	// Note that the next lines have no effect if the camera has not been initialized.
	// Go to setCamera to change default params. 
	// This method is only useful for changed during runtime. 
	_tracking->setSamplingParams(m_producer_param);
	_tracking->setFilterParams(m_filter_method, m_filter_param);


	// set the cpu sampler sampling methods. 
	sampling_param.grid_x = params.sampling_grid_size;
	sampling_param.grid_y = params.sampling_grid_size;
	sampling_param.grid_z = params.sampling_grid_size;
	Sampling::SetMethod(sampling_method, sampling_param);
	return false;
}


/*
Reset the reference model to the state as loaded
*/
void TrackingExpertDemo::resetReferenceModel(void)
{
	// Sample the file
	Sampling::Run(_dm->getRefereceRawPC(), _dm->getReferecePC(), m_verbose);

	_dm->getReferecePC().centroid0 = PointCloudUtils::CalcCentroid(&_dm->getReferecePC());

	
#ifdef _WITH_REGISTRATION
	//-----------------------------------------------------------------------
	// debugging
	PointCloudManager* m = PointCloudManager::getInstance();
	Eigen::Vector3f t(0.0, 0.1, 0.500);
	Eigen::Vector3f R(-45.0, 180.0, 0.0);
	PointCloudTransform::Transform(&m->getReferecePC(), t, R);
#endif

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
		
			break;
		} 
		case 81: // q
		{
		
		
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
			_tracking->step();
				break;
			}
		case 83: // s
			{
				if(_tracking){
					if(m_enable_tracking) m_enable_tracking = false;
					else m_enable_tracking  = true;
					_tracking->enableTracking(m_enable_tracking);

					if(!m_enable_tracking){
						resetReferenceModel();
					}
				}
				break;
			}
		case 49: // 1
			{
				if (_renderer) {
					if (m_render_scene_points == false) {
						m_render_scene_points = true;
					}
					else {
						m_render_scene_points = false;
					}
					_renderer->setRenderFeature(MainRenderProcess::PointsScene, m_render_scene_points);
				}
				break;
			}
		case 50: // 2
			{
				if (_renderer) {
					if (m_render_ref_points == false) {
						m_render_ref_points = true;
					}
					else {
						m_render_ref_points = false;
					}
					_renderer->setRenderFeature(MainRenderProcess::PointsRef, m_render_ref_points);
				}
				break;
			}
		case 51: // 3
			{
				if (_renderer) {
					if(m_render_scene_normals == false){
						m_render_scene_normals = true;
					}else{
						m_render_scene_normals = false;
					}
					_renderer->setRenderFeature(MainRenderProcess::NormalsScene, m_render_scene_normals);
				}
				break;
			}
		case 52: // 4
			{
				if (_renderer) {
					if (m_render_ref_normals == false) {
						m_render_ref_normals = true;
					}
					else {
						m_render_ref_normals = false;
					}
					_renderer->setRenderFeature(MainRenderProcess::NormalsRef, m_render_ref_normals);

				}
		
				break;
			}
		case 53: // 5
			{
				
				break;
			}
		case 54: // 6
			{
			
				break;
			}
		case 88: // x
			{

				break;
			}
		case 66://b
			{
	


				break;
			}	
		case 67://c
			{
				if(m_enable_curvature_renderer)
					m_enable_curvature_renderer = false;
				else
					m_enable_curvature_renderer = true;

				_renderer->setRenderFeature(MainRenderProcess::CurvScene, m_enable_curvature_renderer);
				break;
			}	
		case 70:// f
			{
				
				break;
			}	

		case 82: // r
			{
				resetReferenceModel();
			}


		}
		break;
		

		

		case 1: // key down
			switch (key) {
			case 256: // esc
			{
				m_is_running = false;
				break;
			}
			break;
		}
	}
}


/*
Enable more outputs
@param verbose - true enables more debug outputs. 
*/
bool TrackingExpertDemo::setVerbose(bool verbose)
{
	//assert(m_reg != NULL);

	m_verbose = verbose;

	//m_reg->setVerbose(m_verbose);

	return m_verbose;
}
