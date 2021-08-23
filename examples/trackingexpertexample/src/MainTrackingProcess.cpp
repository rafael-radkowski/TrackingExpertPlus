#include "MainTrackingProcess.h"


/*
Get an instance of the class.
@return Instance of the class
*/
//static 
MainTrackingProcess* MainTrackingProcess::getInstance()
{
	if (m_instance == nullptr) {
		m_instance = new MainTrackingProcess();
	}
	return m_instance;
}

MainTrackingProcess::~MainTrackingProcess()
{


}

/*
Private constructor
*/
MainTrackingProcess::MainTrackingProcess()
{
	m_producer_param.uniform_step = 8;

	m_enable_tracking = false;
	m_tracking_state = IDLE;

	m_producer = NULL;
}


/*
Init the tracking procss and set the camera object
*/
void MainTrackingProcess::init(texpert::ICaptureDevice* camera)
{

	// fetch an instance of the data manager. 
	_dm = PointCloudManager::getInstance();
	if (_dm == NULL) {
		std::cout << "[ERROR] -  Did not obtain a DataManager instance." << std::endl;
	}

	if(camera == NULL)
		return;  // no camera is given, e.g., static scene, or the app is not going to process anything. 

	/*
	Create a point cloud producer. It creates a point cloud
	from a depth map. Assign a camera and the location to store the point cloud data.
	*/
	m_producer = new texpert::PointCloudProducer(*camera, _dm->getCameraPC()); //m_pc_camera
	m_producer->setSampingMode(SamplingMethod::UNIFORM, m_producer_param);
	m_producer->setFilterMethod(m_filter_method, m_filter_param);


	/*
	Init ICP 
	*/
	m_icp = new ICP();
	m_icp->setMinError(0.0001);
	m_icp->setMaxIterations(10);
	m_icp->setVerbose(false, 0);
	m_icp->setRejectMaxAngle(25.0);
	m_icp->setRejectMaxDistance(0.1);
	m_icp->setRejectionMethod(ICPReject::DIST_ANG);
	
	
	// create an ICP and feature descriptor instance. 
	m_fd = new CPFMatchingExp();

	// set the default params.
	m_fd_params.search_radius = 0.1;
	m_fd_params.angle_step = 12.0;
	m_fd->setParams(m_fd_params);



	// add a reference model to the point cloud
	m_model_id = m_fd->addModel(_dm->getReferecePC(), "reference_test");

	m_model_pc = _dm->getReferecePC();
}


/*
Grab a new frame and process the frame
including all tracking steps.
*/
void MainTrackingProcess::process(void)
{
	// grab the next camera image and process it. 
	// Only required if a producer is available, means if a camera is available. 
	if(m_producer) m_producer->process();


	// stop here if tracking is disabled.
	if(!m_enable_tracking) return;

	// run tracking operations.
	switch (m_tracking_state) {
		case IDLE:
			runIdle();
			break;
		case DETECT:
			runDetect();
			break;
		case REGISTRATION:
			runRegistration();
			break;
		case TRACKING:
			runTracking();
			break;
		default:
			runIdle();
	}

}


/*
Set the sampling params for the point cloud producer
@params params - parameter varialb.s
*/
void MainTrackingProcess::setSamplingParams(SamplingParam params)
{
	m_producer_param = params;

	if(m_producer == NULL) return;
	
	m_producer->setSampingMode(SamplingMethod::UNIFORM, m_producer_param);
	
}



/*
Set the parameters for the tracking filter
*/
void MainTrackingProcess::setFilterParams(FilterMethod method, FilterParams params)
{
	m_filter_method = method;
	m_filter_param = params;

	if (m_producer == NULL) return;

	m_producer->setFilterMethod(m_filter_method, m_filter_param);
}


/*
Enable or disable object tracking functionality.
Note that "tracking" refers here to a single-shot function.
Tracking via multiple frames is realized via a Kalman filter.
@param enable - true enables tracking and false disables tracking.
*/
void MainTrackingProcess::enableTracking(bool enable)
{
	m_enable_tracking = enable;
	if(m_enable_tracking){
		m_tracking_state = REGISTRATION;
		std::cout << "[INFO] - TRACKING ENABLED" << std::endl;
	}
	else{
		m_tracking_state = IDLE;
		std::cout << "[INFO] - TRACKING DISABLED" << std::endl;
	}

}


/*!
	Run the idel state operations;
	*/
void MainTrackingProcess::runIdle(void)
{

}

/*!
Run the detect state operations;
*/
void MainTrackingProcess::runDetect(void)
{
	// update the camera point cloud
	m_fd->setScene(_dm->getCameraPC());

	int ret = m_fd->match(m_model_id);

	// positive match
	if (ret) {

		// Do not set the scene data earlier. 
		// icp and fd share the same knn. 
		// m_fd->setScene(points) overwrites the data. 
		m_icp->setCameraData(_dm->getCameraPC());

		std::vector<Eigen::Affine3f > pose;
		getPose(pose);

		if (pose.size() <= 0) {
			return ;
		}

		texpert::Pose Rt;
		Rt.t = pose[0];

		Eigen::Matrix4f out_Rt;
		//float rms = 0.0;

		// ICP works internally with a copy of m_model_pc
		m_icp->compute(m_model_pc, Rt, out_Rt, m_rms);


		// return the pose
		m_model_pose = m_icp->Rt();


	}


}

/*!
Run the registration state operations;
*/
void MainTrackingProcess::runRegistration(void)
{

	Pose pose;
	pose.t = Eigen::Affine3f::Identity();
	// for test results. 
	Eigen::Matrix4f pose_result;
	float rms = 1000.0;

	// run icp
	m_icp->setCameraData(_dm->getCameraPC());
	m_icp->compute(_dm->getReferecePC(), pose, pose_result, rms);
	pose_result = m_icp->Rt();
	//cout << pose_result << endl;


	PointCloudTransform::Transform(&_dm->getReferecePC(), pose_result, false);

	_dm->getReferecePC().pose = pose_result;
	_dm->updatePose();
}

/*!
Run the tracking state operations;
*/
void MainTrackingProcess::runTracking(void)
{

}


/*!
Return the poses for a particular model.
The function returns the 12 best hits by default. Note that this can be reduced to only 1 or so.
@param model_id - the model id of the object to track as int.
@param poses - vector with the poses
@param pose_votes - vector with the pose votes.
*/
bool MainTrackingProcess::getPose(std::vector<Eigen::Affine3f >& poses)
{
	
	assert(m_fd != NULL);

	m_pose_votes.clear();

	m_fd->getPose(m_model_id, poses, m_pose_votes);

	return true;
	

}




/*!
Returns the pose after ICP was applied.
@return a 4x4 matrix in homogenous coordinates with the pose.
*/
Eigen::Matrix4f  MainTrackingProcess::getCurrentPose(void)
{
	assert(m_icp != NULL);

	return m_model_pose;

}


/*
	Process one step only
	*/
void MainTrackingProcess::step(void)
{
	runRegistration();
}