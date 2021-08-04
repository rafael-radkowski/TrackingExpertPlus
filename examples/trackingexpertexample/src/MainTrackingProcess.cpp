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

	m_producer = NULL;
}


/*
Init the tracking procss and set the camera object
*/
void MainTrackingProcess::init(texpert::ICaptureDevice* camera)
{

	// fetch an instance of the data manager. 
	PointCloudManager* dm = PointCloudManager::getInstance();
	if (dm == NULL) {
		std::cout << "[ERROR] -  Did not obtain a DataManager instance." << std::endl;
	}

	if(camera == NULL)
		return;  // no camera is given, e.g., static scene, or the app is not going to process anything. 

	/*
	Create a point cloud producer. It creates a point cloud
	from a depth map. Assign a camera and the location to store the point cloud data.
	*/
	m_producer = new texpert::PointCloudProducer(*camera, dm->getCameraPC()); //m_pc_camera
	m_producer->setSampingMode(SamplingMethod::UNIFORM, m_producer_param);
	m_producer->setFilterMethod(m_filter_method, m_filter_param);

	
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