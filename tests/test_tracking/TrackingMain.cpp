#include "TrackingMain.h"

using namespace texpert;



TrackingMain::TrackingMain()
{
	_producer = NULL;
	_sParam.uniform_step = 4;
	_sParam.random_max_points = 5000;



	_reg = new PCRegistratation();
}


TrackingMain::~TrackingMain()
{

}


/*
Create the tracking instance. 
@param camera - reference to a camera that delivers depth data. 
@return true - if successful. 
*/
bool TrackingMain::create(ICaptureDevice& camera)
{
	if (!camera.isOpen()) {
	std:cout << "[ERROR] - Start the camera first. " << std::endl;
		return false;
	}

	// start the point cloud producer
	_producer = new texpert::PointCloudProducer(camera, _camera_point_cloud);

	/* Set the sampling parameters. 
	The producer instance can provide the full frame (RAW), a uniformly smapled 
	point cloud (UNIFORM), and a randomly sampled (RANDOM) point cloud. */
	_producer->setSampingMode(UNIFORM, _sParam);

}


/*
Return the point cloud
*/
PointCloud& TrackingMain::getPointCloudRef(void)
{
	return _camera_point_cloud;
}


/*
Process the current frame;
*/
bool TrackingMain::process(void)
{
	if(!_producer)
		return false;

	// create a new point cloud
	_producer->process();
}