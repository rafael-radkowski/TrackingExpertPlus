#include "PointCloudProducer.h"
#include "FileWriter.h"
#include "ImageReaderWriter.h"

//#define TIMEFRAME

// This flag moves the entire point cloud preparation process to cude
#define _WTIH_CUDA_PC

#ifdef _WTIH_CUDA_PC
using namespace  tacuda;
#endif


// These are the variables of a plane with CP_A * x + CP_B * y + CP_C * z = CP_D
// where a point gets removed if CP_D - current_D > CP_THRESHOLD
// The values are set in OSGCuttingPlane, when the user change the plane.
extern double g_CP_A = 0.0;
extern double g_CP_B = 0.0;
extern double g_CP_C = 48.0;
extern double g_CP_D = 1960.0;
extern double g_CP_THRESHOLD = .015;
extern bool   g_CP_ACTIV = false; // activates or deactivates the cutting plane. 


PointCloudProducer::PointCloudProducer(vector<dPoint>& dst_points, vector<Vec3>& dst_normals, vector<Vec3>& dst_selected_normals):
	_point_cloud_vector(dst_points), _point_normals_original(dst_normals), _point_normals_selected_normals(dst_selected_normals)
{

	_sensor = NULL;
	_frameHandler = NULL;
	_simSensor = NULL;
	_sensor_connected = -1;
	_sensor_index = -1;
	_sensor_type = NONE;
	_is_running = false;
	_cameraSimulationFile = "";
	_autorun = false;
	_uniform_step_density = 2;
	_depth_width = -1;
	_depth_height = -1;
	_random_percentage = -1.0;
	_normal_vector_stride = 4;
	_random_max_points = 5000;

	_focal_length_x = 500.0;
	_focal_length_y = 500.0;
	_cx = 0;
	_cy = 0;

	_sampling_method = UNIFORM;

	callback_function = NULL;

	/////////////////////////////////////////////////////////////////////////////////////////////////////
	////  Init the frame handler
	using namespace std::placeholders;
	_frameHandler = new DepthStreamListener(std::bind(&PointCloudProducer::newFrameReady, this));
	
}


PointCloudProducer::~PointCloudProducer()
{
}



/*
Set the callback function that is called when a point cloud
is ready being processed.
*/
void PointCloudProducer::setCallbackPtr(std::function<void()> callback)
{
	callback_function = callback;
}


/*
LEGACY CODE
Set a camera index.
@param camera_index - the index of the camera ot pick.
*/
void PointCloudProducer::setCameraIndex(int camera_index)
{
	_sensor_index = camera_index;
}




/*
Connect to a camera device.
Note that all paramaters are overwrite parameters. If non is set,
the default camera parameters will be used.
@param camera_id - an integer indicating the camera id.
@param rgb_width - the desired width of the rgb image in pixels.
@paarm rgb_height - the desired height of the rgb image in pixels.
@param depth_width - the desired width of the depth image in pixels.
@param depth_height - the desired height of the depth image in pixels.
@param fps - the desired frames per sec.
@return int - if successfully returns the connect status
*/
int PointCloudProducer::connectToCamera(int camera_id, int rgb_width, int rgb_height, int depth_width, int depth_height, int fps)
{


	bool rgb = true;
	bool depth = true;
	int w, h, focal_length;
	switch (_sensor_index)
	{
	case 0: rgb = true;  depth = true; _frameHandler->set_focal_length(0); _sensor_type = SensorType::KINECT_V1; break; // kinect rgb + depth
	case 1: rgb = false; depth = true; _frameHandler->set_focal_length(0); _sensor_type = SensorType::KINECT_V1; break; // kinect depth
	case 2: rgb = true;  depth = true; _frameHandler->set_focal_length(0); _sensor_type = SensorType::P60U; break; // P60 with color and depth
	case 3: rgb = false; depth = true; _frameHandler->set_focal_length(0); _sensor_type = SensorType::P60U; break; // P60 with  depth
	case 4: rgb = false; depth = true; _frameHandler->set_focal_length(1); _sensor_type = SensorType::STRUCTUR_IO; break; // structure sensor
	case 5: rgb = false; depth = true; focal_length = 288.25; _frameHandler->set_focal_length(0); w = 640; h = 480; _sensor_type = SensorType::REC_KINECT_V1; break; // Recorded Kinect //TODO Verify focal lengths
	case 6: rgb = false; depth = true; focal_length = 288.25; _frameHandler->set_focal_length(1); w = 320; h = 240; _sensor_type = SensorType::REC_STRUCTUR_IO; break; // Recorded Structure 
	case 7: rgb = false; depth = true; focal_length = Fotonic_P60.focal_length_x; _frameHandler->set_focal_length(1); w = Fotonic_P60.depth_width; h = Fotonic_P60.depth_height; _sensor_type = SensorType::REC_P60U; break; // Recorded P60U 
	}

	if (_sensor_type == SensorType::KINECT_V1 ||
		_sensor_type == SensorType::P60U ||
		_sensor_type == SensorType::STRUCTUR_IO) {
		// Connect to a real sensor
		_sensor = new Sensor();

		if (_sensor_type == SensorType::P60U) {
			_sensor->setRGBResolution(Fotonic_P60.rgb_width, Fotonic_P60.rgb_height, Fotonic_P60.rgb_fps);
		}

		//---------------------------------------------------------------------------------------------
		// Manual rgb override
		if (rgb_width > -1 && rgb_height > -1) {
			_sensor->setRGBResolution(rgb_width, rgb_height, fps);
		}
		//---------------------------------------------------------------------------------------------


		int count = 0;
		while (count < 20)
		{
			_sensor_connected = _sensor->connect(depth, rgb); // 1 if successful;, otherwise 0
			count++;
			if (_sensor_connected != 0)break;
		}

		/* Set all settings here */

		if (_sensor_connected == 1)
		{
			w = _sensor->depth_w();
			h = _sensor->depth_h();
			_depth_width = w;
			_depth_height = h;

			switch (_sensor_type)
			{
			case KINECT_V1:
				if (w == Kinect_V1.depth_width  && h == Kinect_V1.depth_height) // Kinect V1 and V2 are distinguisehd using the depth camera resolution. 
				{
					_focal_length_x = Kinect_V1.focal_length;
					_focal_length_y = Kinect_V1.focal_length;
				}
				else if (w == Kinect_V2.depth_width  && h == Kinect_V2.depth_height) // this is the Kinect V2
				{
					_focal_length_x = Kinect_V2.focal_length;
					_focal_length_y = Kinect_V2.focal_length;
					_sensor_type = SensorType::KINECT_V2;
				}

				break;

			case KINECT_V2:
				if (w == Kinect_V2.depth_width  && h == Kinect_V2.depth_height) // this is the Kinect V2
				{
					_focal_length_x = Kinect_V2.focal_length;
					_focal_length_y = Kinect_V2.focal_length;
					_sensor_type = SensorType::KINECT_V2;
				}
				break;
			case P60U:
				_focal_length_x = Fotonic_P60.focal_length_x;
				_focal_length_y = Fotonic_P60.focal_length_y;
				_cx = Fotonic_P60.principle_x;
				_cy = Fotonic_P60.principle_y;
				_cprintf("P60U selected\n");
				break;
			case STRUCTUR_IO:
				_focal_length_x = StructureIO.focal_length;
				_focal_length_y = StructureIO.focal_length;
				break;
			}


#ifdef _WTIH_CUDA_PC		// Allocating memory
			cuPCU::AllocateDeviceMemory(_sensor->depth_w(), _sensor->depth_h(), 1);   //depth  x width x channels
#endif

			_frameHandler->set_focal_length(_focal_length_x);

			return _sensor_connected;
		}



	}
	else {
		// Simulate a sensor
		_simSensor = new SimulatedSensor(w, h, _focal_length_x);
		//_frameHandler->setNewFrameCallback(newFrameReady); // now in constructor
		using namespace std::placeholders;
		if (_simSensor->initSimulatedSensor(std::bind(&PointCloudProducer::newFrameReady, this), _sensor_index, _cameraSimulationFile, w, h)) {
			if (_sensor_index>4) {
				//Meaning there is a camera connected but its not an OpenNI camera.
				_sensor_connected = 2;
			}
			else {
				//Initialization Failed:
				_sensor_connected = 0;
				return _sensor_connected;
			}
		}
	}

	return _sensor_connected;
}




/*
Load a single frame from a connected camera;
*/
bool  PointCloudProducer::loadSingleFrame(void)
{

	if (_sensor_connected<2) {

		//Real openNI sensor
		openni::VideoStream& depth = _sensor->getDepthStream();

		//float focal_length= 361.6f;//TODO FIX FOCAL LENGTH GET FROM SOURCE
		openni::VideoFrameRef frame;
		openni::Status status = depth.readFrame(&frame);

		//_cprintf("HFOV: %lf\n", _sensor->getDepthStream().getHorizontalFieldOfView());
		//_cprintf("VFOV: %lf\n", _sensor->getDepthStream().getVerticalFieldOfView());

		if (status != openni::STATUS_OK) { return false; }

		//TODO save in super Struct
		uint16_t* imgBuf = (uint16_t*)frame.getData();

		int h = frame.getHeight();
		int w = frame.getWidth();

		// stores the last cv image
		_depth_16U = cv::Mat(h, w, CV_16UC1, (unsigned char*)imgBuf);

#ifdef _WTIH_CUDA_PC

		cuSample::CreateUniformSamplePattern(w, h, _uniform_step_density);
		cuSample::CreateRandomSamplePattern(w, h, _random_max_points);

		cuSample::SetCuttingPlaneParams(g_CP_A, g_CP_B, g_CP_C, g_CP_D, g_CP_THRESHOLD);
		if(_sampling_method == RAW)
			cuSample::CreateUniformSamplePattern(w, h, 1);

		// create a point cloud
		image_to_point_cloud(imgBuf);

#else
		//TODO switch filter on and off
		PCU::depthData_to_PointCloud(_filter.apply_filter(imgBuf), _point_cloud_vector, _point_normals_original, _point_normals_selected_normals, _focal_length_x, w, h);

#endif

	}
	else {
		_simSensor->requestCameraSimulationFrame();
	}
	return true;
}




/*
Set the sensor to autorun mode
*/
bool  PointCloudProducer::setAutoRun(bool enable)
{
	_is_running = enable;

	if (!_sensor_connected) {
		connectToCamera();
		if (!_sensor_connected) {
			return false;
		}
	}


	if (_is_running)  // activate autorun 
	{

#ifdef _WTIH_CUDA_PC
		cuSample::CreateUniformSamplePattern(_depth_width, _depth_height, _uniform_step_density);
		cuSample::CreateRandomSamplePattern(_depth_width, _depth_height, _random_max_points);
		cuSample::SetCuttingPlaneParams(g_CP_A, g_CP_B, g_CP_C, g_CP_D, g_CP_THRESHOLD);
		if (_sampling_method == RAW)
			cuSample::CreateUniformSamplePattern(_depth_width, _depth_height, 1);
#endif

		if (_sensor_connected>1) 
		{
			//Its a simulated sensor
			_simSensor->autoRunCameraSimulation(true);
		}
		else 
		{
			static bool addListenerOnce = true;
			if (addListenerOnce) 
			{
				//_frameHandler->setSimulation(this);
				addListenerOnce = false;
			}
			_sensor->getDepthStream().addNewFrameListener(_frameHandler);

			// Initiate timer
			_autorun = true;

		}


	}
	else // deactivate autorun 
	{  
		if (_sensor_connected>1) 
		{
			//Its not an OpenNI Camera
			_simSensor->autoRunCameraSimulation(false);
		}
		else 
		{
			// Invalidate timer
			if (_frameHandler && _autorun) {
				_sensor->getDepthStream().removeNewFrameListener(_frameHandler);
				_autorun = false;
			}
		}
	}

}



/*
Function is called when a new frame is ready;
*/
void  PointCloudProducer::newFrameReady(void)
{
	//if (_is_running) {
	//	return;
	//}

	if (_simSensor == NULL ) 
	{
		//TODO switch filter on and off
		//TODO GET ACTUAL FOCAL LENGTH
		openni::VideoFrameRef frame;
		openni::Status status = _frameHandler->get_New_Frame_Depth_Data()->readFrame(&frame);
		
		if (status != openni::STATUS_OK) {
			return; 
		}

		//TODO save in super Struct
		uint16_t* imgBuf = (uint16_t*)frame.getData();
		int h = frame.getHeight();
		int w = frame.getWidth();


		// stores the last cv image
		_depth_16U = cv::Mat(h, w, CV_16UC1, (unsigned char*)imgBuf);
		
		// Crete a point cloud
		image_to_point_cloud(imgBuf);

	}
	else 
	{
		if(_simSensor->isSimulatedSensorRunning())
			PCU::depthData_to_PointCloud(_filter.apply_filter(_simSensor->get_New_Simulated_Frame()), _point_cloud_vector, _point_normals_original, _point_normals_selected_normals, _simSensor->getFocalLength(), _simSensor->getDepthW(), _simSensor->getDepthH());
	}

	// Call the tracking sim function
	if (callback_function)
		callback_function();
}


double sum_time = 0;
static int frames = 0;
/*
Sample the point set
*/
void  PointCloudProducer::image_to_point_cloud(unsigned short* imgBuf)
{

	switch (_sampling_method )
	{
	case RAW:
	case UNIFORM:
	{
		process_uniform_sampling(imgBuf);
		break;
	}
	case RANDOM:
		process_random_sampling(imgBuf);
		break;
	default:
		break;
	}

}




/*
Process the uniform sampling point cloud generation
*/
void  PointCloudProducer::process_uniform_sampling(unsigned short* imgBuf)
{
#ifdef _WTIH_CUDA_PC


#ifdef TIMEFRAME
	double start_t = (double)std::clock() / CLOCKS_PER_SEC;
	frames++;
#endif

	vector<float3> points_host(_depth_width * _depth_height);
	vector<float3> normals_host(_depth_width * _depth_height);

	
	cuSample::UniformSampling((unsigned short*)imgBuf, _depth_width, _depth_height, _focal_length_x, _focal_length_y, _cx, _cy, _normal_vector_stride, g_CP_ACTIV, points_host, normals_host);


	_point_cloud_vector.clear();
	_point_normals_original.clear();
	_point_normals_selected_normals.clear();

	int index = 0;
	for (int i = 0; i < points_host.size(); i++)
	{
		if (points_host[i].x == 0.0 && points_host[i].y == 0 && points_host[i].z == 0) continue;

		dPoint d = mycv::dPoint(points_host[i].x, points_host[i].y, points_host[i].z);
		d.setNormalIndex(index); index++;
		_point_cloud_vector.push_back(d);
		//_point_normals_original.push_back(osg::Vec3f(normals_host[i].x, normals_host[i].y, normals_host[i].z));
		_point_normals_selected_normals.push_back(osg::Vec3f(normals_host[i].x, normals_host[i].y, normals_host[i].z));
	}



#ifdef TIMEFRAME

	double stop_t = (double)std::clock() / CLOCKS_PER_SEC;
	sum_time += stop_t - start_t;

	if (frames % 30 == 0) {
		_cprintf("\nms/frame:  %lf", sum_time / 30 * 1000);
		sum_time = 0;
	}
#endif

#else
#define TIMEFRAME
#ifdef TIMEFRAME
	double start_t = (double)std::clock() / CLOCKS_PER_SEC;
	frames++;
#endif


	PCU::depthData_to_PointCloud(_filter.apply_filter(imgBuf), _point_cloud_vector, _point_normals_original, _point_normals_selected_normals, _focal_length_x, _depth_width, _depth_height);


#ifdef TIMEFRAME

	double stop_t = (double)std::clock() / CLOCKS_PER_SEC;
	sum_time += stop_t - start_t;

	if (frames % 30 == 0) {
		_cprintf("\nms/frame:\t%lf", sum_time / 30 * 1000);
		sum_time = 0;
	}
#endif

#endif

}


/*
Process the random sampling point cloud generation
*/
void  PointCloudProducer::process_random_sampling(unsigned short* imgBuf)
{
#ifdef _WTIH_CUDA_PC
	vector<float3> points_host(_depth_width * _depth_height);
	vector<float3> normals_host(_depth_width * _depth_height);

	cuSample::RandomSampling((unsigned short*)imgBuf, _depth_width, _depth_height, _focal_length_x, _normal_vector_stride, g_CP_ACTIV, points_host, normals_host);

	_point_cloud_vector.clear();
	_point_normals_original.clear();
	_point_normals_selected_normals.clear();

	int index = 0;
	for (int i = 0; i < points_host.size(); i++)
	{
		if (points_host[i].x == 0.0 && points_host[i].y == 0 && points_host[i].z == 0) continue;

		dPoint d = mycv::dPoint(points_host[i].x, points_host[i].y, points_host[i].z);
		d.setNormalIndex(index); index++;
		_point_cloud_vector.push_back(d);
		//_point_normals_original.push_back(osg::Vec3f(normals_host[i].x, normals_host[i].y, normals_host[i].z));
		_point_normals_selected_normals.push_back(osg::Vec3f(normals_host[i].x, normals_host[i].y, normals_host[i].z));

	}

#else
	PCU::depthData_to_PointCloud(_filter.apply_filter(imgBuf), _point_cloud_vector, _point_normals_original, _point_normals_selected_normals, _focal_length, _depth_width, _depth_height);
#endif
}





/*
Returns the autorun status
*/
bool PointCloudProducer::getAutoRunStatus(void)
{
	return _autorun;
}


/*
LEGACY FUNCTION
This function is used to overwrite the focal length set by the frame handler
@param focal_length - the focal length.
*/
void  PointCloudProducer::setFocalLengthOverride(float focal_length)
{
	_frameHandler->set_focal_length(focal_length);
}


/*
Override the default intrinsic parameters for the range camera.
@param fx, fy - the focal length
@param cx, cy - the principle point
*/
void PointCloudProducer::setDepthCamIntrinsic(double fx, double fy, double cx, double cy)
{
	_focal_length_x = fx;
	_focal_length_y = fy;
	_cx = cx;
	_cy = cy;

}


/*
Stores a frame to drive
*/
void PointCloudProducer::recordFrame(bool r)
{
	if (_frameHandler)
		_frameHandler->readerRecordFrames(r);
}

/*
*/
void  PointCloudProducer::saveFrame(string filename)
{
	if (!_frameHandler)return;

	tracking_utils::DataWriter fw;
	fw.printFrames(filename, *_frameHandler->readerGetFrames());
}



/*
Saves a single depth frame to a phd.
@param filename - The name of the depth frame
*/
void  PointCloudProducer::saveDepthFrame(string filename)
{
	if (_sensor_connected < 2) {

		//Real openNI sensor
		openni::VideoStream& depth = _sensor->getDepthStream();

		//float focal_length= 361.6f;//TODO FIX FOCAL LENGTH GET FROM SOURCE
		openni::VideoFrameRef frame;
		openni::Status status = depth.readFrame(&frame);

		//_cprintf("HFOV: %lf\n", _sensor->getDepthStream().getHorizontalFieldOfView());
		//_cprintf("VFOV: %lf\n", _sensor->getDepthStream().getVerticalFieldOfView());

		if (status != openni::STATUS_OK) { return ; }

		//TODO save in super Struct
		uint16_t* imgBuf = (uint16_t*)frame.getData();

		int h = frame.getHeight();
		int w = frame.getWidth();


		ImageReaderWriter::saveToPng16(filename, (unsigned short*)imgBuf, w, h, 1, true);

	}
}




/*
Close the sensors and clean up the data
*/
void PointCloudProducer::cleanup(void)
{
	if (!_sensor) return;

	_sensor->cleanup();


#ifdef _WTIH_CUDA_PC		
	cuPCU::FreeDeviceMemory();
#endif
}


/*
Get the depth stream from the sensor
@return stream = video stream
*/
openni::VideoStream& PointCloudProducer::getDepthStream(int stream)
{
	return _sensor->getDepthStream();
}

/*
Get the color stream from the sensor
@return stream = the color stream
*/
openni::VideoStream* PointCloudProducer::getColorStream(int stream)
{
	return _sensor->getColorStream();
}


/*
Return the depth stream as cv image pointer
@param stream_idx - the id of the stream in case multiple streasm are fetched
@return the location of the last depth image.
*/
cv::Mat* PointCloudProducer::getCVDepthStream_16UC1(int stream_idx )
{
	return &_depth_16U;
}


/*
Enable or disable the bilateral filter
@param enable - true enables the filter
*/
void PointCloudProducer::setBilateralFilter(bool enabled)
{
	_filter.set_using_bilateral_filter(enabled);
}



/*
LEGACY CODE
Set the name of the file from which camera data can be loaded.
@param file - a string with path and name to the point cloud file;
*/
int PointCloudProducer::connectToCameraFile(string file)
{
	_cameraSimulationFile = file;

	return 2;
}


/*
In case the data is loaded from a file, jump to a particular frame
@param frame_idx = an interger with teh frame indes.
*/
int PointCloudProducer::cameraFileSetFrame(int frame_idx)
{
	if (!_simSensor)return -1;

	_simSensor->simulatedSensorJumpToFrame(frame_idx);
}


/*
Return the width of the depth image in pixel
*/
int PointCloudProducer::getDepthWidth(void)
{
	//if (!_simSensor)return -1;
	return _sensor->depth_w();
}

/*
Return the height of the depth image in pixel
*/
int  PointCloudProducer::getDepthHeight(void)
{
	//if (!_simSensor)return -1;
	return _sensor->depth_h();
}


/*
Set the density for uniform sampling
*/
void  PointCloudProducer::setUniformSamplingParams(int step_density)
{
	_uniform_step_density = step_density;
}



/*
Set the parameters for the random sampling filter.
Note that either max_points or percentage can be used. The
other must be set to -1.0
@param max_points - the max. number of points.
@param perventage - the percentage of points that should be used.
*/
void PointCloudProducer::setRandomSamplingParams(int max_points, float percentage)
{
	if (max_points > 0)
		_random_max_points = max_points;

	if (percentage > 0.01 && max_points <= 1.0)
		_random_percentage = percentage;

}


/*
Set the sampling method
@param method - RAW = 0,
UNIFORM = 1,
RANDOM = 2,
*/
void  PointCloudProducer::setSamplingMethod(SamplingMethod method)
{
	_sampling_method = method;
}


/*
Set parameters for the normal vector calculation
@param image_stride - the stride that the normal vector calculator should move to the left, right, up, down..
*/
void PointCloudProducer::setNormalVectorParams(int image_stride)
{
	if (image_stride > 0)
		_normal_vector_stride = image_stride;
}