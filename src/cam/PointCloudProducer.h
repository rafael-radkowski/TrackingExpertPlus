#pragma once
/*
class PointCloudProducer

The class point cloud producer produces a point cloud from one or more camera images.
It implements the interface to OpenNI cameras, provides sampling and filter algorithms, 
and returns the complete point cloud. 

Set a function via  void setCallbackPtr(std::function<void()> callback);

The point cloud producer will call this callback as soon as a new image is ready. 

Features:
- Camera connection
- Point cloud Sampling
- Bilateral fitering
- Cutting plane and point cloud cropping functions

Rafael Radkowski
Iowa State University
rafael@iastate.edu
Oct. 20, 2017
MIT License
---------------------------------------------------------------

Last Changes:

May 5th, 2018,RR
- store the last depth image in _depth_16U 
- added a api to access the location of the last depth image. 
July 12, 2019
- Removed a propriatory camera from the set.
*/

#include <opencv2\opencv.hpp>

#include "Tracking.h"
#include "SensorConfig.h"
#include "SampleTypes.h"
#include "Sensor.h"
#include "SensorWatcher.h"
#include "FrameReader.h"
#include "SimulatedSensor.h"
#include "PCU.h"
#include "Filters.h"
#include "SampleTypes.h"

// local cuda
#include "cuda/cuPCU.h"

using namespace std;

namespace texpert{


class PointCloudProducer
{
public:

	/*
	Constructor
	@param dst_points - reference to a container for the point cloud. 
	@param dst_normals - reference to a container for the normal vectors. 
	*/
	PointCloudProducer(vector<dPoint>& dst_points, vector<Vec3>& dst_normals );

	/*
	Destructor
	*/
	~PointCloudProducer();

	/*
	LEGACY CODE
	Set a camera index. 
	@param camera_index - the index of the camera ot pick.
	*/
	void setCameraIndex(int camera_index);

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
	@return int - if successfully returns the connect status, 0 failed, 1 conected to camera
	*/
	int connectToCamera(int camera_id = 0, int rgb_width = -1, int rgb_height = -1, int depth_width = -1, int depth_height = -1, int fps = -1);

	/*
	Load a single frame from a connected camera;
	*/
	bool loadSingleFrame(void);

	/*
	Set the sensor to autorun mode. With auto mode active, 
	the point cloud producer will listen to new camera images
	and fire a message as soon as they arrive. 
	*/
	bool setAutoRun(bool enable);

	/*
	LEGACY FUNCTION
	This function is used to overwrite the focal length set by the frame handler
	@param focal_length - the focal length.
	*/
	void setFocalLengthOverride(float focal_length);

	/*
	Returns the autorun status
	*/
	bool getAutoRunStatus(void);

	/*
	Set the callback function that is called when a point cloud
	is ready being processed. 
	*/
	void setCallbackPtr(std::function<void()> callback);

	/*
	Close the sensors and clean up the data
	*/
	void cleanup(void);

	/*
	Override the default intrinsic parameters for the range camera.
	@param fx, fy - the focal length
	@param cx, cy - the principle point
	*/
	void setDepthCamIntrinsic(double fx, double fy, double cx, double cy);

	/*
	Enable or disable the bilateral filter
	@param enable - true enables the filter
	*/
	void setBilateralFilter(bool enabled);

	/*
	Get the depth stream from the sensor
	@param stream_idx - the id of the stream in case multiple streasm are fetched
	@return stream = video stream
	*/
	openni::VideoStream& getDepthStream(int stream_idx = 0);

	/*
	Return the depth stream as cv image pointer
	@param stream_idx - the id of the stream in case multiple streasm are fetched
	@return the location of the last depth image. 
	*/
	cv::Mat* getCVDepthStream_16UC1(int stream_idx = 0);

	/*
	Get the color stream from the sensor
	@param stream_idx - the id of the stream in case multiple streasm are fetched
	@return stream = the color stream
	*/
	openni::VideoStream* getColorStream(int stream_idx = 0);


	/*
	Return the width of the depth image in pixel
	*/
	int getDepthWidth(void);

	/*
	Return the height of the depth image in pixel
	*/
	int getDepthHeight(void);


	/*
	Set the density for uniform sampling
	*/
	void setUniformSamplingParams(int step_density);

	/*
	Set the parameters for the random sampling filter. 
	Note that either max_points or percentage can be used. The
	other must be set to -1.0
	@param max_points - the max. number of points.
	@param perventage - the percentage of points that should be used. 
	*/
	void setRandomSamplingParams(int max_points, float percentage);


	/*
	Set the sampling method
	@param method - RAW = 0,
					UNIFORM = 1,
					RANDOM = 2,
	*/
	void setSamplingMethod(SamplingMethod method);


	/*
	Set parameters for the normal vector calculation
	@param image_stride - the stride that the normal vector calculator should move to the left, right, up, down..
	*/
	void setNormalVectorParams(int image_stride);


	/*
	Stores a frame.
	*/
	void recordFrame(bool r);

	/*
	*/
	void saveFrame(string filename);


	/*
	Saves a single depth frame to a phd. 
	@param filename - The name of the depth frame
	*/
	void saveDepthFrame(string filename);

	//---------------------------------------------------------------------------------------------------------------------------
	// Simulated sensor data

	/*
	LEGACY CODE
	Set the name of the file from which camera data can be loaded.
	@param file - a string with path and name to the point cloud file;
	*/
	int connectToCameraFile(string file);


	/*
	In case the data is loaded from a file, jump to a particular frame
	@param frame_idx = an interger with teh frame indes.
	*/
	int cameraFileSetFrame(int frame_idx);



private:


	/*
	Function is called when a new frame is ready;
	*/
	void newFrameReady(void);


	/*
	Sample the point set
	*/
	void image_to_point_cloud( unsigned short* imgBuf);


	/*
	Process the uniform sampling point cloud generation 
	*/
	void process_uniform_sampling(unsigned short* imgBuf);


	/*
	Process the random sampling point cloud generation
	*/
	void process_random_sampling(unsigned short* imgBuf);


	//------------------------------------------------------------------------------
	Sensor*									_sensor;
	DepthStreamListener*                    _frameHandler;
	Filters									_filter;

	int										_sensor_connected;
	int										_sensor_index;
	SensorType								_sensor_type;
	double									_focal_length_x;
	double									_focal_length_y;
	double									_cx;
	double									_cy;
	bool									_is_running;
	bool									_autorun;

	int										_depth_width;
	int										_depth_height;

	//------------------------------------------------------------------------------
	// Sampling
	int										_uniform_step_density;
	int										_random_max_points;
	float									_random_percentage;
	SamplingMethod							_sampling_method;



	int										_normal_vector_stride;

	//------------------------------------------------------------------------------
	// Simulated sensor
	string									_cameraSimulationFile;
	SimulatedSensor*						_simSensor;	

	//------------------------------------------------------------------------------
	// Reference to the output point clouds
	vector<dPoint>&							_point_cloud_vector;
	vector<Vec3>&							_point_normals_original;// the normals for the raw Kinect data
	vector<Vec3>&							_point_normals_selected_normals;


	//------------------------------------------------------------------------------
	// Image storage

	// stores the last depth image as cv::Mat
	cv::Mat									_depth_16U;


	//------------------------------------------------------------------------------
	// Function pointer, which is called when the point cloud is ready to be processed
	std::function<void()> callback_function;

};

} //texpert