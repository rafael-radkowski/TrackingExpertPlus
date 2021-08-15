#pragma once
/*
class AzureKinectFromMKV

The class reads Azure Kinect RGB and depth data from an MKV file and processes it. 

Note the class is an unfinished prototype. 
ToDo:
1) It leaks memory somewhere. 
2) The EOF is not correctly handled. The class plays the video in a loop. 
3) The trigger to get the next image is currently part of getDepthFrame(cv::Mat& mFrame),
which is not ideal. The class is made to drive a point cloud search. So the major data is the 
depth data. Nonetheless, this requires a better solution. 


Rafael Radkowski
Aug 2021
radkowski.dev@gmail.com
MIT License
--------------------------------------------------------------------------------
Last edits:

*/

#pragma comment(lib, "k4a.lib")
#include <k4a/k4a.h>
#include <k4arecord/types.h>
#include <k4arecord/record.h>
#include <k4arecord/playback.h>

#include <mutex>

// opencv
#include <opencv2/opencv.hpp>

#include <iostream>
#include "ICaptureDevice.h"


namespace texpert{

class AzureKinectFromMKV : public texpert::ICaptureDevice
{
public:

	/*
	Constructor. 
	Note that the constructor does not automatically opens and loads the file. 
	Call connectToDevice to get the file opened. 
	@param path_and_file - string with the path and filename of a video file. Needs to be an MKV file at this moment. 
	*/
	AzureKinectFromMKV(std::string path_and_file);

	~AzureKinectFromMKV();


	/*
	Open a given mkv video file. 
	*/
	bool connectToDevice(void);


	/**
	* Return the RGB frame
	* @param mFrame - location to store the frame
	*/
	virtual void getRGBFrame(cv::Mat& mFrame);

	/**
	Return the depth frame.
	@param mFrame - location to store the frame.
	mFrame is an OpenCV image of type CV_32FC1. PointCloudProducer requirec CV_32FC1 as input.
	*/
	virtual void getDepthFrame(cv::Mat& mFrame);


	/**
	 * Returns if the capture device is open
	 * @return If the capture device is open
	 */
	virtual bool isOpen();

	/*
	Return the number of image rows in pixel
	@param c - the requested camera component.
	@return - number of image rows in pixel. -1 if the component does not exist.
	*/
	virtual int getRows(CaptureDeviceComponent c);

	/*
	Return the number of image colums in pixel
	@param c - the requested camera component.
	@return - number of image columns in pixel. -1 if the component does not exist.
	*/
	virtual int getCols(CaptureDeviceComponent c);


	/*!
	Return the intrinsic camera parameters
	@return 3x3 cv::Mat with CV_32 values
		[ fx 0 cx ]
		[ 0 fy cy ]
		[ 0 0  1  ]
	*/
	virtual cv::Mat& getCameraParam(void);


private:

	/*

	*/
	std::vector<float> getCalibration(texpert::CaptureDeviceComponent component);


	/*
	
	*/
	void updateCameraParams(void);


	/*

	*/
	float getMetricRadius(texpert::CaptureDeviceComponent component);


	/*

	*/
	void updateDimensions(void);


	/*
	Get the color resolution from the playback info
	@param mode - color resolution model
	@param rows - rows in pixels. 0 if the image does not exist in the playback.
	@param cols - columns in pixels. 0 if the image does not exist in the playback.
	*/
	void getColorRes(k4a_color_resolution_t mode, int& rows, int& cols);


	/*
	Get the color resolution from the playback info
	@param mode - depth mode
	@param rows - rows in pixels. 0 if the image does not exist in the playback.
	@param cols - columns in pixels. 0 if the image does not exist in the playback.
	*/
	void getDepthRes(k4a_depth_mode_t mode, int& rows, int& cols);


	char* serial;
	k4a_device_configuration_t config;

	uint32_t index;
	bool cameraRunning = false;
	k4a_record_t* record;;



	k4a_capture_t*				_capture_handle;
	k4a_playback_t*				_playback;
	char*						_serial;
	bool 						_is_open;

	//calibration info so we do not have to re calculate open CV formats constantly
	cv::Mat _cv_camera_matrix_depth; //float 3x3 matrix, meant to be camera matrix. Should be		fx 0 Cx
																															//	0 fy cy
																															//	0  0  1
	cv::Mat _distortion_coefficients_depth; // float 1x8 matrix. k1, k2, p1, p2, k3, k4, k5, k6

	std::string					_path_and_file;


	//camera parameters
	int			_color_width;
	int			_color_height;
	int			_depth_width; // 1024 x 1024 is WFOV unbinned
	int			_depth_height;

	cv::Mat		_camera_param;



};


}// namespace texpert 