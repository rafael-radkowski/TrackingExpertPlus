/*!
@class   AzureViewer

@brief  Converts the Azure Kinect 2 API to a ICaptureDevice 

This class converts a Kinect into a ICaptureDevice so it can be used with point clouds. The user does not need to know kinect API to use this class.
Also allows the user to create a recording(see AzureRecording.h to access them)

Features:
- Get depth and color images live
- Get dimensions
- Create recordings for AzureRecordings
- Change certain specs of the camera




Tyler Ingebrand
Iowa State University
tyleri@iastate.edu
Undergraduate

-------------------------------------------------------------------------------
Last edited:

Apr 7, 2020, Tyler Ingebrand
- Added documentation

*/
#pragma once

#pragma comment(lib, "k4a.lib")
#include <k4a/k4a.h>
#include <k4arecord/types.h>
#include <k4arecord/record.h>


#include <mutex>

// opencv
#include <opencv2/opencv.hpp>

#include <iostream>
#include "ICaptureDevice.h"



class KinectAzureCaptureDevice : public texpert::ICaptureDevice
{
private: //member variables
	
	k4a_device_t *device;
	char *serial;
	k4a_device_configuration_t config = K4A_DEVICE_CONFIG_INIT_DISABLE_ALL;
	k4a_capture_t *capture_handle;
	uint32_t index;
	bool cameraRunning = false;
	k4a_record_t *record;;

	//defaults
	int color_width = 0;
	int color_height = 0;
	int depth_width = 0; // 1024 x 1024 is WFOV unbinned
	int depth_height = 0;
		
	cv::Mat		_camera_param;
public:
	/*!
	Enum for selecting which camera functions to activate.Possible camera functions are RGB, IR, and Depth(D)
	
	*/
	enum class Mode {RGB, RGBIRD};
	/*!
	Enum for selecting which image type is desired, RGB, IR, or Depth
	*/
	enum class ImageType {RGB, IR, Depth};
	/*!
	Enum for selecting color image data type, RGBA or JPEG format
	*/
	enum class ColorType { RGBA, JPEG };


	/*
	Static function to detect number of cameras attached
	@return The number of cameras attached
	*/
	static int getNumberConnectedCameras();
	
	/*!
   Constructor. Assumes camera index 0, RGB, IR, and Depth images. 15 FPS. 
   */
	KinectAzureCaptureDevice();
	//records index and initial config
	
	
	
	/*
	@brief This constructor initializes a camera object with your configurations
	It does not start the cameras, to do so use startCameras()
	By default, the camera operates at 15 FPS and 1280 x 720 pixels. This can be changed with changeFPS and changeResolution
	@param the index of the camera you want to access. 0 is the default index you should use if you are unsure.
	@param The mode you want to set the camera to. You can choose to only take RGB, IR, or Depth pictures, or a combonation of them. See the enum Mode for available options.
	*/
	KinectAzureCaptureDevice(uint32_t cameraNumber, KinectAzureCaptureDevice::Mode cameras);
	
	/*
	*/
	~KinectAzureCaptureDevice();



	/*
	@brief this connects to the device, preventing other applications from using it. This method must be called before all other methods (except the constructor)
	@return true if it successfully connects, false otherwise
	This will fail if there is not an avaiable camera at the index provided in the constructor
	
	*/
	bool connectToDevice();
	



	/*
	@brief activates the cameras for you to use. You must call connectToDevice before this function.
	@return True if successful, false otherwise

	*/
	bool startCamera();



	/*
	@brief returns a k4a_capture_t from the camera. This is used internally as a helper method. This capture contains a buffer from all the camera picture types that are active.
	You must start the camera before calling this.
	@return the k4a_capture_t at that given moment.
	*/
	k4a_capture_t getCapture();



	/*
	@brief a helper method used internally. Returns the image desired of a given type from the current capture.
	You must have that image type available active, or it returns null
	@param The image type desired. See the enum ImageType for options
	@return the image type requested (RGB, IR, or Depth)
	*/
	k4a_image_t getImage(KinectAzureCaptureDevice::ImageType img);




	/*
	@brief updates the height and width dimensions according to the dimensions of the image type desired. Useful for programs that require the height and width to be known, such as OpenCV. You must connect to and open the camera before calling
	@param type The type of image you want to know the dimensions of
	@param width the variable to store the width in
	@param height the variable to sore the height in
	*/
	void updateDimensions(KinectAzureCaptureDevice::ImageType type, int *width, int *height);
	
	
	
	/*
	@brief returns the buffer for the desired image type. You must connect to and start camera before calling.
	@param The desired image type, either RGB, IR, or Depth
	@return an array of unsigned, 8 bit ints
	*/
	uint8_t* getBuffer(KinectAzureCaptureDevice::ImageType type);


	/*
	@brief changes the FPS configurations for the camera. After using this function, the camera will stop and you must restart it. You must connect to camera before using this method.
	@param frameRate the framerate desired. Can only be 5, 15, or 30 due to camera constraints
	
	*/
	void changeFPS(int frameRate);
	

	/*
	@brief changes the resolution configurations for the camera. After using this function, the camera will stop and you must restart it. You must connect to camera before using this method.
	@param res The resolution desired. It can only be a few preset ratios:
			 1280 * 720 (16:9)  
			 1920 * 1080 (16:9)  
			 2560 * 1440 (16:9)  
			 2048 * 1536 (4:3)  
			 3840 * 2160 (16:9)  
			 4096 * 3072 (4:3)  
			 Enter your choice as the second integer in the dimensions. EX: 720 

	*/
	void changeResolution(int res);
	
	
	
	/*!
	Changes the color type format. Must change the format to JPEG for recording
	@param type - the color format type desired
	*/
	void changeColorType(KinectAzureCaptureDevice::ColorType type);

	/*!
	Creates a recording file at the filepath given. See savecapture() and EndRecording(). 
	@param title - the total file path and file name of the video that will be created
	*/
	void createRecording(char* title);

	/*!
	Saves a capture(frame) to the recording created. 
	*/
	void saveCapture();
	/*!
	Flushes all data to recording file, saves recording file. You must call this after you are done recording
	*/
	void endRecording();


	/*
	@brief stops the camera. After calling, it will no longer return image buffers. Must have started connected to and started camera before calling
	*/

	void stopCamera();

	/*
	@brief returns the serial number of the device you are using
	@return the serial number
	*/
	char* getSerialNumber();






	//Icapture device required functions -----------------------------------------------------------------------------------------------------------------
	
	//deconstructor doesn't work, but default one c++ creates works. Leaving it unimplemented for now/
	/*~AzureViewer()
	{
		if(cameraRunning == true)
			this->stopCamera();
		if((*device)->is_valid())
			k4a_device_close(*device);	
		
	}*/

	/**
	 * Stores the current RGB frame in the frame provided
	 * @param mFrame - location to store the frame
	 */
	virtual void getRGBFrame(cv::Mat &mFrame);

	/**
	 Stores the current depth frame in the frame provided	
	 @param mFrame - location to store the frame
	*/
	virtual void getDepthFrame(cv::Mat &mFrame);
	
	/**
	 Stores the current IR frame in the frame provided
	 @param mFrame - location to store the frame
	*/
	void getIRFrame(cv::Mat &mFrame);


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
	virtual int getRows(texpert::CaptureDeviceComponent c);

	/*
	Return the number of image colums in pixel
	@param c - the requested camera component.
	@return - number of image columns in pixel. -1 if the component does not exist.
	*/
	virtual int getCols(texpert::CaptureDeviceComponent c);
	

	/*!
	Return the intrinsic camera parameters
	@return 3x3 cv::Mat with
		[ fx 0 cx ]
		[ 0 fy cy ]
		[ 0 0  1  ]
	*/
	virtual cv::Mat& getCameraParam(void);


};

