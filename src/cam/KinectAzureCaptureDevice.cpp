
#pragma once

//#pragma comment(lib, "k4a.lib")
#include <k4a/k4a.h>
#include <k4arecord/types.h>
#include <k4arecord/record.h>


#include <mutex>

// opencv
#include <opencv2/opencv.hpp>

#include <iostream>
#include "ICaptureDevice.h"

#include "KinectAzureCaptureDevice.h"

int KinectAzureCaptureDevice::getNumberConnectedCameras()
{
	return k4a_device_get_installed_count();
}

	KinectAzureCaptureDevice::KinectAzureCaptureDevice()
	{
		index = 0;
		config.camera_fps = K4A_FRAMES_PER_SECOND_30;
		config.color_format = K4A_IMAGE_FORMAT_COLOR_BGRA32;
		config.color_resolution = K4A_COLOR_RESOLUTION_720P;
		config.synchronized_images_only = true;
		config.depth_mode = K4A_DEPTH_MODE_WFOV_UNBINNED;
		config.wired_sync_mode = K4A_WIRED_SYNC_MODE_STANDALONE;


		color_width = 1280;
		color_height = 720;
		depth_width = 640; // 1024 x 1024 is WFOV unbinned
		depth_height = 576;

		
		//depth_width = 1024; // 1024 x 1024 is WFOV unbinned
		//depth_height = 1024;

		capture_handle = (k4a_capture_t*)malloc(8);
		record = (k4a_record_t*)malloc(8);
		device = (k4a_device_t*)malloc(8);

		connectToDevice();
	}
	
	KinectAzureCaptureDevice::KinectAzureCaptureDevice(uint32_t cameraNumber, KinectAzureCaptureDevice::Mode cameras, bool syncMode)
	{
		//config that does not vary
		index = cameraNumber;
		config.camera_fps = K4A_FRAMES_PER_SECOND_15; //for color and depth
		config.color_format = K4A_IMAGE_FORMAT_COLOR_BGRA32; //default K4A_IMAGE_FORMAT_COLOR_BGRA32, trying mjpg
		config.color_resolution = K4A_COLOR_RESOLUTION_720P;

		//color default size
		color_width = 1280;
		color_height = 720;

		//if we are wired togehotr for sync,
		if (syncMode == true)
		{

			if (index == 0)
			{
				std::cout << "Sync mode: Master camera is at index 0.\n";
				config.wired_sync_mode = K4A_WIRED_SYNC_MODE_MASTER; //0 is always master
			}
			else
			{
				config.wired_sync_mode = K4A_WIRED_SYNC_MODE_SUBORDINATE; // all others are subordinate
				config.subordinate_delay_off_master_usec = index * 160; //need some delay for lasers on depth camera to not interfere with each other.
			}
		}
		else  config.wired_sync_mode = K4A_WIRED_SYNC_MODE_STANDALONE;

		switch (cameras)
		{
			//RGB camera defaults to 15 FPS, BGRA32, and 720 P
			//captures are syncronized, meaning every capture has all image types. This may reduce framerate, but it is a lot easier to use in code because captures behave uniformly
		case KinectAzureCaptureDevice::Mode::RGB:
			config.depth_mode = K4A_DEPTH_MODE_OFF;
			break;

		case KinectAzureCaptureDevice::Mode::RGBIRD:
			config.synchronized_images_only = true;
			config.depth_mode = K4A_DEPTH_MODE_NFOV_UNBINNED;
			depth_width = 640; // 1024 x 1024 is WFOV unbinned
			depth_height = 576;
			break;

		default:
			printf("Invalid configuration. Defaulting to RGB, IR, and Depth cameras active. \n");
			config.synchronized_images_only = true;
			config.depth_mode = K4A_DEPTH_MODE_NFOV_UNBINNED;
			depth_width = 1024; // 1024 x 1024 is WFOV unbinned
			depth_height = 1024;
			break;

		}
		//init capture handle
		capture_handle = (k4a_capture_t*)malloc(8);
		record = (k4a_record_t*)malloc(8);
		device = (k4a_device_t*)malloc(8);

		connectToDevice();
	}

	std::vector<float> KinectAzureCaptureDevice::getCalibration(texpert::CaptureDeviceComponent c)
	{
		k4a_calibration_t calib;
		k4a_result_t result = k4a_device_get_calibration(*device, config.depth_mode, config.color_resolution, &calib);
		if (result == K4A_RESULT_FAILED)
		{
			return std::vector<float>(1, 0.0); // bad value
		}
		_k4a_calibration_camera_t cameraCalib;
		switch (c)
		{
		case texpert::DEPTH:
			cameraCalib = calib.depth_camera_calibration;
			break;
		case texpert::COLOR:
			cameraCalib = calib.color_camera_calibration;
			break;
		}

		k4a_calibration_intrinsics_t intrinsics = cameraCalib.intrinsics;
		k4a_calibration_intrinsic_parameters_t params = intrinsics.parameters;
		//copy data over
		std::vector<float> ret(14, 0.0);
		for (int i = 0; i < 14; i++)
		{
			ret[i] = params.v[i];
		}


		return ret;


	}
	KinectAzureCaptureDevice::~KinectAzureCaptureDevice()
	{
		stopCamera();
		//free(record);
		//free(device);
		//free(capture_handle);
	}

	

	bool KinectAzureCaptureDevice::connectToDevice()
	{
		//check to see if we can get the device
		uint32_t count = k4a_device_get_installed_count();
		if (count <= index)
		{
			printf("No deviced listed under the index you choose!\n To troubleshoot, insure the device is plugged in and that no other application is using it.\n");
			return false;
		}
		//attempt to open device
		if (K4A_FAILED(k4a_device_open(index, device)))
		{
			printf("Failed to open k4a device!\n");
			return false;
		}

		// Get the size of the serial number
		size_t serial_size = 0;
		k4a_device_get_serialnum(*device, NULL, &serial_size);

		// Allocate memory for the serial, then acquire it
		serial = (char*)(malloc(serial_size));
		k4a_device_get_serialnum(*device, serial, &serial_size);
		printf("Opened device: %s\n", serial);
		printf("Device index: %d\n", index);

		_camera_param = cv::Mat::eye(3,3, CV_32F);
		k4a_calibration_t calibration;
		if (K4A_RESULT_SUCCEEDED !=  k4a_device_get_calibration(*device, config.depth_mode, config.color_resolution, &calibration))
		{
			printf("Failed to get calibration\n");
		}else
		{
			_camera_param.at<float>(0,0) = calibration.depth_camera_calibration.intrinsics.parameters.param.fx;
			_camera_param.at<float>(1,1) = calibration.depth_camera_calibration.intrinsics.parameters.param.fy;
			_camera_param.at<float>(0,2) = calibration.depth_camera_calibration.intrinsics.parameters.param.cx;
			_camera_param.at<float>(1,2) = calibration.depth_camera_calibration.intrinsics.parameters.param.cy;
		}

		return startCamera();
	}


	
	bool KinectAzureCaptureDevice::startCamera()
	{
		bool in, out;
		if (config.wired_sync_mode == K4A_WIRED_SYNC_MODE_MASTER)
		{
			k4a_device_get_sync_jack(*device, &in, &out);
			if (out == false)
			{
				std::cout << "Sync not possible, cable not connected to out on master device " << serial << "\n";
				return false;
			}
		}
		else if (config.wired_sync_mode == K4A_WIRED_SYNC_MODE_SUBORDINATE)
		{
			k4a_device_get_sync_jack(*device, &in, &out);

			if (in == false)
			{
				std::cout << "Sync not possible, cable not connected to input on device " << serial << "\n";
				return false;
			}
		}
		//start camera with given settings
		if (K4A_FAILED(k4a_device_start_cameras(*device, &config)))
		{
			std::cout << "Failed to start cameras!\n";
			std::cout << "Solutions : \n";
			std::cout << "\tMake sure no other application is using it.\n";
			std::cout << "\tCheck it is plugged in(there should be a solid white light on the back of the camera).\n";
			std::cout << "\tIf you are syncronizing via external cables, make sure the audio cables are connected.\n";
			std::cout << "\tFollow Azure Kinect API error trace.\n";
			return false;
		}

		//printf("Camera started.\n");
		cameraRunning = true;
		//need to add frame so it is not null
		k4a_capture_create(capture_handle);
		//adding images to frame
		k4a_image_t image;
		k4a_image_create(K4A_IMAGE_FORMAT_COLOR_BGRA32, color_width, color_height, 4 * color_width, &image);
		k4a_capture_set_color_image(*capture_handle, image);

		k4a_image_create(K4A_IMAGE_FORMAT_DEPTH16, depth_width, depth_height, 2 * depth_width, &image);
		k4a_capture_set_depth_image(*capture_handle, image);

		k4a_image_create(K4A_IMAGE_FORMAT_IR16, depth_width, depth_height, 2 * depth_width, &image);
		k4a_capture_set_ir_image(*capture_handle, image);
	

		return true;
	}


	k4a_capture_t KinectAzureCaptureDevice::getCapture()
	{
		//make sure camera is running
		if (cameraRunning == false)
		{
			printf("Camera isn't running. \n");
			return NULL;
		}

		k4a_capture_t *temp = (k4a_capture_t*)malloc(8);

		//get capture from camera. If there is not one currently, should not edit the current capture handle
		k4a_wait_result_t result = k4a_device_get_capture(*device, temp, 0);

		//if successful, replace pointer from capture handle to point to temp, and deallocate previous capture
		if (result == K4A_WAIT_RESULT_SUCCEEDED)
		{
			k4a_capture_release(*capture_handle);
			capture_handle = temp;
		}

		return *capture_handle;
	}



	
	k4a_image_t KinectAzureCaptureDevice::getImage(KinectAzureCaptureDevice::ImageType img)
	{
		this->getCapture();

		k4a_image_t ret = NULL;
		switch (img)
		{
		case KinectAzureCaptureDevice::ImageType::RGB:
			ret = k4a_capture_get_color_image(*capture_handle);
			break;
		case KinectAzureCaptureDevice::ImageType::IR:
			ret = k4a_capture_get_ir_image(*capture_handle);
			break;
		case KinectAzureCaptureDevice::ImageType::Depth:
			ret = k4a_capture_get_depth_image(*capture_handle);
			break;
		}
		return ret;
	}

	
	void KinectAzureCaptureDevice::updateDimensions(KinectAzureCaptureDevice::ImageType type, int *width, int *height)
	{
		k4a_image_t image = this->getImage(type);
		*height = k4a_image_get_height_pixels(image);
		*width = k4a_image_get_width_pixels(image);
		k4a_image_release(image);
	}
	
	uint8_t* KinectAzureCaptureDevice::getBuffer(KinectAzureCaptureDevice::ImageType type)
	{
		k4a_image_t image = this->getImage(type);

		uint8_t* ret = k4a_image_get_buffer(image);
		k4a_image_release(image);
		return ret;
	}



	
	void KinectAzureCaptureDevice::changeFPS(int frameRate)
	{
		//stops camera if running
		if (cameraRunning == true)
		{
			k4a_device_stop_cameras(*device);
			cameraRunning = false;
		}
		switch (frameRate)
		{

		case 5:
			config.camera_fps = K4A_FRAMES_PER_SECOND_5;
			break;
		case 15:
			config.camera_fps = K4A_FRAMES_PER_SECOND_15;
			break;
		case 30:
			config.camera_fps = K4A_FRAMES_PER_SECOND_30;
			break;
		default:
			printf("Invalid frame rate. Frame rate may only be 5, 15, or 30. Frame rate remains unchanged.");
			break;
		}

	}

	
	void KinectAzureCaptureDevice::changeResolution(int res)
	{
		//stops camera if running
		if (cameraRunning == true)
		{
			k4a_device_stop_cameras(*device);
			cameraRunning = false;
		}
		switch (res)
		{
		case 720:
			config.color_resolution = K4A_COLOR_RESOLUTION_720P;
			color_width = 1280;
			color_height = 720;
			break;
		case 3072:
			config.color_resolution = K4A_COLOR_RESOLUTION_3072P;
			color_width = 4096;
			color_height = 3072;
			break;
		case 2160:
			config.color_resolution = K4A_COLOR_RESOLUTION_2160P;
			color_width = 3840;
			color_height = 2160;
			break;
		case 1536:
			config.color_resolution = K4A_COLOR_RESOLUTION_1536P;
			color_width = 1280;
			color_height = 1536;
			break;
		case 1440:
			config.color_resolution = K4A_COLOR_RESOLUTION_1440P;
			color_width = 2560;
			color_height = 1440;
			break;
		case 1080:
			config.color_resolution = K4A_COLOR_RESOLUTION_1080P;
			color_width = 1920;
			color_height = 1080;
			break;
		default:
			printf("Invalid entry. Camera resolution can only be:\n");
			printf("1280 * 720 (16:9) \n");
			printf("1920 * 1080 (16:9) \n");
			printf("2560 * 1440 (16:9) \n");
			printf("2048 * 1536 (4:3) \n");
			printf("3840 * 2160 (16:9) \n");
			printf("4096 * 3072 (4:3) \n");
			printf("Enter your choice as the second integer in the dimensions. EX: 720\n");

		}

	}
	
	void KinectAzureCaptureDevice::changeColorType(KinectAzureCaptureDevice::ColorType type)
	{
		this->stopCamera();
		switch (type)
		{
		case KinectAzureCaptureDevice::ColorType::RGBA:
			config.color_format = K4A_IMAGE_FORMAT_COLOR_BGRA32;
			break;
		case KinectAzureCaptureDevice::ColorType::JPEG:
			config.color_format = K4A_IMAGE_FORMAT_COLOR_MJPG;
			break;
		}

	}

	//no recording in this project so commented out
	//void KinectAzureCaptureDevice::createRecording(char* title)
	//{
	//	//create file, will overwrite probably
	//	k4a_record_create(title, *device, config, record);
	//	//add tags for necesary variables in icapture device
	//	char * COLOR_WIDTH = (char*)malloc(8);
	//	char * COLOR_HEIGHT = (char*)malloc(8);
	//	char * DEPTH_HEIGHT = (char*)malloc(8);
	//	char * DEPTH_WIDTH = (char*)malloc(8);
	//	sprintf_s(COLOR_WIDTH, "%d", color_width);
	//	sprintf(COLOR_HEIGHT, "%d", color_height);
	//	sprintf(DEPTH_HEIGHT, "%d", depth_height);
	//	sprintf(DEPTH_WIDTH, "%d", depth_width);

	//	k4a_record_add_tag(*record, "COLOR_WIDTH", COLOR_WIDTH);
	//	k4a_record_add_tag(*record, "COLOR_HEIGHT", COLOR_HEIGHT);
	//	k4a_record_add_tag(*record, "DEPTH_HEIGHT", DEPTH_HEIGHT);
	//	k4a_record_add_tag(*record, "DEPTH_WIDTH", DEPTH_WIDTH);
	//	k4a_record_add_tag(*record, "SERIAL", serial);

	//	//write header, done before writing captures and after tags
	//	k4a_record_write_header(*record);
	//}

	//
	//void KinectAzureCaptureDevice::saveCapture()
	//{
	//	this->getCapture();
	//	k4a_record_write_capture(*record, *capture_handle);
	//}
	//
	//void KinectAzureCaptureDevice::endRecording()
	//{
	//	//k4a_record_flush(*record);
	//	k4a_record_close(*record);

	//}


	
	void KinectAzureCaptureDevice::stopCamera()
	{
		if (cameraRunning == true)
		{
			k4a_device_stop_cameras(*device);
			cameraRunning = false;
			k4a_device_close(*device);
			//printf("Camera %d stopped.\n", index);
		}
	}

	
	char* KinectAzureCaptureDevice::getSerialNumber()
	{
		return serial;
	}






	//Icapture device required functions -----------------------------------------------------------------------------------------------------------------

	

	
	void KinectAzureCaptureDevice::getRGBFrame(cv::Mat &mFrame)
	{
		uint8_t* rgbBuffer = getBuffer(KinectAzureCaptureDevice::ImageType::RGB);
		mFrame = cv::Mat(color_height, color_width, CV_8UC4, rgbBuffer);
	}

	
	void KinectAzureCaptureDevice::getDepthFrame(cv::Mat &mFrame)
	{
		uint8_t* depthBuffer = getBuffer(KinectAzureCaptureDevice::ImageType::Depth);
		cv::Mat img = cv::Mat(depth_height, depth_width, CV_16UC1, depthBuffer);
		img.convertTo(mFrame, CV_32FC1);
	}
	
	void KinectAzureCaptureDevice::getIRFrame(cv::Mat &mFrame)
	{
		uint8_t* irBuffer = getBuffer(KinectAzureCaptureDevice::ImageType::IR);
		mFrame = cv::Mat(depth_height, depth_width, CV_16UC1, irBuffer);
	}

	
	bool KinectAzureCaptureDevice::isOpen()
	{
		return cameraRunning;
	}

	
	int KinectAzureCaptureDevice::getRows(texpert::CaptureDeviceComponent c)
	{
		switch (c) 
		{
		case texpert::CaptureDeviceComponent::COLOR:
			return color_height;

		case texpert::CaptureDeviceComponent::DEPTH:
			return depth_height;

		}
		return -1; //bad value
	}

	
	int KinectAzureCaptureDevice::getCols(texpert::CaptureDeviceComponent c)
	{
		switch (c) {
		case texpert::CaptureDeviceComponent::COLOR:
			return color_width;

		case texpert::CaptureDeviceComponent::DEPTH:
			return depth_width;
		}
		return -1; //bad value
	}


	//virtual 
	cv::Mat& KinectAzureCaptureDevice::getCameraParam(void)
	{
		return _camera_param;
	}





