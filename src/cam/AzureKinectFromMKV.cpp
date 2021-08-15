#include "AzureKinectFromMKV.h"

using namespace texpert;


/*
Constructor.
Note that the constructor does not automatically opens and loads the file.
Call connectToDevice to get the file opened.
@param path_and_file - string with the path and filename of a video file. Needs to be an MKV file at this moment.
*/
AzureKinectFromMKV::AzureKinectFromMKV(std::string path_and_file):
	_path_and_file(path_and_file)
{
	_color_width = 0;
	_color_height = 0;
	_depth_width = 0; 
	_depth_height = 0;

	_cv_camera_matrix_depth = cv::Mat::zeros(3, 3, CV_64F); //float 3x3 matrix, meant to be camera matrix. Should be		fx 0 Cx
																															//	0 fy cy
																															//	0  0  1
	_distortion_coefficients_depth = cv::Mat::zeros(1, 8, CV_64F); // float 1x8 matrix. k1, k2, p1, p2, k3, k4, k5, k6


	_playback = (k4a_playback_t*)malloc(8);
	_capture_handle = (k4a_capture_t*)malloc(8);
	_is_open = false;
	_serial = (char*)malloc(50 * sizeof(char));

	_camera_param = cv::Mat::eye(3, 3, CV_32F);


	connectToDevice();

}

AzureKinectFromMKV::~AzureKinectFromMKV()
{
	if (_is_open) {
		_is_open = false;
		k4a_playback_close(*_playback);
	}

	free(_playback);
	free(_capture_handle);
	free(_serial);
}


/*
Open a given mkv video file.
*/
bool AzureKinectFromMKV::connectToDevice(void)
{

	k4a_result_t result = k4a_playback_open(_path_and_file.c_str(), _playback);
	
	if (result == K4A_RESULT_SUCCEEDED){
		_is_open = true;
	}else
	{
		std::cout << "[ERROR] AzureKinectFromMKV - cannot open file " << _path_and_file << "." << std::endl;
		system("pause");
		return false;
	}

	k4a_playback_set_color_conversion(*_playback, K4A_IMAGE_FORMAT_COLOR_BGRA32);
	updateDimensions();

	updateCameraParams();

	//get first image, reduces edge cases
	k4a_playback_get_next_capture(*_playback, _capture_handle);

	return true;
}


/**
* Return the RGB frame
* @param mFrame - location to store the frame
*/
//virtual 
void AzureKinectFromMKV::getRGBFrame(cv::Mat& mFrame)
{
	k4a_image_t image = k4a_capture_get_color_image(*_capture_handle);
	uint8_t* rgbBuffer = k4a_image_get_buffer(image);
	mFrame = cv::Mat(_color_height, _color_width, CV_8UC4, rgbBuffer);

	k4a_image_release(image);
}

/**
Return the depth frame.
@param mFrame - location to store the frame
*/
//virtual 
void AzureKinectFromMKV::getDepthFrame(cv::Mat& mFrame)
{
	k4a_stream_result_t result = K4A_STREAM_RESULT_SUCCEEDED;
	result = k4a_playback_get_next_capture(*_playback, _capture_handle);

	if (result == K4A_STREAM_RESULT_SUCCEEDED)
	{
		//// Process capture here
		//k4a_capture_release(capture);
	}
	else if (result == K4A_STREAM_RESULT_EOF)
	{
		if (k4a_playback_seek_timestamp(*_playback, 0, K4A_PLAYBACK_SEEK_BEGIN) != K4A_RESULT_SUCCEEDED)
		{
			std::cout << "[ERROR] - AzureKinectFromMKV: cannot jump to the start of the file." << std::endl;
		}
		result = k4a_playback_get_next_capture(*_playback, _capture_handle);
	}

	k4a_image_t image = k4a_capture_get_depth_image(*_capture_handle);
	uint8_t* depthBuffer = k4a_image_get_buffer(image);
	cv::Mat img = cv::Mat(_depth_height, _depth_width, CV_16UC1, depthBuffer);
	img.convertTo(mFrame, CV_32FC1);



	//cv::Mat distorted = cv::Mat(depth_height, depth_width, CV_16UC1, depthBuffer);
	//undistort
	//cv::undistort(distorted, mFrame, cv_camera_matrix_depth, distortion_coefficients_depth);
	k4a_image_release(image);

}


/**
 * Returns if the capture device is open
 * @return If the capture device is open
 */
//virtual 
bool AzureKinectFromMKV::isOpen()
{
	return _is_open;
}

/*
Return the number of image rows in pixel
@param c - the requested camera component.
@return - number of image rows in pixel. -1 if the component does not exist.
*/
//virtual 
int AzureKinectFromMKV::getRows(CaptureDeviceComponent c)
{
	switch (c) {
	case texpert::CaptureDeviceComponent::COLOR:
		return _color_height;

	case texpert::CaptureDeviceComponent::DEPTH:
		return _depth_height;

	}
	return -1;
}

/*
Return the number of image colums in pixel
@param c - the requested camera component.
@return - number of image columns in pixel. -1 if the component does not exist.
*/
//virtual 
int AzureKinectFromMKV::getCols(CaptureDeviceComponent c)
{
	switch (c) {
	case texpert::CaptureDeviceComponent::COLOR:
		return _color_width;

	case texpert::CaptureDeviceComponent::DEPTH:
		return _depth_width;

	}
	return -1;
}



/**< Calibration model is Brown Conrady (compatible with
														  * OpenCV) */
std::vector<float> AzureKinectFromMKV::getCalibration(texpert::CaptureDeviceComponent component)
{
	k4a_calibration_t calib;
	k4a_result_t result = k4a_playback_get_calibration(*_playback, &calib);
	if (result == K4A_RESULT_FAILED)
	{
		return std::vector<float>(1, 0.0);
	}


	_k4a_calibration_camera_t cameraCalib;
	switch (component)
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



void AzureKinectFromMKV::updateCameraParams(void)
{
	std::vector<float> calib = getCalibration(texpert::DEPTH);

	//Creating the camera matrix with the parameters of the camera

	_cv_camera_matrix_depth.at<double>(0, 0) = calib[2]; //fx
	_cv_camera_matrix_depth.at<double>(1, 1) = calib[3]; //fy
	_cv_camera_matrix_depth.at<double>(0, 2) = calib[0]; //cx
	_cv_camera_matrix_depth.at<double>(1, 2) = calib[1]; //cy
	_cv_camera_matrix_depth.at<double>(2, 2) = 1.0;

	//filling up of an array with the generated distortion parameters from camera

	_distortion_coefficients_depth.at<double>(0, 0) = calib[4];
	_distortion_coefficients_depth.at<double>(0, 1) = calib[5];
	_distortion_coefficients_depth.at<double>(0, 2) = calib[13];
	_distortion_coefficients_depth.at<double>(0, 3) = calib[12];
	_distortion_coefficients_depth.at<double>(0, 4) = calib[6];
	_distortion_coefficients_depth.at<double>(0, 5) = calib[7];
	_distortion_coefficients_depth.at<double>(0, 6) = calib[8];
	_distortion_coefficients_depth.at<double>(0, 7) = calib[9];


	

	_camera_param.at<float>(0, 0) = calib[2]; //fx
	_camera_param.at<float>(1, 1) = calib[3]; //fy
	_camera_param.at<float>(0, 2) = calib[0]; //cx
	_camera_param.at<float>(1, 2) = calib[1]; //cy
	_camera_param.at<float>(2, 2) = 1.0f; 
	
}



float AzureKinectFromMKV::getMetricRadius(texpert::CaptureDeviceComponent component)
{
	k4a_calibration_t calib;
	k4a_result_t result = k4a_playback_get_calibration(*_playback, &calib);

	if (result == K4A_RESULT_FAILED) 
	return -1.0;

	_k4a_calibration_camera_t cameraCalib;

	switch (component)
	{
	case texpert::DEPTH:
		cameraCalib = calib.depth_camera_calibration;
		break;
	case texpert::COLOR:
		cameraCalib = calib.color_camera_calibration;
		break;
	}

	return cameraCalib.metric_radius;
}

void AzureKinectFromMKV::updateDimensions(void)
{

	char buffer[50] = "";
	size_t size = 50;

	k4a_buffer_result_t err;


	k4a_record_configuration_t config;
	k4a_result_t handle =  k4a_playback_get_record_configuration(*_playback, &config);


	getColorRes(config.color_resolution, _color_height, _color_width);
	getDepthRes(config.depth_mode, _depth_height, _depth_width);


	//for each case, get tag and put in buffer. cast to variable. reset size

	/*
	err = k4a_playback_get_tag(*_playback, "K4A_COLOR_WIDTH", buffer, &size);
	sscanf(buffer, "%d", &_color_width);
	size = 50;

	err = k4a_playback_get_tag(*_playback, "COLOR_HEIGHT", buffer, &size);
	sscanf(buffer, "%d", &_color_height);
	size = 50;

	err = k4a_playback_get_tag(*_playback, "DEPTH_WIDTH", buffer, &size);
	sscanf(buffer, "%d", &_depth_width);
	size = 50;

	err = k4a_playback_get_tag(*_playback, "DEPTH_HEIGHT", buffer, &size);
	sscanf(buffer, "%d", &_depth_height);
	size = 50;

	*/

	err = k4a_playback_get_tag(*_playback, "K4A_DEVICE_SERIAL_NUMBER", buffer, &size);
	strcpy(_serial, buffer);

	

}





/*!
Return the intrinsic camera parameters
@return 3x3 cv::Mat with
	[ fx 0 cx ]
	[ 0 fy cy ]
	[ 0 0  1  ]
*/
//virtual 
cv::Mat& AzureKinectFromMKV::getCameraParam(void)
{
	return _camera_param;
}


void AzureKinectFromMKV::getColorRes(k4a_color_resolution_t mode, int& rows, int& cols)
{
	switch (mode) {

		case K4A_COLOR_RESOLUTION_OFF:
			rows = 0;
			cols = 0;
			break;

		case K4A_COLOR_RESOLUTION_720P:
			cols = 1280;
			rows = 720;
			break;

		case K4A_COLOR_RESOLUTION_1080P:
			cols = 1920;
			rows = 1080;
			break;
		case K4A_COLOR_RESOLUTION_1440P:
			cols = 2560;
			rows = 1440;
			break;

		case K4A_COLOR_RESOLUTION_1536P:
			cols = 2048;
			rows = 1536;
			break;

		case K4A_COLOR_RESOLUTION_2160P:
			cols = 3840;
			rows = 2160;

		case K4A_COLOR_RESOLUTION_3072P:
			cols = 4096;
			rows = 3072;
		default:
			rows = 0;
			cols = 0;
			break;
	}

}

void AzureKinectFromMKV::getDepthRes(k4a_depth_mode_t mode, int& rows, int& cols)
{

	//https://microsoft.github.io/Azure-Kinect-Sensor-SDK/master/group___enumerations_ga3507ee60c1ffe1909096e2080dd2a05d.html#ga3507ee60c1ffe1909096e2080dd2a05d
	switch (mode) {
		case K4A_DEPTH_MODE_OFF:
			rows = 0;
			cols = 0;
			break;

		case K4A_DEPTH_MODE_NFOV_2X2BINNED:
			rows = 288;
			cols = 320;
			break;
			//Passive IR is also captured at 320x288.

		case K4A_DEPTH_MODE_NFOV_UNBINNED:
			rows = 576;
			cols = 640;
			break;
			//Passive IR is also captured at 640x576.

		case K4A_DEPTH_MODE_WFOV_2X2BINNED:
			rows = 512;
			cols = 512;
			break;
			
			//Passive IR is also captured at 512x512.

		case K4A_DEPTH_MODE_WFOV_UNBINNED:
			rows = 1024;
			cols = 1024;
			break;

			//Passive IR is also captured at 1024x1024.

		case K4A_DEPTH_MODE_PASSIVE_IR:
			rows = 1024;
			cols = 1024;
			break;
			//Passive IR only, captured at 1024x1024.
	}

}


