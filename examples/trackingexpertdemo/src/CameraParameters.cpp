#include "CameraParameters.h"

namespace CameraParameters_ns {

	bool parameters_ready = false; // indicate that the parameters are loaded from a file. 

	cv::Mat intrinsic_matrix = cv::Mat::zeros(3,3, CV_32FC1);
	cv::Mat distorion_params = cv::Mat::zeros(5,1, CV_32FC1);

	int		img_width = 640;
	int		img_height = 480;

	cv::Mat intrinsic_d_matrix = cv::Mat::zeros(4,4, CV_64FC1);
	cv::Mat distorion_d_params = cv::Mat::zeros(1,5, CV_64FC1);

	int		depth_width = 640;
	int		depth_height = 480;
}

using namespace CameraParameters_ns;


//static 
bool CameraParameters::Read(string path_and_file, bool verbose)
{
	//CameraParameters::Write(path_and_file);

	if (!FileUtils::Exists(path_and_file)) {
		cout << "[ERROR] - Cannot find file " << path_and_file << "." << endl;
		return false;
	}


	cv::FileStorage fs(path_and_file, cv::FileStorage::READ);

	if(!fs.isOpened()) {
		cout << "[ERROR] - Cannot open file " << path_and_file << "." << endl;
		return false;
	}

	cv::Size2i img;

	fs["intrinsic"] >> intrinsic_matrix;
	fs["dist"] >> distorion_params;
	fs["imgSize"] >> img;
	img_width = img.width;
	img_height = img.height;

	fs.release();

	if (verbose) {
		cout << "Intrinsic\n" << intrinsic_matrix << endl;
		cout << "Distortion\n" << distorion_params << endl;
		cout << "Image size\n" << img_width << " : " << img_height << endl;
	}

	//getIntrinsicAsFoV();

	return true;
}


//static 
bool CameraParameters::Write(string path_and_file)
{
	cv::FileStorage fs(path_and_file, cv::FileStorage::WRITE);

	if(!fs.isOpened()) {
		cout << "[ERROR] - Cannot open file " << path_and_file << "." << endl;
		return false;
	}
	
	fs << "intrinsic"  << intrinsic_matrix;
	fs << "dist" << distorion_params;
	fs << "imgSize" << cv::Size2i(img_width, img_height);

	fs.release();

	return true;
}


//static 
cv::Mat CameraParameters::getIntrinsic(void)
{
	return intrinsic_matrix;
}



//static 
cv::Mat CameraParameters::getDistortions(void)
{
	return distorion_params;
}



//static 
cv::Size CameraParameters::getImgSize(void)
{
	return cv::Size2i(img_width, img_height);
}



//static 
cv::Size2f CameraParameters::getIntrinsicAsFoV(void)
{
	if (!parameters_ready) {
		cout << "[ERROR] - Cannot calculate FoV. Load camera parameters first. " << endl;
		return cv::Size2i(0, 0);
	}

	double fovx;
	double fovy;
	double focalLength;
	cv::Point2d principalPoint;
	double aspectRatio;

	cv::calibrationMatrixValues(intrinsic_matrix, cv::Size2i(img_width, img_height), 0.0, 0.0, fovx, fovy, focalLength, principalPoint, aspectRatio);

	return cv::Size2f((float)fovx, (float)fovy);
}

