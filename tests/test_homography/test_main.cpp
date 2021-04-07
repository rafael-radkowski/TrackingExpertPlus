#include "opencv2/opencv.hpp"
#include "KinectAzureCaptureDevice.h"

KinectAzureCaptureDevice camera;

int main(int argc, char** argv)
{
	camera = KinectAzureCaptureDevice();
}