/*
 Camera homography test

 A program to help test homography classes in Tracking Expert Plus

 William Blanchard
 Iowa State University
 Apr 2021
 wsb@iastate.edu
 +1 (847) 707-1421

-----------------------------------------------------------
Last edits:
Apr 9, 2021, WB
	- Initial commit
*/

#include "opencv2/opencv.hpp"
#include "KinectAzureCaptureDevice.h"
#include <Windows.h>

#include "HomographyHelper.h"

KinectAzureCaptureDevice* camera;
cv::Mat img, marked_img;
char keyInput;

int clicknum;
bool getting_input;

std::vector<cv::Point2f> points;

const char* winname = "Homography Test";
const char* rgb_pic = "../../data/pics/pc_pic_overlay.png";
const char* depth_pic = "../../data/pics/pc_pic_overlay.png";
const char* out_file = "output.txt";

int main(int argc, char** argv)
{
	//camera = new KinectAzureCaptureDevice();
	//keyInput = 'r';

	//if (!camera->isOpen())
	//{
	//	return -1;
	//}

	////system("pause");

	//points = std::vector<cv::Point2f>(8);
	//getting_input = false;
	//clicknum = 0;

	//camera->getRGBFrame(img);
	//cv::imshow(winname, img);
	//cv::setMouseCallback(winname, MouseCallback);

	//while (keyInput != 'e')
	//{
	//	keyInput = NULL;
	//	while (keyInput != 'c' && keyInput != 'C')
	//	{
	//		camera->getRGBFrame(img);
	//		cv::imshow(winname, img);
	//		keyInput = cv::waitKey(25);
	//	}



	//	getting_input = true;
	//	std::cout << "Click the points you want to use" << std::endl;

	//	clicknum = 0;

	//	while (clicknum < 8)
	//	{
	//		marked_img = img;
	//		for (int i = 0; i < clicknum; i++)
	//		{
	//			if (i / 4 == 0)
	//				cv::circle(marked_img, points[i], 4, cv::Scalar(255, 30, 255));
	//			else
	//				cv::circle(marked_img, points[i], 4, cv::Scalar(30, 255, 255));
	//		}

	//		cv::imshow(winname, marked_img);

	//		cv::waitKey(25);
	//	}

	//	

	//	cv::Point2f srcpts[4] = { points[0], points[1], points[2], points[3] };
	//	cv::Point2f dstpts[4] = { points[4], points[5], points[6], points[7] };

	//	cv::Mat p_trans = cv::getPerspectiveTransform(srcpts, dstpts);

	//	cv::Mat resMat;
	//	cv::warpPerspective(img, resMat, p_trans, cv::Size(img.cols, img.rows));

	//	cv::imshow(winname, resMat);
	//	keyInput = cv::waitKey(0);
	//}

	cv::Mat colImg = cv::imread(rgb_pic);
	cv::Mat depthImg = cv::imread(depth_pic);
	cv::Mat resMat;

	cv::resize(colImg, colImg, cv::Size(colImg.cols / 2, colImg.rows / 2));
	cv::resize(depthImg, depthImg, cv::Size(depthImg.cols / 2, depthImg.rows / 2));

	HomographyHelper::Homography22d(colImg, resMat);
	HomographyHelper::SaveHomography(resMat, out_file);

	cv::imshow(winname, resMat);
	keyInput = cv::waitKey(0);

	return 0;
}