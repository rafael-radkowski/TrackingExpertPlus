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

KinectAzureCaptureDevice* camera;
cv::Mat img, marked_img;
char keyInput;

int clicknum;
bool getting_input;

std::vector<cv::Point2f> points;

const char* winname = "Homography Test";
const char* rgb_pic = "../../data/pics/antenna_coupler_5parts_1.png";
const char* depth_pic = "../../data/pics/2.png";

void MouseCallback(int action, int x, int y, int flag, void* userInput)
{
	if (getting_input)
	{
		if (action == CV_EVENT_LBUTTONUP)
		{
			points[clicknum].x = x;
			points[clicknum].y = y;
			clicknum++;
		}
	}

	//DEBUG
	if (action == CV_EVENT_LBUTTONUP)
	{
		std::cout << x << ", " << y << std::endl;
	}
}

void transformImg(cv::Mat& input, cv::Mat& output, cv::Point2f* srcpts, cv::Point2f* dstpts)
{
	cv::Mat p_trans = cv::getPerspectiveTransform(srcpts, dstpts);
	cv::warpPerspective(input, output, p_trans, cv::Size(input.cols, input.rows));
}

int main(int argc, char** argv)
{
	camera = new KinectAzureCaptureDevice();
	keyInput = 'r';

	if (!camera->isOpen())
	{
		return -1;
	}

	//system("pause");

	points = std::vector<cv::Point2f>(8);
	getting_input = false;
	clicknum = 0;

	camera->getRGBFrame(img);
	cv::imshow(winname, img);
	cv::setMouseCallback(winname, MouseCallback);

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

	if (colImg.empty())
	{
		std::cout << "Could not find color image.  Aborting..." << std::endl;
		return -1;
	}
	if (depthImg.empty())
	{
		std::cout << "Could not find depth image.  Aborting..." << std::endl;
		return -1;
	}


	while (keyInput == 'r')
	{
		cv::Mat tempImg;
		colImg.copyTo(tempImg);
		cv::imshow(winname, colImg);

		clicknum = 0;

		getting_input = true;
		while (clicknum < 4)
		{
			for(int i = 0; i < clicknum; i++)
				cv::circle(tempImg, points[i], 4, cv::Scalar(50 * i, 50 * i, 255));

			cv::imshow(winname, tempImg);

			cv::waitKey(25);
		}
		getting_input = false;

		for (int i = 0; i < clicknum; i++)
			cv::circle(tempImg, points[i], 4, cv::Scalar(50 * i, 50 * i, 255));

		cv::imshow(winname, tempImg);

		std::cout << "Press r to redo capture, and any other key to continue" << std::endl;
		keyInput = cv::waitKey(0);
	}

	keyInput = 'r';
	while (keyInput == 'r')
	{
		cv::Mat tempImg;
		depthImg.copyTo(tempImg);
		cv::imshow(winname, depthImg);

		clicknum = 4;

		getting_input = true;
		while (clicknum < 8)
		{
			for (int i = 4; i < clicknum; i++)
				cv::circle(tempImg, points[i], 4, cv::Scalar(30, 40 * i, 255));

			cv::imshow(winname, tempImg);

			cv::waitKey(25);
		}
		getting_input = false;

		for (int i = 0; i < clicknum; i++)
			cv::circle(tempImg, points[i], 4, cv::Scalar(30, 40 * i, 255));

		cv::imshow(winname, tempImg);

		std::cout << "Press r to redo capture, and any other key to continue" << std::endl;
		keyInput = cv::waitKey(0);
	}

	cv::Point2f srcpts[4] = { points[0], points[1], points[2], points[3] };
	cv::Point2f dstpts[4] = { points[4], points[5], points[6], points[7] };

	cv::Mat p_trans = cv::getPerspectiveTransform(srcpts, dstpts);

	cv::Mat resMat;
	cv::warpPerspective(colImg, resMat, p_trans, cv::Size(std::fmax(colImg.cols, depthImg.cols), std::fmax(colImg.rows, depthImg.rows)));

	cv::imshow(winname, resMat);
	keyInput = cv::waitKey(0);

	return 0;
}