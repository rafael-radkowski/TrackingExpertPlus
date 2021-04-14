#include "HomographyHelper.h"

namespace ns_HomographyHelper
{
	bool getting_input = false;
	int clicknum = 0;
	cv::Point2d* points = (cv::Point2d*)malloc(8 * sizeof(cv::Point2d));

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
}

using namespace ns_HomographyHelper;


//static
void HomographyHelper::Homography22d(cv::Mat& imgsrc, cv::Mat& imgdst, cv::Mat& output, bool verbose)
{
	if (imgsrc.empty())
	{
		std::cout << "Could not find color image.  Aborting..." << std::endl;
		return;
	}
	if (imgdst.empty())
	{
		std::cout << "Could not find depth image.  Aborting..." << std::endl;
		return;
	}

	const char* winname = "Homography Chooser";
	getting_input = false;
	clicknum = 0;
	char keyInput = 'r';


	cv::imshow(winname, imgsrc);
	cv::setMouseCallback(winname, MouseCallback);


	while (keyInput == 'r')
	{
		cv::Mat tempImg;
		imgsrc.copyTo(tempImg);
		cv::imshow(winname, imgsrc);
		clicknum = 0;

		getting_input = true;
		while (clicknum < 4)
		{
			for (int i = 0; i < clicknum; i++)
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
		imgdst.copyTo(tempImg);
		cv::imshow(winname, imgdst);

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

	output = cv::getPerspectiveTransform(srcpts, dstpts);

	if (verbose)
	{
		cv::Mat viewMat;
		cv::warpPerspective(imgsrc, viewMat, output, cv::Size(std::fmax(imgsrc.cols, imgdst.cols), std::fmax(imgsrc.rows, imgdst.rows)));
		cv::imshow(winname, viewMat);
		cv::waitKey(0);
	}

	cv::destroyWindow(winname);
}


//static
void HomographyHelper::Homography22d(cv::Mat& input, cv::Mat& output, bool verbose)
{
	if (input.empty())
	{
		std::cout << "Could not find color image.  Aborting..." << std::endl;
		return;
	}

	const char* winname = "Homography Chooser";
	getting_input = false;
	clicknum = 0;
	char keyInput = 'r';


	cv::imshow(winname, input);
	cv::setMouseCallback(winname, MouseCallback);


	while (keyInput == 'r')
	{
		cv::Mat tempImg;
		input.copyTo(tempImg);
		cv::imshow(winname, input);
		clicknum = 0;

		getting_input = true;
		while (clicknum < 4)
		{
			for (int i = 0; i < clicknum; i++)
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
		input.copyTo(tempImg);
		cv::imshow(winname, input);

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

	output = cv::getPerspectiveTransform(srcpts, dstpts);

	if (verbose)
	{
		cv::Mat viewMat;
		cv::warpPerspective(input, viewMat, output, cv::Size(input.cols, input.rows));
		cv::imshow(winname, viewMat);
		cv::waitKey(0);
	}

	cv::destroyWindow(winname);
}


//static
void HomographyHelper::Homography22d(cv::Point2f srcpts[4], cv::Point2f dstpts[4], cv::Mat& output)
{
	output = cv::getPerspectiveTransform(srcpts, dstpts);
}


//static
void HomographyHelper::SaveHomography(cv::Mat& input, const char* filepath)
{
	ofstream file;
	file.open(filepath);

	if (!file.is_open())
	{
		std::cout << "WARNING: Could not open file " << filepath << ".\n";
		return;
	}

	char* matrix = (char*)malloc(300 * sizeof(char));
	sprintf(matrix, "%0.6f\t, %0.6f\t, %0.6f\t\n%0.6f\t, %0.6f\t, %0.6f\t\n%0.6f\t, %0.6f\t, %0.6f\t\n",
		((double*)input.data)[0], ((double*)input.data)[1], ((double*)input.data)[2],
		((double*)input.data)[3], ((double*)input.data)[4], ((double*)input.data)[5],
		((double*)input.data)[6], ((double*)input.data)[7], ((double*)input.data)[8]);

	file.close();
}


//static
void HomographyHelper::LoadHomography(cv::Mat& output, const char* filepath)
{
	ifstream file;
	file.open(filepath);

	if (!file.is_open())
	{
		std::cout << "WARNING: Could not open file " << filepath << ".\n";
		return;
	}

	char* buffer = (char*)malloc(300 * sizeof(char));
	output = cv::Mat(3, 3, CV_32F);

	for (int i = 0; i < 3; i++)
	{
		file.getline(buffer, 300);

		sscanf(buffer, "%f\t, %f\t, %f\t\n",
			&((float*)output.data)[0 + i * 3], &((float*)output.data)[1 + i * 3], &((float*)output.data)[2 + i * 3]);
	}

	file.close();
}