#include "HomographyHelper.h"

namespace ns_HomographyHelper
{
	bool getting_input = false;
	int clicknum = 0;
	cv::Point2d* points = (cv::Point2d*)malloc(8 * sizeof(cv::Point2d));

	//Namespace OpenCV mouse callback
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
	}
}

using namespace ns_HomographyHelper;


//static
void HomographyHelper::Homography22dProcess(cv::Mat& imgsrc, cv::Mat& imgdst, cv::Mat& output, bool verbose)
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

	//Initialize loop variables
	const char* winname = "Homography Chooser";
	getting_input = false;
	clicknum = 0;
	char keyInput = 'r';

	//Start window, set mouse callback
	cv::Mat merged;
	cv::addWeighted(imgsrc, 1.0, imgdst, 1.0, 0.0, merged);

	cv::imshow(winname, merged);
	cv::setMouseCallback(winname, MouseCallback);

	//Select points on the source image
	while (keyInput == 'r')
	{
		cv::Mat tempImg;
		merged.copyTo(tempImg);
		cv::imshow(winname, merged);
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

	//Select points on the destination image
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

	//Put points on arrays
	cv::Point2f srcpts[4] = { points[0], points[1], points[2], points[3] };
	cv::Point2f dstpts[4] = { points[4], points[5], points[6], points[7] };

	//Set output matrix to calculated homography
	output = cv::getPerspectiveTransform(srcpts, dstpts);

	//Show original image transformed by the homography matrix
	if (verbose)
	{
		cv::Mat viewMat;
		cv::Mat mergedRes;
		cv::warpPerspective(imgsrc, viewMat, output, cv::Size(std::fmax(imgsrc.cols, imgdst.cols), std::fmax(imgsrc.rows, imgdst.rows)));
		cv::addWeighted(viewMat, 1.0, imgdst, 1.0, 0.0, mergedRes);
		cv::imshow(winname, mergedRes);
		cv::imshow("Reference", merged);
		cv::waitKey(0);
		cv::destroyWindow("Reference");
	}

	cv::destroyWindow(winname);
}


//static
void HomographyHelper::Homography22dProcess(cv::Mat& input, cv::Mat& output, bool verbose)
{
	if (input.empty())
	{
		std::cout << "Could not find color image.  Aborting..." << std::endl;
		return;
	}

	//Initialize loop variables
	const char* winname = "Homography Chooser";
	getting_input = false;
	clicknum = 0;
	char keyInput = 'r';

	//Start window, set mouse callback
	cv::imshow(winname, input);
	cv::setMouseCallback(winname, MouseCallback);

	//Select source points on the image
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

	//Select destination points on the image
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

	//Put points on arrays
	cv::Point2f srcpts[4] = { points[0], points[1], points[2], points[3] };
	cv::Point2f dstpts[4] = { points[4], points[5], points[6], points[7] };

	//Set output matrix to calculated homography
	output = cv::getPerspectiveTransform(srcpts, dstpts);

	//Show original image transformed by the homography matrix
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
void HomographyHelper::Homography22d(std::vector<cv::Point2f> srcpts, std::vector<cv::Point2f> dstpts, cv::Mat& output)
{
	output = cv::getPerspectiveTransform(srcpts, dstpts);
}

//static
void HomographyHelper::SaveHomography(cv::Mat& input, std::string filepath)
{
	// determine the file type
	int idx = filepath.find_last_of(".");
	std::string sub = filepath.substr(idx + 1, 3);
	std::transform(sub.begin(), sub.end(), sub.begin(), ::tolower);

	if (!(sub.compare("jso") == 0))
	{
		std::cerr << "[HomographyHelper] ERROR - cannot load " << filepath << ". Wrong file format. json is required." << std::endl;
		return;
	}

	//Open the file and write the homography matrix to it
	cv::FileStorage file = cv::FileStorage(filepath, cv::FileStorage::WRITE);
	file.write("matrix", input);
	file.release();
}


//static
void HomographyHelper::LoadHomography(cv::Mat& output, std::string filepath)
{
	//Find the file
	if (!FileUtils::Exists(filepath))
	{
		std::cerr << "[HomographyHelper] ERROR - cannot find file " << filepath << "." << std::endl;
		return;
	}

	//Make sure the file is a .json file
	int idx = filepath.find_last_of(".");
	std::string sub = filepath.substr(idx + 1, 3);
	std::transform(sub.begin(), sub.end(), sub.begin(), ::tolower);
	if (!(sub.compare("jso") == 0))
	{
		std::cerr << "[HomographyHelper] ERROR - cannot load " << filepath << ". Wrong file format. json is required." << std::endl;
		return;
	}

	//Open the homography file
	cv::FileStorage file = cv::FileStorage(filepath, cv::FileStorage::READ);
	
	//Check if a matrix exists in the file
	if (file["matrix"].isNone())
	{
		std::cerr << "[HomographyHelper] ERROR - cannot find matrix in file " << filepath << "." << std::endl;
		return;
	}

	char* buffer = (char*)malloc(300 * sizeof(char));
	output = file["matrix"].mat();
}