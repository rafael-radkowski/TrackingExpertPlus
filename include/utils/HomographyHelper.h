#pragma once
/*
@class HomographyHelper.h
@brief Static class to aid in determining homography between points

Features:
- Finds the homography between a set of 2D-2D correspondences
- Finds the homography between a set of 2D-3D correspondences
- Saves and loads homography to/from a file

William Blanchard
Iowa State University
Apr 2021
wsb@iastate.edu
MIT License
---------------------------------------------------------------
Last edits:
Apr 14, 2021, WB
	- Added Save/Load homography functions
	- Added Homography22d function where source and destination points are already found
*/

//opencv
#include "opencv2/opencv.hpp"

//local
#include "FileUtilsExt.h"

class HomographyHelper
{

public:
	static void Homography22d(cv::Mat& imgsrc, cv::Mat& imgdst, cv::Mat& output, bool verbose = false);

	static void Homography22d(cv::Mat& input, cv::Mat& output, bool verbose = false);

	static void Homography22d(cv::Point2f srcpts[4], cv::Point2f dstpts[4], cv::Mat& output);

	static void SaveHomography(cv::Mat& input, const char* filepath);

	static void LoadHomography(cv::Mat& output, const char* filepath);
};