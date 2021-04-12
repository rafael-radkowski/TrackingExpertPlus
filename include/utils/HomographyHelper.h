#pragma once
/*
@class HomographyHelper.h
@brief Static class to aid in determining homography between points

Features:
- Finds and optionally saves the homography between a set of 2D-2D correspondences
- Finds and optionally saves the homography between a set of 2D-3D correspondences

William Blanchard
Iowa State University
Apr 2021
wsb@iastate.edu
MIT License
---------------------------------------------------------------
Last edits:
Apr 12, 2021, WB
	- Initial commit
*/

//opencv
#include "opencv2/opencv.hpp"

class HomographyHelper
{
public:
	static void Homography22d(cv::Mat& imgsrc, cv::Mat& imgdst, cv::Mat& output, bool save = false, char* filepath = "../../data");

	static void Homography22d(cv::Mat& input, cv::Mat& output, bool save = false, char* filepath = "../../data");
};