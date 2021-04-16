#pragma once
/*
@class HomographyHelper.h
@brief Static class to aid in determining homography between points

This class is meant to help find the homograpy matrix between multiple sets of points, 
either in 3D or 2D space.  

Functions named Homography22d find the homography between 2D points.  They are either 
passed images, in which case they will initiate a process where the user chooses points 
from the images to use for the homography calculation, or they are passed two arrays of 
points, in which case they will simply find and return the homography between those arrays.

In addition, this class holds functions meant to read and write matrices to and from .json
files.  They use OpenCV's FileStorage class to perform these actions.

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
Apr 16, 2021, WB
	- Fixed Save/Load homography functions to use .json files instead of any text file
	- Added verbose option to Homography22d process functions
	- Added documentation
*/

//opencv
#include "opencv2/opencv.hpp"

//local
#include "FileUtilsExt.h"

class HomographyHelper
{

public:
	/*
		Prompt the user to choose 4 points each from imgsrc and imgdst and output the homography 
		from the points chosen from imgsrc to imgdst

		@param imgsrc - the image with the original reference points
		@param imgdst - the image with the destination reference points
		@param output - the matrix to put the homography into
		@param verbose - show imgsrc transformed by the resulting homography matrix if true
	*/
	static void Homography22d(cv::Mat& imgsrc, cv::Mat& imgdst, cv::Mat& output, bool verbose = false);

	/*
		Prompt the user to choose 8 points from image input and output the homography
		from the first 4 chosen points to the last 4 chosen points

		@param input - the image with the reference points
		@param output - the matrix to put the homography into
		@param verbose - show imgsrc transformed by the resulting homography matrix if true
	*/
	static void Homography22d(cv::Mat& input, cv::Mat& output, bool verbose = false);

	/*
		Find the homography between the 2D points stored in srcpts and those stored in dstpts

		@param srcpts - the original points as an array of 4 cv::Point2f
		@param dstpts - the destination points as an array of 4 cv::Point2f
		@param output - the matrix to put the homography into
	*/
	static void Homography22d(cv::Point2f srcpts[4], cv::Point2f dstpts[4], cv::Mat& output);

	static void Homography23d(std::vector<cv::Point2f> imgpts, std::vector<cv::Point3f> modelpts, cv::Mat& output, bool verbose = false);

	/*
		Save a homography matrix into a .json file specified by filepath

		@param input - the homography matrix to save
		@param filepath - the .json file the matrix will be stored in
	*/
	static void SaveHomography(cv::Mat& input, std::string filepath);

	/*
		Load a homography matrix from a .json file specified by filepath

		@param output - the homography matrix to store the data in
		@param filepath - the .json file the matrix will be parsed from
	*/
	static void LoadHomography(cv::Mat& output, std::string filepath);
};