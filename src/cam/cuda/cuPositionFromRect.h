#pragma once
/*************************************************************************************
The class extract the 3D position of an object from the region or interest in an image.

Input is an image with [x, y, z] points stored in an image grid. The PointCloudProducer
creates an image such as this and kept it in the gpu as float3* g_cu_point_output_dev, stored
by cuDeviceManager.

Allocate the memory first using the device manger and run the point cloud producer 
to obtain the required image. 

It returns the positon as a float3 vector

Features:
-  Extracts the position of points in a given region of interest. 

Rafael Radkowski
Iowa State University
rafael@iastate.edu
May 5th, 2018
All rights reserved.

**************************************************************************************/


// stl
#include <stdio.h>
#include <iostream>
#include <vector>
#include <algorithm>

// cuda
#include "cuda_runtime.h"

// OpenCV
#include <opencv2\highgui.hpp>
#include <opencv2\core.hpp>

#include <osg/Node>
#include <osg/Matrix>
#include <osg/Vec3>

// local
#include "cuDeviceMemory.h"


using namespace std;


namespace tacuda
{



	class cuPositionFromRect
	{
	public:

		/*
		Init the class
		*/
		static void Init(int width, int height);



		/*
		Extract the position of a set of points within a given region of interest. The position
		is the mean of all valid points.
		@param rect_vector - a vector with opencv rectangles. Each rectangle gives a region of interest
		@param position - a reference to a vector to store the positions.
		@return - the number of extracted positions
		*/
		static int GetPosition(vector<cv::Rect> rect_vector, vector<float3>& position, vector<float3>& orientation);





	private:

		/*
		Extract the position of one region of interest
		@param - the rectangle given the region of interest.
		@param - location to store the position
		@return - 1 if the location was successfully extracted
		*/
		static int ExtractPositionAndOrientation(cv::Rect rect, float3& position, float3& orientation, bool with_orient);

	};

} //namespace tacuda


