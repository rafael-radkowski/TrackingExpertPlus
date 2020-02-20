#pragma once
/*

class DataReaderWriter.h

@brief







Rafael Radkowski
Iowa State University
rafael@iastate.edu
December 2019
MIT License
------------------------------------------------------------------------------------------------------
Last edits:

*/


// stl
#include <iostream>
#include <string>
#include <vector>
#include <fstream>
#include <strstream>
#include <conio.h>
// Eigen
#include <Eigen/Dense>

// opencv
#include <opencv2/opencv.hpp>


// local
#include "ReaderWriterPLY.h"
#include "ReaderWriterOBJ.h"
#include "FileUtilsX.h"

#include "Types.h"  // PointCloud data type


namespace texpert {

	class DataReaderWriter {

		public:

			/*
			Write a complete dataset to a folder. 
			@param path - string containing the path
			@param name - label for all images. 
			@param point_cloud - reference to the point cloud object. 
			@param rgb_img - cv::Mat containing the rgb image of type CV_8UC3
			@param depth_map - cv::Mat containing the depth image of type CV_16UC1
			@return true - if the image was sucessfully written.
			*/
			static bool Write(std::string path, std::string name, PointCloud& point_cloud, cv::Mat& rgb_img, cv::Mat& depth_map );

		private:

	};


}// namespace texpert