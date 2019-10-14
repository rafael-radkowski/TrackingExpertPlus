/*
class SensorRealSense

Created by Jiale Feng on 11/7/2018 
Copyright (c) 2018 Jiale Feng. All rights reserved.

Rafael Radkowski
Iowa State University
rafael@iastate.edu
MIT License
---------------------------------------------------------------
*/

#include "ICP_Exports.h"
#include <iostream>

// RealSense
#include <librealsense2/rs.hpp> // Include RealSense Cross Platform API

namespace texpert {

class ICPTrackingLib_EXPORTS SensorRealSense
{
private:
	// Declare pointcloud object, for calculating pointclouds and texture mappings
	rs2::pointcloud pc;
	// We want the points object to be persistent so we can display the last cloud when a frame drops
	rs2::points points;

	// Declare RealSense pipeline, encapsulating the actual device and sensors
	rs2::pipeline pipe;

	int		_depth_width;
	int		_depth_height;
	int		_color_width;
	int		_color_height;

	int		_fps;
	bool	_param_user_set;


public:
	SensorRealSense();
	~SensorRealSense();

	/*
	Connect the sensor.
	@return success = 1 if successful
	*/
	int connect();

	/*
	Get the depth frame from the sensor
	*/
	rs2::depth_frame& getDepthStream();

	/*
	Get the color frame from the sensor
	*/
	rs2::video_frame* getColorStream();

	/*
	Get the sensor resolutions in pixel.
	if -1, the sensor is not available.
	*/
	int depth_w() { return _depth_width; };
	int depth_h() { return _depth_height; };
	int color_w() { return _color_width; };
	int color_h() { return _color_height; };
};

} //texpert 