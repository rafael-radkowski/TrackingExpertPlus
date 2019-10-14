#include "SensorRealSense.h"

using namespace texpert; 

SensorRealSense::SensorRealSense()
{
	_depth_height = -1;
	_depth_width = -1;
	_color_height = -1;
	_color_width = -1;
	_fps = 30;
	_param_user_set = false;
}


SensorRealSense::~SensorRealSense()
{
}

int SensorRealSense::connect()
{	
	// Start streaming with default recommended configuration
	pipe.start();

	// Wait for the next set of frames from the camera
	auto frames = pipe.wait_for_frames();
	auto depth = frames.get_depth_frame();
	auto color = frames.get_color_frame();

	_depth_width = depth.get_width();
	_depth_height = depth.get_height();
	_color_width = color.get_width();
	_color_height = color.get_height();
	
	return 0;
}

rs2::depth_frame & SensorRealSense::getDepthStream()
{
	auto frames = pipe.wait_for_frames();
	rs2::depth_frame depth = frames.get_depth_frame();
	return depth;
}

rs2::video_frame * SensorRealSense::getColorStream()
{
	auto frames = pipe.wait_for_frames();
	rs2::video_frame color = frames.get_color_frame();
	return &color;
}

