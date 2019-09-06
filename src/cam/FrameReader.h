//
//  FrameReader.h
//  DenseModelTracking
//
//  Created by David Wehr on 10/29/14.
//  Copyright (c) 2014 Dr. Rafael Radkowski. All rights reserved.
//

#ifndef __DenseModelTracking__FrameReader__
#define __DenseModelTracking__FrameReader__

#pragma once

#include <iostream>
#include <thread>
#include "Sensor.h"
#include "PointCloud.h"
#include <osg/Vec3>

using namespace osg;
using namespace std;
using namespace mycv;

class ICPTrackingLib_EXPORTS DepthStreamListener : public openni::VideoStream::NewFrameListener {
private:
	//Sensor data
    int frame_width;
	int frame_height;
    openni::VideoStream* frame;
    bool new_frame;
	double focal_length_kinect;
	double focal_length_structure_sensor;
	double focal_length; //288.25
	int _filter_kernel; 
	vector<vector<uint16_t>>* recordedFrames;
	// Record Frames Temp Data
	bool isRecordFrames;
	std::function<void()> NewFrameCallBack;
	
	vector<dPoint> getConnectedNeighbors(const uint16_t* imgBuf,int x, int y,int w, int h);
public:


	DepthStreamListener(std::function<void()> callback);
	
	openni::VideoStream* get_New_Frame_Depth_Data(){
		return frame;
	}
	
	/*!
	Set the focal length of the image
	@param focal_length = device 0 = kinect, 1 = strcutre sensor
	*/
    void set_focal_length(int index = 1);

	/*!
	Set the focal length of the image
	@param focal_length = the focal length of the image
	*/
	void set_focal_length(double fl);
    
	/*!
	Virtual function for OpenNI
	*/
    virtual void onNewFrame(openni::VideoStream&);

	/*!
	Tells the reader to record frames
	@param r = true/false record frames
	*/
	void readerRecordFrames(bool r);

	/*!
	Get the depth data
	@return imgBuf = the depth data
	*/
	vector<vector<uint16_t>>* readerGetFrames();
};

#endif /* defined(__DenseModelTracking__FrameReader__) */
