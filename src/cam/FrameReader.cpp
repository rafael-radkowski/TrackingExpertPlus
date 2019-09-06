//
//  FrameReader.cpp
//  DenseModelTracking
//
//  Created by David Wehr on 10/29/14.
//  Copyright (c) 2014 Dr. Rafael Radkowski. All rights reserved.
//
#include "stdafx.h"
#include "FrameReader.h"

DepthStreamListener::DepthStreamListener(std::function<void()> callback){
	new_frame = false;
	 focal_length_kinect=361.6;
	 focal_length_structure_sensor=288.25;
	focal_length = 361.6; //288.25
	//Sampling Method

	/* Focal length 
	- Kinect V1: 520.0
	- Kinect V2: 361.1
	- P60U: 
	*/

	_filter_kernel=3;
	// Record Frames Temp Data
	isRecordFrames = false;
	recordedFrames=new vector<vector<uint16_t>>();
	NewFrameCallBack=callback;
}

// Implementation of the frame callback handler
void DepthStreamListener::onNewFrame(openni::VideoStream& depth) {
	frame = &depth;
	if(isRecordFrames){
			openni::VideoFrameRef rf;
			depth.readFrame(&rf);
			
			const uint16_t* t = (const uint16_t*)rf.getData();
			recordedFrames->push_back(vector<uint16_t>(t, t + (rf.getWidth()*rf.getHeight())));
	}
	
    NewFrameCallBack();
}

void DepthStreamListener::set_focal_length(int index)
{
	switch(index)
	{
	case 0: focal_length = focal_length_kinect;break;// step = 8; 
	case 1: focal_length =  focal_length_structure_sensor;break;// step = 4; 
	}
}

void DepthStreamListener::set_focal_length(double fl)
{
	if(fl > 1.0)
		focal_length = fl;
}

void DepthStreamListener::readerRecordFrames(bool r)
{
	if(r){
		recordedFrames->clear();
	}
	isRecordFrames=r;
}

vector<vector<uint16_t>>* DepthStreamListener::readerGetFrames(){
	return recordedFrames;
}


