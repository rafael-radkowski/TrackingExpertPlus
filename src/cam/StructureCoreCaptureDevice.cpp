/**
 * StructureCoreCaptureDevice.cpp
 * Capture device for the Structure Core
 *
 * David Wehr (dawehr@iastate.edu)
 * 2019.05.24
 */

// STL
#include <iostream>
#include <memory>
#include <thread>
#include <chrono>
#include <fstream>

// Local
#include "StructureCoreCaptureDevice.h"


afrl::StructureCoreCaptureDevice::StructureCoreCaptureDevice()
{



	settings.source = ST::CaptureSessionSourceId::StructureCore;
	settings.structureCore.sensorInitializationTimeout = sensorConnectTimeout_ms;
	settings.structureCore.depthEnabled = true;
	settings.structureCore.visibleEnabled = true;
	settings.structureCore.depthResolution = ST::StructureCoreDepthResolution::_640x480;
	settings.structureCore.visibleResolution = ST::StructureCoreVisibleResolution::_640x480;

	//ST::CaptureSessionSettings settings;
 //   settings.source = ST::CaptureSessionSourceId::StructureCore;
 //   settings.structureCore.depthEnabled = true;
 //   settings.structureCore.visibleEnabled = true;
  //   settings.structureCore.infraredEnabled = true;
      settings.structureCore.accelerometerEnabled = true;
	  settings.structureCore.gyroscopeEnabled = true;
 //   settings.structureCore.depthResolution = ST::StructureCoreDepthResolution::VGA;
 //   settings.structureCore.imuUpdateRate = ST::StructureCoreIMUUpdateRate::AccelAndGyro_200Hz;





	// Begin camera callbacks on delegate
	session.setDelegate(&delegate);
	if (!session.startMonitoring(settings)) {
		std::cerr << "Unable to start Structure Core camera" << std::endl;
		return;
	}

	// Wait until sensor ready if using real sensor
	auto start_wait_time = std::chrono::system_clock::now();
	auto end_wait_time = std::chrono::system_clock::now() + std::chrono::milliseconds(sensorConnectTimeout_ms);

	while (!delegate.sensorReady) 
	{
		std::this_thread::sleep_for(std::chrono::milliseconds(100));
		if (std::chrono::system_clock::now() > end_wait_time) {
			std::cerr << "Timeout while waiting for sensor to connect" << std::endl;
			return;
		}
	}
	
	session.startStreaming();

	//std::unique_lock<std::mutex> lock(delegate.frameMutex);

	// Wait until there are valid frames
	while (!delegate.colorValid && !delegate.depthValid) {
		std::this_thread::sleep_for(std::chrono::milliseconds(5));
	}



	// Get the minimum and maximum depth ranges
	// float minDepthMm = 0.1, maxDepthMm = 10.0;
	//settings.minMaxDepthInMmOfDepthRangeMode(
	//	settings.structureCore.depthRangeMode,  // ST::StructureCoreDepthRangeMode::Default
	//	minDepthMm,
	//	maxDepthMm 
	//);

}

afrl::StructureCoreCaptureDevice::~StructureCoreCaptureDevice() {
	session.stopStreaming();
}

void afrl::StructureCoreCaptureDevice::getRGBFrame( cv::Mat &mFrame)
{
	delegate.getLastColorFrame(mFrame);
	return;
}

void afrl::StructureCoreCaptureDevice::getDepthFrame( cv::Mat &mFrame)
{
	delegate.getLastDepthFrame(mFrame);
	return;
}


bool afrl::StructureCoreCaptureDevice::isOpen()
{
	return (delegate.depthValid && delegate.colorValid);
}

void afrl::StructureCoreCaptureDevice::SessionDelegate::captureSessionEventDidOccur(ST::CaptureSession *session, ST::CaptureSessionEventId event) {
	switch (event) {
		case ST::CaptureSessionEventId::Booting: break;
		case ST::CaptureSessionEventId::Ready:
			sensorReady = true;
			break;
		case ST::CaptureSessionEventId::Disconnected:
		case ST::CaptureSessionEventId::Error:
			printf("Capture session error\n");
			exit(1);
			break;
		case ST::CaptureSessionEventId::EndOfFile:
			// When streaming and looping, will reach end of file. No problem.
			break;
		default:
			std::cout << "Capture session event unhandled: " << ST::CaptureSessionSample::toString(event);
	}
}

void afrl::StructureCoreCaptureDevice::SessionDelegate::captureSessionDidOutputSample(ST::CaptureSession *, const ST::CaptureSessionSample& sample) {
	// Obtain lock so latestColorFrame and latestDepthFrame are not modified while being accessed elsewhere

	switch (sample.type) {
		case ST::CaptureSessionSample::Type::DepthFrame:
			 //printf("Depth frame: size %dx%d\n", sample.depthFrame.width(), sample.depthFrame.height());
			 if (sample.depthFrame.isValid()) {
				latestDepthFrame = sample.depthFrame;
				copyDepthToMat();
				depthValid = true;
			}
			break;
		case ST::CaptureSessionSample::Type::VisibleFrame:
			//printf("Visible frame: size %dx%d\n", sample.visibleFrame.width(), sample.visibleFrame.height());
			if (sample.visibleFrame.isValid()) {
				latestColorFrame = sample.visibleFrame;
				copyColorToMat();
				colorValid = true;
			}
			break;
		case ST::CaptureSessionSample::Type::SynchronizedFrames:
			//printf("Synchronized frames: depth %dx%d visible %dx%d infrared %dx%d\n", sample.depthFrame.width(), sample.depthFrame.height(), sample.visibleFrame.width(), sample.visibleFrame.height(), sample.infraredFrame.width(), sample.infraredFrame.height());

			if (sample.visibleFrame.isValid()) {
				latestColorFrame = sample.visibleFrame;
				copyColorToMat();
				colorValid = true;
			}
			if (sample.depthFrame.isValid()) {
				latestDepthFrame = sample.depthFrame;
				copyDepthToMat();
				depthValid = true;
			}
			break;
		default:
			//printf("Structure Core sample type unhandled\n");
			break;
	}
}

void afrl::StructureCoreCaptureDevice::SessionDelegate::getLastColorFrame(cv::Mat& mFrame) 
{
	mFrame = colorFrame;
}

void afrl::StructureCoreCaptureDevice::SessionDelegate::getLastDepthFrame(cv::Mat& mFrame) {
	mFrame = depthFrame;
}


void afrl::StructureCoreCaptureDevice::SessionDelegate::copyColorToMat()
{
	cv::Mat cameraFrame(latestColorFrame.height(), latestColorFrame.width(), CV_8UC3, (void*)latestColorFrame.rgbData());
	cv::cvtColor(cameraFrame, colorFrame, cv::COLOR_RGB2BGR);
}
			
void afrl::StructureCoreCaptureDevice::SessionDelegate::copyDepthToMat()
{

	cv::Mat cameraFrame(latestDepthFrame.height(), latestDepthFrame.width(), CV_32F, (void*)latestDepthFrame.depthInMillimeters());
	depthFrame = cameraFrame.clone();
}