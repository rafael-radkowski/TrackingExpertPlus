/**
 * ICaptureDevice.h
 * Abstract class for capture devices
 * 
 * Tim Garrett (garrettt@iastate.edu)
 * 2019.02.06

 --------------------------------------------------------------
 Last edits:

 Oct 27, 2019, RR
 - Added a file with camera types. 

 */
#pragma once

// OpenCV
#include <opencv2/opencv.hpp>

// local
#include "ICaptureDeviceTypes.h"

namespace texpert 
{

	class ICaptureDevice
	{
	  public:
		ICaptureDevice() 
		{}

		virtual ~ICaptureDevice() {}

		/**
		 * Return the RGB frame
		 * @param mFrame - location to store the frame
		 */
		virtual void getRGBFrame( cv::Mat &mFrame) = 0;

		/**
		Return the depth frame.
		@param mFrame - location to store the frame
		*/
		virtual void getDepthFrame( cv::Mat &mFrame) = 0;
		

		/**
	     * Returns if the capture device is open
	     * @return If the capture device is open
	     */
		virtual bool isOpen() = 0;

		/*
		Return the number of image rows in pixel
		@param c - the requested camera component. 
		@return - number of image rows in pixel. -1 if the component does not exist.
		*/
		virtual int getRows(CaptureDeviceComponent c) = 0;

		/*
		Return the number of image colums in pixel
		@param c - the requested camera component. 
		@return - number of image columns in pixel. -1 if the component does not exist.
		*/
		virtual int getCols(CaptureDeviceComponent c) = 0;

	protected:

		
			
		
	};

}