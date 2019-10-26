/**
 * ICaptureDevice.h
 * Abstract class for capture devices
 * 
 * Tim Garrett (garrettt@iastate.edu)
 * 2019.02.06
 */
#pragma once

// OpenCV
#include <opencv2/opencv.hpp>



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

	protected:

		
			
		
	};

}