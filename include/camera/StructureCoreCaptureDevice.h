/**
 * StructureCoreCaptureDevice.h
 * Capture device for the Structure Core
 * 
 * David Wehr (dawehr@iastate.edu)
 * 2019.05.24
 */
#pragma once

// STL
#include <mutex>
#include <atomic>

// OpenCV
#include <opencv2/opencv.hpp>

// StructureCore
#include <ST/CaptureSession.h>

// local
#include "ICaptureDevice.h"
#include "CameraParameters.h"	// to read camera parameters

namespace texpert {

class StructureCoreCaptureDevice : public ICaptureDevice
{
	public:
		/**
		 * @param loadFile The OCC file to load and play. If empty, runs camera
		 */
		StructureCoreCaptureDevice();
		~StructureCoreCaptureDevice();

		/**
		 * Gets a frame
		 * @param type The type of frame
		 * @param mFrame The image
		 */
		void getRGBFrame( cv::Mat &mFrame);
		void getDepthFrame( cv::Mat &mFrame);
		
		/**
		 * Returns if the capture device is open
		 * @return If the capture device is open
		 */
		bool isOpen();


		/*
		Return the number of image rows in pixel
		@param c - the requested camera component. 
		@return - number of image rows in pixel. -1 if the component does not exist.
		*/
		int getRows(CaptureDeviceComponent c);

		/*
		Return the number of image colums in pixel
		@param c - the requested camera component. 
		@return - number of image columns in pixel. -1 if the component does not exist.
		*/
		int getCols(CaptureDeviceComponent c);


		/*!
		Read camera parameters for the depth camera from a file.
		@param path_and_file - string with a relative or absolute path pointing to the 
		camera parameters
		*/
		bool readCameraParameters(std::string path_and_file, bool verbose = false);


		/*!
		Return the intrinsic camera parameters
		@return 3x3 cv::Mat with
			[ fx 0 cx ]
			[ 0 fy cy ]
			[ 0 0  1  ]
		*/
		cv::Mat& getCameraParam(void);


		/*
		Set a callback to be invoked as soon as a frame arrives
		@param cb - function pointer for a callback. 
		*/
		void setCallbackPtr(std::function<void()> cb);

	private:
		struct SessionDelegate : ST::CaptureSessionDelegate {
			SessionDelegate() : sensorReady(false), colorValid(false), depthValid(false) {
				frame_couter = 0;
			}
			virtual ~SessionDelegate() {}

			void captureSessionEventDidOccur(ST::CaptureSession *session, ST::CaptureSessionEventId event) override;
			void captureSessionDidOutputSample(ST::CaptureSession *, const ST::CaptureSessionSample& sample) override;


			void getLastColorFrame(cv::Mat& mFrame);
			void getLastDepthFrame(cv::Mat& mFrame);

			void copyColorToMat();
			void copyDepthToMat();

			std::mutex frameMutex;
			std::mutex depthMutex;
			
			ST::ColorFrame latestColorFrame;
			ST::DepthFrame latestDepthFrame;

			cv::Mat	colorFrame;
			cv::Mat depthFrame;

			bool sensorReady;
			bool colorValid;
			bool depthValid;
			int  frame_couter; // counts the number of frames that arrive

			std::function<void()>	callback_function;
		};




		ST::CaptureSessionSettings	settings;
		ST::CaptureSession			session;
		SessionDelegate delegate; //!< Delegate that handles callbacks for device events
		
		int sensorConnectTimeout_ms = 6000; //!< Seconds to wait for sensor connecting

		int _color_height; //!< Image height
		int _color_width; //!< Image width
		int _depth_height; //!< Image height
		int _depth_width; //!< Image width
		
		// intrinsic camera parameters
		cv::Mat _intrinsic;
		cv::Mat _distortion;

	};
}