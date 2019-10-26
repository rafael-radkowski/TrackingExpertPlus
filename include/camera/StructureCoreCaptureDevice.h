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

		int _height; //!< Image height
		int _width; //!< Image width

	};
}