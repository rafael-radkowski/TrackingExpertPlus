#pragma once
/*



---------------------------------------------------------
Rafael Radkowski
Iowa State University
12 Aug 2020
rafael@iastate.edu
MIT License

----------------------------------------------------------
Last edit:

*/

// GLM include files
#define GLM_FORCE_INLINE
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtx/transform.hpp>  // transformation
#include <glm/gtx/quaternion.hpp> // quaternions

// STL
#include <iostream>
#include <string>
#include <fstream>

// TrackingExpert
#include "trackingx.h"
#include "graphicsx.h"



namespace texpert {

	class TrackingMain
	{
	public:

		TrackingMain();
		~TrackingMain();

		/*
		Create the tracking instance. 
		@param camera - reference to a camera that delivers depth data. 
		@return true - if successful. 
		*/
		bool create(ICaptureDevice& camera);


		/*
		Process the current frame;
		*/
		bool process(void);


		/*
		Return the point cloud
		*/
		PointCloud& getPointCloudRef(void);

	private:

		// Instance of a PointCloudProducer
		// It creates a 3D point cloud using the given depth image. 
		texpert::PointCloudProducer*		_producer;


		// Global storage for the point cloud data. 
		PointCloud							_camera_point_cloud;

		// Point cloud sampling parameters
		SamplingParam						_sParam;

		// Point Cloud registration
		PCRegistratation*					_reg;

	};


}