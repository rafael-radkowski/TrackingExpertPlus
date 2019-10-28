#pragma once
/*
class PointCloudProducer

@brief: 


Features:


Rafael Radkowski
Iowa State University
Oct 2019
rafael@iastate.edu
+! (515) 294 7044

-----------------------------------------------------------
Last edits:

*/
#include <iostream>
#include <vector>

// opencv
#include <opencv2/opencv.hpp>

// Eigen
#include <Eigen/Dense>

// local
#include "ICaptureDevice.h"
#include "ICaptureDeviceTypes.h"
#include "Types.h"  // PointCloud data type
#include "SamplingTypes.h"


namespace texpert {

class PointCloudProducer {

public:
	/*!
	*/
	PointCloudProducer(ICaptureDevice& capture_device, PointCloud& the_cloud);
	~PointCloudProducer();

	/*!
	Set the sampling method 
	@param method - set the sampling method of type SamplingMethod. 
		Can me RAW, UNIFORM, or RANDOM
	@param param - set the sampling parameters associated to the 
		sampling method. 
	*/
	void setSampingMode(SamplingMethod method, SamplingParam param);

	/*
	For normal vector calculation - set the related parameters
	@param step_size - the size of the normal vector search window in pixel. 
		The function looks for depth values in adjacent pixels up to a distance step_size. 
	*/
	void setNormalVectorParams(int step_size);

	/*!
	Process the current camera frame
	@return true, if successful, otherwise false. 
	*/
	bool process(void);


	

private:
	// raw point cloud sampling
	bool run_sampling_raw(float* imgBuf);

	// uniform point cloud sampling
	bool run_sampling_uniform(float* imgBuf);

	// random point cloud sampling
	bool run_sampling_random(float* imgBuf);

	//----------------------------------------------------------------------------------

	// the camera capture device. 
	ICaptureDevice&			_capture_device;

	int						_depth_rows;
	int						_depth_cols;

	// location of an external storage to write the camera
	// point cloud data to. 
	PointCloud&				_the_cloud;


	SamplingMethod			_sampling_method;
	SamplingParam			_sampling_param;

	int						_normal_vector_step_size;

	bool					_producer_ready;
};


}//namespace texpert 