#pragma once
/*
class PointCloudProducer

Input:
- Camera of class ICaptureDevice. Note that the depth frame must be provided as CV_32FC1 type.

@brief: 


Features:


Rafael Radkowski
Iowa State University
Oct 2019
rafael@iastate.edu
+! (515) 294 7044

-----------------------------------------------------------
Last edits:

Feb 7, 2020, RR
- Added a camera parameter loader.

Feb 20, 2020, RR
- Added a point cloud storage as a member
- Added a function that copies the point from the internal to the external storage. 
	and removes all 0,0,0 points. 

Aug 8, 2020, RR
- Added a function to set the point cloud filter method. 

Aug 9, 2020, RR
- BUGFIX: fixed a severe bug in setSampingMode(). The sampling pattern used the color image resolution
  to create sampling patterns and not the depth resolution. That resulted in double-points in the point cloud. 

Aug 27, 2020, RR
- Removed a copy_if operator and added a loop to copy points. Copy_if return incorrect sized vectors. 
Aug 4, 2021
- Fixed a bug when reading camera parameters. The class read the incorrect principle point y-displacement from the camera. 

*/
#include <iostream>
#include <vector>
#include <algorithm>
// opencv
#include <opencv2/opencv.hpp>

// Eigen
#include <Eigen/Dense>

// local
#include "ICaptureDevice.h"
#include "ICaptureDeviceTypes.h"
#include "Types.h"  // PointCloud data type
#include "SamplingTypes.h"
#include "FilterTypes.h"

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


	/*!
	Set a point cloud filter methods. 
	@param method - can be NONE or BILATERAL
	@param param - the parameters for the filter
	*/
	void setFilterMethod(FilterMethod method, FilterParams param);

	/*
	For normal vector calculation - set the related parameters
	@param step_size - the size of the normal vector search window in pixel. 
		The function looks for depth values in adjacent pixels up to a distance step_size. 
	*/
	void setNormalVectorParams(int step_size);


	/*
	Flip the normal vectors. They are not flipped by default.
	But some cameras require them to be inverted. 
	@param flip - true flips the normal vectors. 
	*/
	void setFlipNormalVectors(bool flip = false);


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

	// copy points from the internal storage to the external
	// and remove all points with values (0,0,0)
	bool copy_and_clear_points(void);

	//----------------------------------------------------------------------------------

	// the camera capture device. 
	ICaptureDevice&			_capture_device;

	int						_depth_rows;
	int						_depth_cols;

	// the depth camera focal length
	// the default values are set for the structure core. 
	float					_fx_depth;
	float					_fy_depth;

	// caemra principal points.
	float					_cx_depth;
	float					_cy_depth;

	// location of an external storage to write the camera
	// point cloud data to. 
	PointCloud&				_the_cloud;

	// Internal location for the point cloud.
	// The external cloud in _the_cloud only contains valid values
	// with points no 0,0,0. The return from cuda can still contain (0,0,0) points. 
	PointCloud				_pc_storage;

	// Parameters to inidcate the sampling method
	SamplingMethod			_sampling_method;
	SamplingParam			_sampling_param;

	// step size for normal vector estimation. 
	// The normal vectors are estimated using cross-products
	// to adjacent pixels. The variable indicates the distance to the neighbor. 
	int						_normal_vector_step_size;

	float					_flip_normal_vectors;

	// true, if all initialization steps were completed sucessfull.y
	bool					_producer_ready;
};


}//namespace texpert 