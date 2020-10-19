#include "PointCloudProducer.h"


// cuda bindings
#include "cuda/cuPCU3f.h"  // point cloud samping


using namespace texpert;
using namespace std;


PointCloudProducer::PointCloudProducer(ICaptureDevice& capture_device, PointCloud& the_cloud):
	_capture_device(capture_device), _the_cloud(the_cloud)
{
	// defaults
	_sampling_method = SamplingMethod::UNIFORM;
	_sampling_param.uniform_step = 8;
	_normal_vector_step_size = 4;
	_depth_cols = -1;
	_depth_rows = -1;

	cv::Mat param = _capture_device.getCameraParam();

	_fx_depth = param.at<float>(0,0);
	_fy_depth = param.at<float>(1,1);

	_cx_depth = 0.0;
	_cy_depth = 0.0;

	_flip_normal_vectors = 1.0; // This value can either be 1.0 or -1.0;


//	if (_capture_device.isOpen()) {
//		cout << "[ERROR] - PointCloudProducer: error when opening the camera." << endl;
//		return;
//	}

	// fetch the resolution. 
	_depth_cols = _capture_device.getCols(DEPTH);
	_depth_rows = _capture_device.getRows(DEPTH);

	// reserve memory
	_pc_storage.points.reserve(_depth_rows * _depth_cols);
	_pc_storage.normals.reserve(_depth_rows * _depth_cols);


	// Allocate device memory for point cloud processing
	cuPCU3f::AllocateDeviceMemory(_depth_cols, _depth_rows, 1);  

	// allocate memory for all sampling units. 
	cuSample3f::CreateUniformSamplePattern(_depth_cols, _depth_rows, _sampling_param.uniform_step);
	cuSample3f::CreateRandomSamplePattern(_depth_cols, _depth_rows, _sampling_param.random_max_points, _sampling_param.ramdom_percentage);

	_producer_ready = true;
}
	
	
PointCloudProducer::~PointCloudProducer()
{
	
}


/*!
Set the sampling method 
@param method - set the sampling method of type SamplingMethod. 
	Can me RAW, UNIFORM, or RANDOM
@param param - set the sampling parameters associated to the 
	sampling method. 
*/
void PointCloudProducer::setSampingMode(SamplingMethod method, SamplingParam param)
{
	int w = _depth_cols;//_capture_device.getCols(COLOR);
	int h = _depth_rows;//_capture_device.getRows(COLOR);

	if (w <= 0 || h <= 0) {
		std::cout << "[ERROR] - setSampingMode: invalid camera resolution." << std::endl; 
	}

	_sampling_method = method;
	_sampling_param = param;
	_sampling_param.validate(); // check the values and correct if necessary. 

	// set sampling parameters and create the required cuda structures. 
	cuSample3f::CreateUniformSamplePattern(w, h, param.uniform_step);
	cuSample3f::CreateRandomSamplePattern(w, h, param.random_max_points, param.ramdom_percentage);
}

/*!
Set a point cloud filter methods. 
@param method - can be NONE or BILATERAL
@param param - the parameters for the filter
*/
void PointCloudProducer::setFilterMethod(FilterMethod method, FilterParams param)
{
	// pass-through function
	cuFilter3f::SetFilterMethod( method,  param);
}

/*
For normal vector calculation - set the related parameters
@param step_size - the size of the normal vector search window in pixel. 
	The function looks for depth values in adjacent pixels up to a distance step_size. 
*/
void PointCloudProducer::setNormalVectorParams(int step_size)
{
	if (step_size > 0) {
		_normal_vector_step_size = step_size;
	}
}


/*
Flip the normal vectors. They are not flipped by default.
But some cameras require them to be inverted. 
@param flip - true flips the normal vectors. 
*/
void  PointCloudProducer::setFlipNormalVectors(bool flip )
{
	if(flip)
		_flip_normal_vectors = -1.0;
	else
		_flip_normal_vectors = 1.0;
}



/*!
Process the current camera frame
@return true, if successful, otherwise false. 
*/
bool PointCloudProducer::process(void)
{
	if(!_producer_ready) return false;

	// grab an image
	cv::Mat img_depth;
	_capture_device.getDepthFrame(img_depth);

	// note that the pointcloud just resize itself if the current size does not match the image size. 
	_the_cloud.resize(_depth_rows * _depth_cols);

	switch (_sampling_method) {
		case RAW:
			run_sampling_raw( (float*)img_depth.data);
			break;
		case UNIFORM:
			run_sampling_uniform( (float*)img_depth.data);
			break;
		case RANDOM:
			run_sampling_random( (float*)img_depth.data);
			break;
	}

	// copy the new points. 
	copy_and_clear_points();

	return true;
}

// raw point cloud sampling
bool PointCloudProducer::run_sampling_raw(float* imgBuf)
{
	// sampling
	cuSample3f::UniformSampling((float*)imgBuf, _depth_cols, _depth_rows, _fx_depth, _fy_depth, _cx_depth, _cy_depth, 1, _flip_normal_vectors, false,
								(vector<float3>&)_pc_storage.points, 
								(vector<float3>&)_pc_storage.normals);

	return true;
}

// uniform point cloud sampling
bool PointCloudProducer::run_sampling_uniform(float* imgBuf)
{
	
	// sampling
	cuSample3f::UniformSampling((float*)imgBuf, _depth_cols, _depth_rows, _fx_depth, _fy_depth, _cx_depth, _cy_depth, _normal_vector_step_size, _flip_normal_vectors, false,
								(vector<float3>&)_pc_storage.points, 
								(vector<float3>&)_pc_storage.normals);

	return true;
}

// random point cloud sampling
bool PointCloudProducer::run_sampling_random(float* imgBuf)
{
	// sampling
	cuSample3f::RandomSampling((float*)imgBuf, _depth_cols, _depth_rows, _fx_depth, _normal_vector_step_size, _flip_normal_vectors, false,
								(vector<float3>&)_pc_storage.points, 
								(vector<float3>&)_pc_storage.normals);

	return true;
}


// copy points from the internal storage to the external
// and remove all points with values (0,0,0)
bool  PointCloudProducer::copy_and_clear_points(void)
{
/*
	// the copy_if operator does not work correctly. It returns an incorrect sized point cloud, and the storage containers get out of sync.
	// Added an ugly for loop to make it working. 

	_the_cloud.resize(_pc_storage.points.size());

	auto itp = std::copy_if(_pc_storage.points.begin(),  _pc_storage.points.end(), _the_cloud.points.begin(), [](Eigen::Vector3f p){return !(p.z()==0);});
	auto itn = std::copy_if(_pc_storage.normals.begin(), _pc_storage.normals.end(), _the_cloud.normals.begin(), [](Eigen::Vector3f n){return !(n.z()==0);});

	_the_cloud.points.resize(std::distance(_the_cloud.points.begin(), itp ));
	_the_cloud.normals.resize(std::distance(_the_cloud.normals.begin(), itn ));

	//std::copy_if(_pc_storage.points.begin(), _pc_storage.points.end(), std::back_inserter(_the_cloud.points), [](Eigen::Vector3f p) { return !(p[2] == 0.0f); });
	//std::copy_if(_pc_storage.normals.begin(), _pc_storage.normals.end(), std::back_inserter(_the_cloud.normals), [](Eigen::Vector3f n) {return !(n[2] == 0.0f); });
*/
	
	_the_cloud.points.clear();
	_the_cloud.normals.clear();
	// copy the epoint
	for (int i = 0; i < _pc_storage.points.size(); i++) {
		Eigen::Vector3f p0 = _pc_storage.points[i];
		Eigen::Vector3f n0 = _pc_storage.normals[i];

		if (!p0.z() == 0 && !n0.z() == 0) {
			_the_cloud.points.push_back(p0);
			_the_cloud.normals.push_back(n0);
		}
	}
	
	
//#define _TEST_COPY
#ifdef _TEST_COPY
// brute force test to check if the result is equal

/****************************************************************************
Test passed on:

- Feb 20, 2020, RR, Structure Core camera: no problems
- Aug 8, 2020, RR, Azure kinect, no problem. 
****************************************************************************/

	if (_the_cloud.points.size() != _the_cloud.normals.size()) {
		cout << "[TEST] - ERROR _the_cloud.points.size() != _the_cloud.normals.size()) -> " << _the_cloud.points.size() << " != " <<_the_cloud.normals.size() << endl;
	}

	PointCloud	test_cloud;
	
	// copy the epoint
	for (int i = 0; i < _pc_storage.points.size(); i++) {
		Eigen::Vector3f p0 = _pc_storage.points[i];
		Eigen::Vector3f n0 = _pc_storage.normals[i];

		if (!p0.z() == 0 && !n0.z() == 0) {
			test_cloud.points.push_back(p0);
			test_cloud.normals.push_back(n0);
		}
	}

	// compare the vectors
	if (test_cloud.points.size() == test_cloud.normals.size() && _the_cloud.points.size() == _the_cloud.normals.size() && test_cloud.points.size() == _the_cloud.points.size()) {
		//cout << "[TEST] - Sizes ok" << endl;

		int error_count = 0;
		for (int i = 0; i < test_cloud.size(); i++) {
			Eigen::Vector3f p0 = test_cloud.points[i];
			Eigen::Vector3f n0 = test_cloud.normals[i];
			Eigen::Vector3f p1 = _the_cloud.points[i];
			Eigen::Vector3f n1 = _the_cloud.normals[i];

			if((p0 - p1).norm()  > 0.01 )
				error_count++;
			if((n0 - n1).norm()  > 0.01 )
				error_count++;
		}

		if(error_count > 0) 
			cout << "[TEST] - Found " <<  error_count << " errors." << endl;

	}else
	{	
		cout << "[TEST] - ERROR - sizes do not match" << endl;
		cout << "[TEST] - " << test_cloud.points.size() << " != " << test_cloud.normals.size() << " && " << test_cloud.normals.size() << " != " << _the_cloud.normals.size() << " && " << test_cloud.points.size() << " != " << _the_cloud.points.size() << endl;

		int error_count = 0;
		for (int i = 0; i < test_cloud.size(); i++) {
			Eigen::Vector3f p0 = test_cloud.points[i];
			Eigen::Vector3f n0 = test_cloud.normals[i];
			Eigen::Vector3f p1 = _the_cloud.points[i];
			Eigen::Vector3f n1 = _the_cloud.normals[i];

			if ((p0 - p1).norm() > 0.01)
				error_count++;
			if ((n0 - n1).norm() > 0.01)
				error_count++;
		}
		if (error_count > 0)
			cout << "[TEST] - Found " << error_count << " errors." << endl;
	}

#endif

	return true;
}