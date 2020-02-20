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
	_sampling_param.uniform_step = 4;
	_normal_vector_step_size = 4;
	_depth_cols = -1;
	_depth_rows = -1;

	cv::Mat param = _capture_device.getCameraParam();

	_fx_depth = param.at<float>(0,0);
	_fy_depth = param.at<float>(1,1);

	_cx_depth = 0.0;
	_cy_depth = 0.0;


//	if (_capture_device.isOpen()) {
//		cout << "[ERROR] - PointCloudProducer: error when opening the camera." << endl;
//		return;
//	}

	// fetch the resolution. 
	_depth_cols = _capture_device.getCols(COLOR);
	_depth_rows = _capture_device.getRows(COLOR);


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
	int w = _capture_device.getCols(COLOR);
	int h = _capture_device.getRows(COLOR);

	_sampling_method = method;
	_sampling_param = param;

	// set sampling parameters and create the required cuda structures. 
	cuSample3f::CreateUniformSamplePattern(w, h, param.uniform_step);
	cuSample3f::CreateRandomSamplePattern(w, h, param.random_max_points, param.ramdom_percentage);
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
			return run_sampling_raw( (float*)img_depth.data);
			break;
		case UNIFORM:
			return run_sampling_uniform( (float*)img_depth.data);
			break;
		case RANDOM:
			return run_sampling_random( (float*)img_depth.data);
			break;
	}

}

// raw point cloud sampling
bool PointCloudProducer::run_sampling_raw(float* imgBuf)
{
	// sampling
	cuSample3f::UniformSampling((float*)imgBuf, _depth_cols, _depth_rows, _fx_depth, _fy_depth, _cx_depth, _cy_depth, 1, false,
								(vector<float3>&)_the_cloud.points, 
								(vector<float3>&)_the_cloud.normals);

	return true;
}

// uniform point cloud sampling
bool PointCloudProducer::run_sampling_uniform(float* imgBuf)
{
	
	// sampling
	cuSample3f::UniformSampling((float*)imgBuf, _depth_cols, _depth_rows, _fx_depth, _fy_depth, _cx_depth, _cy_depth, _normal_vector_step_size, false,
								(vector<float3>&)_the_cloud.points, 
								(vector<float3>&)_the_cloud.normals);

	return true;
}

// random point cloud sampling
bool PointCloudProducer::run_sampling_random(float* imgBuf)
{
	// sampling
	cuSample3f::RandomSampling((float*)imgBuf, _depth_cols, _depth_rows, _fx_depth, _normal_vector_step_size, false,
								(vector<float3>&)_the_cloud.points, 
								(vector<float3>&)_the_cloud.normals);

	return true;
}
