


// stl
#include <stdio.h>
#include <iostream>
#include <vector>
#include <algorithm>
#include <fstream>

// cuda
#include "cuda_runtime.h"

// OpenCV
#include <opencv2\highgui.hpp>
#include <opencv2\core.hpp>
#include <opencv2\imgproc.hpp>



// local
#include "cuPCU.h"

using namespace std;
using namespace cv;
using namespace tacuda;

string type2str(int type) {
	string r;

	uchar depth = type & CV_MAT_DEPTH_MASK;
	uchar chans = 1 + (type >> CV_CN_SHIFT);

	switch (depth) {
	case CV_8U:  r = "8U"; break;
	case CV_8S:  r = "8S"; break;
	case CV_16U: r = "16U"; break;
	case CV_16S: r = "16S"; break;
	case CV_32S: r = "32S"; break;
	case CV_32F: r = "32F"; break;
	case CV_64F: r = "64F"; break;
	default:     r = "User"; break;
	}

	r += "C";
	r += (chans + '0');

	return r;
}



int main(int argc, char** argv)
{

	



	//--------------------------------------------------------------------------------------


	//
	Mat range_in = cv::imread("../test_rangemap.png", CV_LOAD_IMAGE_ANYDEPTH | CV_LOAD_IMAGE_GRAYSCALE);

	int width = range_in.cols;
	int height = range_in.rows;

	/*cv::namedWindow("Range image in");
	cv::moveWindow("Range image in", 20, 500);
*/
	//cv::namedWindow("Range image out");
	//cv::moveWindow("Range image out", 700, 500);
	
	cout << "Input image is of type " <<type2str(range_in.type()) << " (<- Must be of type CV_16UC1)." << endl;

	//------------------------------------------------------------------------------
	// Conversion to get the range image into a type float datatype


	//std::ofstream data("test_image.txt");

	//for (int i = 0; i < width; i++)
	//{
	//	for (int j = 0; j < height; j++)
	//	{
	//		unsigned short intensity = range_in.at<unsigned short>(j, i);
	//		/*float blue = intensity.val[0];
	//		float green = intensity.val[1];
	//		float red = intensity.val[2];*/

	//		data  << intensity << ", ";
	//	}
	//	data << "\n";
	//}

	//data.close();


//	cv::imshow("Range image in", range_in);




	//------------------------------------------------------------------------------
	// Range image processing
	int tests = 100;

	// Allocating memory
	cuPCU::AllocateDeviceMemory(range_in.cols, range_in.rows, 1);

	// Create a sample pattern for uniform sampling
	cuSample::CreateUniformSamplePattern(range_in.cols, range_in.rows, 6);

	//for(int i=0; i<10; i++)
	//	cuSample::CreateRandomSamplePattern(range_in.cols, range_in.rows, 0.8, 5000);

	//return 1;

	// Compute
	vector<float3> points_host(width * height);
	vector<float3> normals_host(width * height);

	cout << "Starting " << tests << " test cycles." << endl;

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	cudaEventRecord(start);
	for (int i = 0; i < tests; i++)
	{
		//cout << "Calculate round " << i << endl;
	//	cuPCU::CreatePointCloud((unsigned short*)range_in.data, range_in.cols, range_in.rows, 1, 577.0f, 6, points_host, normals_host, false);
		
		// Create a uniformly sampled point cloud pattern.
		cuSample::UniformSampling((unsigned short*)range_in.data, range_in.cols, range_in.rows, 577.0f, 6, points_host, normals_host);

	}
	cudaEventRecord(stop);

	cudaEventSynchronize(stop);
	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);

	cout << "Processing time: " << milliseconds <<  " ms." << endl;
	cout << "Average Processing time: " << milliseconds / float(tests) << " ms." << endl;


	

	std::ofstream off("test_points.obj");

	for (int i = 0; i < points_host.size(); i++)
	{
		if (points_host[i].x == 0.0 && points_host[i].y == 0 && points_host[i].z == 0) continue;

		off << "vn " << normals_host[i].x << "  " << normals_host[i].y << "  " << normals_host[i].z << "\n";
		off << "v " << points_host[i].x << "  " << points_host[i].y << "  " << points_host[i].z << "\n";
	}
	off.close();



	cv::Mat output_depth = cv::Mat::zeros(height, width, CV_32FC3);
	cv::Mat output_normals = cv::Mat::zeros(height, width, CV_32FC3);

	cuPCU::GetDepthImage(output_depth);
	cuPCU::GetNormalImage(output_normals);

	// Free the memory
	cuPCU::FreeDeviceMemory();


	cv::Mat out0, out1;
	output_depth.convertTo(out0, CV_8UC3);
	output_normals.convertTo(out1, CV_8UC3);

	Mat bgr[3];   //destination array
	split(output_depth, bgr);//

	cv::imwrite("depth_image.png", bgr[2]);
	cv::imwrite("normal_image.png", output_normals);



	//cv::imshow("Range image out", out0);
	//cv::imshow("Normal image out", out1);

	cv::waitKey(10);

	

	//cv::namedWindow("Output image");
	//cv::moveWindow("Output image", 700, 20);
	//cv::imshow("Output image", output_img);


	//cv::waitKey();




	return 1;
};