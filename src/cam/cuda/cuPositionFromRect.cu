#pragma once
#include "cuPositionFromRect.h"
#include "cuDeviceMemory.h"

#include "../MatrixToEuler.h"


using namespace  tacuda;



namespace tacuda
{
	// Pointer to the cuda image
	float*			image_ptr_dev = NULL;

	// device positions
	float3*			positions_dev = NULL;

	// memory for region of interests
	int				maxROI = 400 * 400;

	// image width and height
	int				width;
	int				height;
	int				channels = 3;

	bool			ready = false;
}




//////////////////////////////////////////////////////////////////////////////////////
// class




/*
Init the class
*/
//static 
void cuPositionFromRect::Init(int width, int height)
{
	tacuda::image_ptr_dev = cuDevMem::DevPointImagePtr();
	tacuda::height = height;
	tacuda::width = width;

	assert(tacuda::image_ptr_dev != NULL);
	if (tacuda::image_ptr_dev == NULL) return;

	tacuda::ready = true;


	//---------------------------------------------------------------------------------
	// Allocating memory

	// image memory on device. It stores the input image, the depth values as array A(i) = {d0, d1, d2, ...., dN} as float
//	cudaError err = cudaMalloc((void **)&PositionFromRect::positions_dev, sizeof(float3)*(PositionFromRect::maxROI));
	//if (err != 0) { std::cout << "\n[cuPositionFromRect] - cudaMalloc error for positions_dev.\n"; }
}


/*
Extract the position of a set of points within a given region of interest. The position
is the mean of all valid points.
@param rect_vector - a vector with opencv rectangles. Each rectangle gives a region of interest
@param position - a reference to a vector to store the positions.
@return - the number of extracted positions
*/
//static 
int cuPositionFromRect::GetPosition(vector<cv::Rect> rect_vector, vector<float3>& position, vector<float3>& orientation)
{
	if (tacuda::ready == false)
		return 0;


	int size = rect_vector.size();
	position.resize(size);
	orientation.resize(size);
	int count = 0;
	for (int i = 0; i < size; i++)
	{
		int ret =cuPositionFromRect::ExtractPositionAndOrientation(rect_vector[i], position[i], orientation[i], true);
		count += ret;
	}

	return count;

}





float mymean(cv::Mat I, int& count)
{
	float sum = 0;
	count = 0;
	cv::MatIterator_<float> it, end;
	for (it = I.begin<float>(), end = I.end<float>(); it != end; ++it)
	{
		if ((*it) != 0.0)
		{
			sum += (*it);
			count++;
		}
	}
	
	return sum / (float)count;
}


/*
Extract the position of one region of interest
@param - the rectangle given the region of interest.
@param - location to store the position
@return - 1 if the location was successfully extracted
*/
// static 
int cuPositionFromRect::ExtractPositionAndOrientation(cv::Rect rect, float3& position, float3& orientation, bool with_orient)
{
	int output_size = tacuda::width* tacuda::height * 3 * sizeof(float);  // three channels
	

	cv::Mat output_points = cv::Mat::zeros(tacuda::height, tacuda::width, CV_32FC3);
	cudaMemcpy((float*)output_points.data, tacuda::image_ptr_dev, output_size, cudaMemcpyDeviceToHost);
	

	// The flip operation is too slow
	//cv::flip(output_points, output_points_flipped, 1);
	//cv::imshow("Range image out", test_outf);
	//cv::waitKey(1);

	// rectangle flip. The image is flipped
	rect.x = tacuda::width - rect.x - rect.width;

	cv::Mat image_roi = output_points(rect);

	int count = 0;
	vector<cv::Mat> channels;
	split(image_roi, channels);
	cv::Scalar u = mymean(channels[0], count);
	cv::Scalar v = mymean(channels[1], count);
	cv::Scalar w = mymean(channels[2], count);
	float x = u[0];
	float y = v[0];
	float z = w[0];

	position.x = x;
	position.y = y;
	position.z = z;



	if (!with_orient) return 1;


	int i = 0;
	cv::Mat data_pts = cv::Mat(count, 3, CV_64FC1);
	cv::MatIterator_<float3> it, end;
	for (it = image_roi.begin<float3>(), end = image_roi.end<float3>(); it != end; ++it)
	{
		if ((*it).z != 0.0)
		{
			data_pts.at<double>(i, 0) = (double)(*it).x;
			data_pts.at<double>(i, 1) = (double)(*it).y;
			data_pts.at<double>(i, 2) = (double)(*it).z;
			i++;
		}
	}

	if (data_pts.rows == 0)
	{
		orientation.x = 0.00;
		orientation.y = 0.0;
		orientation.z = 0.0f;
			return 1;
	}

	cv::PCA pca_analysis(data_pts, cv::Mat(), CV_PCA_DATA_AS_ROW);


	////Store the eigenvalues and eigenvectors
	vector<osg::Vec3f> eigen_vecs(3);
	vector<double> eigen_val(3);
	for (int i = 0; i < 3; ++i)
	{
		eigen_vecs[i] = osg::Vec3d(pca_analysis.eigenvectors.at<double>(i, 0),
									pca_analysis.eigenvectors.at<double>(i, 1),
									pca_analysis.eigenvectors.at<double>(i, 2));

		eigen_val[i] = pca_analysis.eigenvalues.at<double>(0, i);
	}

	double angle1 = atan2(eigen_vecs[0].y(), eigen_vecs[0].z()); // orientation in radians
	double angley = atan2(eigen_vecs[0].z(), eigen_vecs[0].x()); // orientation in radians
	double angle3 = atan2(eigen_vecs[0].y(), eigen_vecs[0].x()); // orientation in radians

	//_cprintf("mean x: %lf, y: %lf, z: %lf\n", x, y, z);
	//_cprintf("angle x: %lf, y: %lf, z: %lf\n", angle1/3.1415 * 180, angley / 3.1415 * 180, angle3 / 3.1415 * 180);

	orientation.x = angle1 / 3.1415 * 180;
	orientation.y = 180.0 - ( angley / 3.1415 * 180);
	orientation.z = 0.0f;

	/*eigen_vecs[0].normalize();
	eigen_vecs[1].normalize();
	eigen_vecs[2].normalize();
	
	osg::Matrix mat;

	mat.set(eigen_vecs[0].x(), eigen_vecs[0].y(), eigen_vecs[0].z(), 0,
			eigen_vecs[1].x(), eigen_vecs[1].y(), eigen_vecs[1].z(), 0,
			eigen_vecs[2].x(), eigen_vecs[2].y(), eigen_vecs[2].z(), 0,
			0,0,0,1);

	vector<float> R = MatrixToEuler::eulerAngles(mat);

	_cprintf("\nangle x: %lf, y: %lf, z: %lf\n", R[0] / 3.1415 * 180, R[1] / 3.1415 * 180, R[2] / 3.1415 * 180);
	_cprintf("eig 0: %lf, y: %lf, z: %lf, %lf\n", eigen_vecs[0].x(), eigen_vecs[0].y(), eigen_vecs[0].z(), eigen_val[0]);
	_cprintf("eig 1: %lf, y: %lf, z: %lf, %lf\n", eigen_vecs[1].x(), eigen_vecs[1].y(), eigen_vecs[1].z(), eigen_val[1]);
	_cprintf("eig 2: %lf, y: %lf, z: %lf, %lf\n", eigen_vecs[2].x(), eigen_vecs[2].y(), eigen_vecs[2].z(), eigen_val[2]);*/


	/*cv::Mat test_out;
	image_roi.convertTo(test_out, CV_8UC3, 80.0);

	cv::imshow("Range image out", test_out);
	cv::waitKey(1);*/

	return 2;
}




