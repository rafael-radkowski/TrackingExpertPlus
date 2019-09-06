#include "PointCloudEval.h"


using namespace isu_ar;



/*
Calculate the root mean square error between matched points of pc_ref and pc_test.
The function calculates
			
	1/N sum | pc_ref_i - T_t2r * pc_test_j  |^2

The transformation T is multiplied with all points in pc_test. 
The points i and j are index aligned. The vector matches contains a test point for each reference points. 

i  0    1    2               N
[ j_0 | j_1 | j_2 | ...... | j_N ] 
The index refers to the reference points. 

@param pc_ref -  pointer to the source point cloud of type PointCloud
@param pc_test - pointer to the destination point cloud of type PointCloud
@param T_t2r - vec 3 with the translation in x, y, z/
@param matches - vector with index aligned matches between the points of ref and test. 
@return  root mean squared error;
*/
//static 
double PointCloudEval::RMS(PointCloud& pc_ref, PointCloud& pc_test, Eigen::Affine3f& T_t2r, std::vector<int> matches)
{
	double error = 0;
	int idx = 0;
	vector<Eigen::Vector3f>& r = pc_test.points;

	for_each(pc_ref.points.begin(), pc_ref.points.end(), [&](Eigen::Vector3f& p) {

		error += (p - T_t2r * r[matches[idx]]).squaredNorm();
		idx++;
	});

	return error;

}