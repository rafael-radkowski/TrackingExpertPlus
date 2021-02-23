#pragma once

/*
class PointCloudTrans

This class contains all of the functions necessary to translate a point
or point cloud given an input Affine matrix.

William Blanchard
Iowa State University
Feb 2021
wsb@iastate.edu
MIT License
---------------------------------------------------------------
Last edited:

Feb 23, 2021, WB
- Initial commit
*/

#include <Eigen/Dense>
#include <vector>

#include "MatrixConv.h"

using namespace std;

class PointCloudTrans
{
public:
	static Eigen::Vector3d Transform(Eigen::Affine3f& Rt, Eigen::Vector3d& point);

	static vector<Eigen::Vector3d> Transform(Eigen::Affine3f& Rt, vector<Eigen::Vector3d>& points);

	static void TransformInPlace(Eigen::Affine3f& Rt, Eigen::Vector3d& point);

	static void TransformInPlace(Eigen::Affine3f& Rt, vector<Eigen::Vector3d>& points);
};