#pragma once

/*
class PointCloudTrans

This class contains multiple functions necessary to transform a point
or point cloud in various ways.

This static class allows the user to transform a point or point cloud in various
ways depending on their situation.  The Transform and TransformInPlace functions
simply transform a given point or point cloud by a transform matrix.  Transform
returns the result of this transformation while TransformInPlace changes the 
point passed into the function itself.  TransformAboutCentroid takes an existing
transformation matrix and 

William Blanchard
Iowa State University
Feb 2021
wsb@iastate.edu
MIT License
---------------------------------------------------------------
Last edited:

March 2, 2021, WB
- Added documentation
*/

//std
#include <vector>

//sdk
#include <Eigen/Dense>

//local
#include "MatrixConv.h"

using namespace std;

class PointCloudTrans
{
public:

	/*
		Returns the position of a point transformed by a transformation matrix.

		@param Rt - The transformation matrix
		@param point - The point that will be transformed by the transformation matrix

		@return the position of the point after it has been transformed by the 
			transformation matrix.
	*/
	static Eigen::Vector3d Transform(Eigen::Affine3f& Rt, Eigen::Vector3d& point);

	/*
		Returns a cloud of points that is the result of transforming a given point
		cloud by a transformation matrix.

		@param Rt - The transformation matrix
		@param points - The point cloud that will be transformed by the transformation
			matrix

		@return the point cloud after it has been transformed by the transformation
			matrix.
	*/
	static vector<Eigen::Vector3d> Transform(Eigen::Affine3f& Rt, vector<Eigen::Vector3d>& points);

	/*
		Transforms the given point according to a transformation matrix

		@param Rt - The transformation matrix
		@param point - The point that will be transformed by the transformation matrix
	*/
	static void TransformInPlace(Eigen::Affine3f& Rt, Eigen::Vector3d& point);

	/*
		Transforms the given point cloud according to a transformation matrix

		@param Rt - The transformation matrix
		@param points - The point cloud that will be transformed by the transformation
			matrix
	*/
	static void TransformInPlace(Eigen::Affine3f& Rt, vector<Eigen::Vector3d>& points);

	/*
		A transformation specific to the ICP algorithm.  
	*/
	static void TransformAboutCentroid(Eigen::Vector3f& accum_t, Eigen::Matrix3f& accum_R, Eigen::Vector3f& centroid, Eigen::Affine3f& init_affine, Eigen::Matrix4f& result);
};