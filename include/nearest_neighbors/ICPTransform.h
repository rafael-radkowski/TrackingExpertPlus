#pragma once
/*
class ICPTransform 

This class implements an ICP point alignment from two set of points. 
It only implements the transformation part of a point-to-point ICP algorithm.

Means, the points in the input vectors points0 and points1 MUST be index-aligned nearest neighbors
and come as Vector3f with [x,y, z] per point. 

All matrices are in Eigen-standard column major, with matrix to points from the left -> M p

Note, the class uses float, which reduces the precision if the point values are already numerically
small.

Usage:
Matrix4f Rt = ICPCorrection::Compute(points0, points1);

Rafael Radkowski
Iowa State University
rafael@iastate.edu
Nov. 17, 2017
MIT License
---------------------------------------------------------------

Last edited:
June 11, 2020, RR
- Added a function to return the centroid of a object. 

*/



// stl
#include <iostream>
#include <vector>
#include <numeric>

// Eigen 3
#include <Eigen/Dense>

// local
#include "FDTypes.h"


using namespace Eigen;
using namespace std;

namespace texpert {

class ICPTransform
{
public:

	/*
	Computes the difference between two set of points
	@param points0 - a vector with Eigen::Vec3 points, [x, y, z]
	@param points1 - a vector with Eigen::Vec3 points,  [x, y, z]
	@return the rotation and translation difference between those points as 
		4x4 matri in homogenous coordinate frames.
	*/
	static  Matrix4f Compute(vector<Vector3f>& points0, vector<Vector3f>& points1, Pose initial_pose);




	/*
	Calculate the rotation using the method from Arun et al. 1987
	Arun, K., Huang, T.S., and Blostein, S.D., 1987. “Least - squares fitting of two 3 - d point sets”.
	IEEE Transactions on Pattern Analysis and Machine Intel - ligence, PAMI - 9(5), Sept, pp. 698–700

	@param pVec0 - first point array, each element needs to be a vector3 [[x, y, z,], [x, y, z,], ...]
	@param pVec1 - second point array, each element needs to be a vector3 [[x, y, z,], [x, y, z,], ...]
	@param return: 3x3 matrix with the delta rotation
	*/
	static Matrix3f CalcRotationArun( vector<Vector3f>& pVec0, vector<Vector3f>& pVec1);


	/*
	Calculates the translation delta between both point sets as mean.
	@param pVec0 - first point array, each element needs to be a vector3 [[x, y, z], [x, y, z], ...]
	@param pVec1 - second point array, each element needs to be a vector3 [[x, y, z], [x, y, z], ...]
	@param return: Vec 3 with the delta translation
	*/
	static Vector3f CalculateTranslation(vector<Vector3f>& pVec0, vector<Vector3f>& pVec1);

	/*
	Check the root mean square error between the two points sets. 
	Do not forget to translate the reference point set before checking. 
	@param pVec0 - first point array, each element needs to be a vector3 [[x, y, z], [x, y, z], ...]
	@param pVec1 - second point array, each element needs to be a vector3 [[x, y, z], [x, y, z], ...]
	@param return: float values representing the RMS
	*/
	static float CheckRMS(vector<Vector3f>& pVec0, vector<Vector3f>& pVec1);


	/*!
	Return the centroid of a set of points. 
	@param pVec0 - a vector with points (x, y, z).
	@return a vector containing the centroid of the object in local space. 
	*/
	static Vector3f CalculateCentroid(vector<Vector3f>& pVec0);

};

} //texpert 