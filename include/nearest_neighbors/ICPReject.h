#pragma once
/*
class ICPReject

@brief: The class implements several point-to-point outlier rejection mechanisms that allow one 
to reject points which a knn search initially identifies as nearest neighbors. 

Implemented outlier criteria are:
- distance: points are rejected as outliers if 
	their distance is larger than a threshold distance
- normal vector alignment: points are considered as outliers if their 
	normal vector direction deviate for more than a threshold
- distance + normal: both criteria combined. 

Rafael Radkowski
Iowa State University
rafael@iastate.edu
Aug 12, 2019

MIT License
------------------------------------------------------------------------------------------------------
Last edited:




*/
// stl
#include <iostream>
#include <vector>
#include <string>
#include <functional>
#include <algorithm>

// Eigen 3
#include <Eigen/Dense>

// local
#include "FDTypes.h"
#include "Types.h"
#include "MatrixUtils.h"


class ICPReject{

public:

		typedef enum {
			NONE,
			DIST,
			ANG,
			DIST_ANG
		}Testcase;


		ICPReject();
		~ICPReject();



		/*
		Test whether two points are close enough to be considered as inliers. 
		@param p0 - reference to the first point of type Vector3f with (x, y, z) coordinates. 
		@param p1 - reference to the second point of type Vector3f with (x, y, z) coordinates. 
		@return true - if the points are inliers, false if the are outliers. 
		*/
		bool testDistance(const Eigen::Vector3f& p0, const Eigen::Vector3f& p1);


		/*
		Test whether two normal vectors align so that they can be considered as inliers
		@param n0 - reference to the first normal vectors of type Vector3f with (nx, ny, nz) coordinates. 
		@param n1 - reference to the second normal vectors of type Vector3f with (nx, ny, nz) coordinates. 
		@return true - if the points are inliers, false if the are outliers. 
		*/
		bool testAngle(Eigen::Vector3f n0, Eigen::Vector3f n1);

		/*
		Test for both, point distance and normal alignment
		@param p0 - reference to the first point of type Vector3f with (x, y, z) coordinates. 
		@param p1 - reference to the second point of type Vector3f with (x, y, z) coordinates. 
		@param n0 - reference to the first normal vectors of type Vector3f with (nx, ny, nz) coordinates. 
		@param n1 - reference to the second normal vectors of type Vector3f with (nx, ny, nz) coordinates. 
		@return true - if the points are inliers, false if the are outliers. 
		*/
		bool testDistanceAngle(const Eigen::Vector3f& p0, const Eigen::Vector3f& p1, Eigen::Vector3f n0, Eigen::Vector3f n1);



		/*
		Test for outliers and select the case. 
		@param p0 - reference to the first point of type Vector3f with (x, y, z) coordinates. 
		@param p1 - reference to the second point of type Vector3f with (x, y, z) coordinates. 
		@param n0 - reference to the first normal vectors of type Vector3f with (nx, ny, nz) coordinates. 
		@param n1 - reference to the second normal vectors of type Vector3f with (nx, ny, nz) coordinates. 
		@return true - if the points are inliers, false if the are outliers. 
		*/
		bool test(const Eigen::Vector3f& p0, const Eigen::Vector3f& p1, Eigen::Vector3f n0, Eigen::Vector3f n1, Testcase testcase );


		/*
		Set the maximum distance limit for two points to be considered as inliers. 
		@param max_distance - float value with a limit > 0.0;
		Note that the point will be set to 0.0 if the value is negative
		@return true
		*/
		bool setMaxThreshold(float max_distance);

		/*
		Set the maximum angle distance for two normal vectors to be considered as aligned. 
		@param max_angle_degree - float value in the range [0,180]. The value must be in degree. 
		@return true
		*/
		bool setMaxNormalVectorAngle(float max_angle_degree);

private:

		float	_max_distance;
		float	_max_angle;

};