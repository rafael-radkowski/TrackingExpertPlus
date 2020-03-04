#include "ICPReject.h"


ICPReject::ICPReject()
{
	_max_distance = 0.01;
	_max_angle = 0.1;
}

ICPReject::~ICPReject()
{

}

/*
Test whether two points are close enough to be considered as inliers. 
@param p0 - reference to the first point of type Vector3f with (x, y, z) coordinates. 
@param p1 - reference to the second point of type Vector3f with (x, y, z) coordinates. 
@return true - if the points are inliers, false if the are outliers. 
*/
bool ICPReject::testDistance(const Eigen::Vector3f& p0, const Eigen::Vector3f& p1)
{
	return ( ((p0 - p1).norm() < (_max_distance*_max_distance)) ? true : false );
}


/*
Test whether two normal vectors align so that they can be considered as inliers
@param n0 - reference to the first normal vectors of type Vector3f with (nx, ny, nz) coordinates. 
@param n1 - reference to the second normal vectors of type Vector3f with (nx, ny, nz) coordinates. 
@return true - if the points are inliers, false if the are outliers. 
*/
bool ICPReject::testAngle(Eigen::Vector3f n0, Eigen::Vector3f n1)
{
//https://developer.rhino3d.com/samples/cpp/calculate-the-angle-between-two-vectors/
	
	// normalize
	n0.normalize();
	n1.normalize();

	// calculate the dot product
	float dot = n0.dot(n1);

	// for into -1 , 1 range for acos
	dot = std::max(-1.0f, std::min(1.0f, dot));
	
	// return the angle
	float angle = std::acosf(dot);

	return ( (angle < _max_angle) ? true : false);

}

/*
Test for both, point distance and normal alignment
@param p0 - reference to the first point of type Vector3f with (x, y, z) coordinates. 
@param p1 - reference to the second point of type Vector3f with (x, y, z) coordinates. 
@param n0 - reference to the first normal vectors of type Vector3f with (nx, ny, nz) coordinates. 
@param n1 - reference to the second normal vectors of type Vector3f with (nx, ny, nz) coordinates. 
@return true - if the points are inliers, false if the are outliers. 
*/
bool ICPReject::testDistanceAngle(const Eigen::Vector3f& p0, const Eigen::Vector3f& p1, Eigen::Vector3f n0, Eigen::Vector3f n1)
{
	return ( testDistance(p0, p1) );
	return ( testDistance(p0, p1) && testAngle(n0, n1) );
}


/*
Set the maximum distance limit for two points to be considered as inliers. 
@param max_distance - float value with a limit > 0.0;
Note that the point will be set to 0.0 if the value is negative
@return true
*/
bool ICPReject::setMaxThreshold(float max_distance)
{
	_max_distance = std::max(0.0f, max_distance);
	return true;
}

/*
Set the maximum angle distance for two normal vectors to be considered as aligned. 
@param max_angle_degree - float value in the range [0,180]. The value must be in degree. 
@return true
*/
bool ICPReject::setMaxNormalVectorAngle(float max_angle_degree)
{
	float angle = std::min(180.0f, std::max(0.0f, max_angle_degree));
	_max_angle = angle/ 180.0 * 3.14159265358;

	return true;
}