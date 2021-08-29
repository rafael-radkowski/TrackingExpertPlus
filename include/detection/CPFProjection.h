#pragma once
/*
@class CPFProjection

The class contains calculates the orthogonal plane to a normal vector, provides a transformation matrix to rotate and
plane into the calculated plane, and it projects the any other normal onto the plane. 

The input is a normal vector n, which describes the plane. The normal vector is perpendicular to the plane. 
The tangent plane is calculated as
	
					T = I - n * n.T

with I the identity matrix and T the perpendicular tangent plane projection. 

To project a vector onto this normal, calcualte 

					v' = T v

with v, the arbitrary vector and v' its projection onto the tangent plane. 


Calculatiion a transformation matrix: note that the axis in T do not need to per orthogonal to each other. 
Thus, T is not a valid transformation matrix. 

To find a transformation matrix, one first normalize the matrix T, its axis in particular so that v1, v2, v3 are of length one,
with T == [ v1 | v2 | v3 ], the three columns spanning the T-space. 

Next, check whether one of these axis, v1, v2, v3 is orthogonal to n. They all should be, just to be sure. 
The code takes v1 if valid (90 deg) and the normal vector n as two axis of the coordinate system. 
The cross project 
						y = n x v1 

is the third axis of the transformation matrix. 
The entire transformation matrix is composed of three column vectors y, v1, n as:

					M = [y | v1 | n ]^{-1}    (perhaps not the best naming. 

which can be directly used to transform any plane or other object. 
M can be directly used as a glm transformation matrix. 

To get the projected vector v' into local object space:
					v'_local  = M v'

(Eigen::Vector3f projection_local = transformation * projection;)

Rafael Radkowski
radkowski.dev@gmail.com
Aug 28, 2021
MIT License

------------------------------------------------------------------------------------------------------------------------------------
Last edited. 

*/
// stl. 
#include <iostream>
#include <string>

// glm
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtx/transform.hpp>  // transformation
#include <glm/gtx/quaternion.hpp> // quaternions

// eigen 3
#include <Eigen/Dense>

// enable for testing. 
// disable for speed. 
//#define _WITH_TESTING

class CPFProjection
{
public:


	/*
	Calculate the projection to a tangent plane. 
	Assuming the normal vector 'normal' describes the tangent plane, calculate a matrix 'projection_matrix' that allows
	one to project any vector v onto the tangent plane of 'normal'
	@param normal - a vector3 normal vector representing the plane.
	@param tangent_plane_transform - the projection matrix that projects any vector onto the tangent plane. 
	@param transformation - a 3x3 transformation matrix to transform a 2D plane or any other graphics object into n. 
	@param with_transformation - enable or disable the 'transformation' calculation. The matrix 'transformation' will be an identity matrix if set to false. Default is true. 
	*/
	static int GetTangentPlaneTransform( Eigen::Vector3f normal,  Eigen::Matrix3f& tangent_plane_transform, Eigen::Matrix3f& transformation, bool with_transformation = true);

	/*
	Project a vector 'v' to a plane given the projection matrix for the normal is known. Use GetTangentPlaneTransform to calcualte this projectoin. 
	It is calculated as 
					v' = tangent_plane_transform v

	@param v - the vector of type Vector3f
	@param tangent_plane_transform - the transformation to transform v onto the plane given by its normal. 
	@return the new vector parallel to the plane. 
	*/
	static Eigen::Vector3f ProjectVectorToPlane(const Eigen::Vector3f& v, const Eigen::Matrix3f& tangent_plane_transform);


	/*
	Project a vector v to a tangent plane given by its normal n. The vector n is perpendicular to the plane. 
	The function computes 
							T = I - n * n.T
							v' = T * v
					
	@param v - the vector of type Vector3f.
	@param n - the normal vector representing the plane. 
	@return the new vector v' which is parallel to the plane n. 
	*/
	static Eigen::Vector3f ProjectVectorToPlane(const Eigen::Vector3f& v, const Eigen::Vector3f& n);


	/*
	Project vector v onto vector u
	The function calculated the vector projection
					v' = [(v u)/ |u|^2] u
	@param v - the vector to be projected as type vector3
	@param u - the vector onto v is projected as type vector3.
	@return the projected vector as vector3. 
	*/
	static Eigen::Vector3f ProjectVectorToVector(const Eigen::Vector3f& v, const Eigen::Vector3f& u);


	/*
	This is just a function to check if all features for testing 
	are enabled. The automatic test for this class does not run if the features are disabled. 
	Add the preprocessor symbol _WITH_TESTING, to enable testing.
	@return true, if testing is enabled, otherwise false. 
	*/
	static bool TestEnabled(void);

private:

	/*
	Calculate the angle between the vectors a and b.
	@param a - the first vector as vector 3
	@param b - the second vector as vector3
	@return the angle between the two vectors in RAD.
	*/
	static float AngleBetween(const Eigen::Vector3f& a, const Eigen::Vector3f& b);

	/*
	Calculate the cross product between the vectors a and b. 
	@param a - the first vector as vector 3
	@param b - the second vector as vector3
	@return a vector perpendicular to the plane spanned by a and b. 
	*/
	static Eigen::Vector3f Cross(const Eigen::Vector3f& a, const Eigen::Vector3f& b);

	/*
	Check if the vectors a1 and a2 are the same. 
	Note that the default tolerance is 0.001
	@param a1 - the first vector as vector 3
	@param a2 - the second vector as vector3
	@return true if they are the same, otherwise false. 
	*/
	static bool TheSame(float a1, float a2);
};

