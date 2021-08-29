#include "CPFProjection.h"



/*
Calculate the projection to a tangent plane.
Assuming the normal vector 'normal' describes the tangent plane, calculate a matrix 'projection_matrix' that allows
one to project any vector v onto the tangent plane of 'normal'
@param normal - a vector3 normal vector representing the plane.
@param tangent_plane_transform - the projection matrix that projects any vector onto the tangent plane.
@param transformation - a 3x3 transformation matrix to transform a 2D plane or any other graphics object into n.
*/
//static 
int CPFProjection::GetTangentPlaneTransform(Eigen::Vector3f normal, Eigen::Matrix3f& tangent_plane_transform, Eigen::Matrix3f& transformation, bool with_transformation)
{

	Eigen::Matrix3f nt;
	Eigen::RowVector3f normal_norm = normal;
	normal_norm.normalize();

	for(int i=0; i<3; i++){
		for(int j=0; j<3; j++){
			nt(i,j) = normal_norm[i] * normal_norm[j];
		}
	}
	//---------------------------------------------------
	// Calculate T, the tangent plane. 
	// identity matrix
	Eigen::Matrix3f I = Eigen::Matrix3f::Identity();
	Eigen::Matrix3f T = I - nt;

	tangent_plane_transform = T.transpose();
	

	//--------------------------------------------------------
	// Calculate a transformation matrix. 
	// Note there is no gurantee that the vectors v1, v2, v3 are orthogornal. 
	// They only help with a project. 
	// The code below computes a tranformation matrix, in case one needs to move something along the normal vector. 

	transformation = Eigen::Matrix3f::Identity();

	if(!with_transformation)
		return 0;

	Eigen::Vector3f y_axis;
	Eigen::Vector3f x_axis;

#ifdef _WITH_TESTING
	Eigen::Vector3f v1(T(0, 0), T(1, 0), T(2, 0));
	Eigen::Vector3f v2(T(0, 1), T(1, 1), T(2, 1));
	Eigen::Vector3f v3(T(0, 2), T(1, 2), T(2, 2));

	v1.normalize();
	v2.normalize();
	v3.normalize();

	float a1 = AngleBetween(normal_norm, v1);
	float a2 = AngleBetween(normal_norm, v2);
	float a3 = AngleBetween(normal_norm, v3);

	
	int a_case = 0;
	if(TheSame(a1, 1.57079)){
		y_axis = Cross(normal_norm, v1);
		x_axis =  v1;
		a_case = 1;
	}
	else if (TheSame(a2, 1.57079)){
		y_axis = Cross(normal_norm, v2);
		x_axis = v2;
		a_case = 2;
	}
	else if (TheSame(a3, 1.57079)){
		y_axis = Cross(normal_norm, v3);
		x_axis = v3;
		a_case = 3;
	}

	if(a_case == -1){
		std::cout << "[ERROR] - did not find perpendicular axis to construct a transformation matrix. " << std::endl;
		return a_case;
	}
	
	float a4 = AngleBetween(x_axis, y_axis);
	float a5 = AngleBetween(normal_norm, y_axis);

	
#else
	Eigen::Vector3f v1(T(0, 0), T(1, 0), T(2, 0));
	v1.normalize();
	y_axis = Cross(normal_norm, v1);
	x_axis = v1;

#endif

	y_axis.normalize();
	// the transformation here assumes that the objects z-axis is its normal axis. 
	// and that the x and y axis are perpendicular to the normal. 
	// For a plane, the plane extends into the x and y direction and the normal goes into its z-direction. 
	transformation(0, 0) = y_axis[0];
	transformation(1, 0) = y_axis[1];
	transformation(2, 0) = y_axis[2];

	transformation(0, 1) = x_axis[0];
	transformation(1, 1) = x_axis[1];
	transformation(2, 1) = x_axis[2];

	transformation(0, 2) = normal_norm[0];
	transformation(1, 2) = normal_norm[1];
	transformation(2, 2) = normal_norm[2];

	// transpose to get the transformation matrix that moves any graphica object into the normal. 
	transformation.transposeInPlace();

#ifdef _WITH_TESTING
	return a_case;
#else
	return 1;
#endif
}



/*
Project a vector 'v' to a plane given the projection matrix for the normal is known. Use GetTangentPlaneTransform to calcualte this projectoin.
It is calculated as
				v' = tangent_plane_transform v

@param v - the vector of type Vector3f
@param tangent_plane_transform - the transformation to transform v onto the plane given by its normal.
@return the new vector parallel to the plane.
*/
//static 
Eigen::Vector3f CPFProjection::ProjectVectorToPlane(const Eigen::Vector3f& v, const Eigen::Matrix3f& tangent_plane_transform)
{
	return tangent_plane_transform * v;
}


/*
Project a vector v to a tangent plane given by its normal n. The vector n is perpendicular to the plane.
The function computes
						T = I - n * n.T
						v' = T * v

@param v - the vector of type Vector3f.
@param n - the normal vector representing the plane.
@return the new vector v' which is parallel to the plane n.
*/
//static 
Eigen::Vector3f CPFProjection::ProjectVectorToPlane(const Eigen::Vector3f& v, const Eigen::Vector3f& n)
{
	Eigen::Matrix3f T;
	Eigen::Matrix3f M;

	// Get the transformation 
	GetTangentPlaneTransform(n, T, M, false);

	// Return the projection. 
	return ProjectVectorToPlane(v, T);
}


/*
Project vector v onto vector u
The function calculated the vector projection
				v' = [(v u)/ |u|^2] u
@param v - the vector to be projected as type vector3
@param u - the vector onto v is projected as type vector3.
@return the projected vector as vector3.
*/
//static 
Eigen::Vector3f CPFProjection::ProjectVectorToVector(const Eigen::Vector3f& v, const Eigen::Vector3f& u)
{
	float uv = v.dot(u);
	float u_length = u.norm();

	// project v onto u. 
	return (uv/(u_length*u_length)) * u;
}



// static
float CPFProjection::AngleBetween(const Eigen::Vector3f& a, const Eigen::Vector3f& b)
{
	Eigen::Vector3f a_norm = a.normalized();
	Eigen::Vector3f b_norm = b.normalized();
	Eigen::Vector3f c = a_norm.cross(b_norm);
	return atan2f(c.norm(), a_norm.dot(b_norm));
}


//static 
Eigen::Vector3f CPFProjection::Cross(const Eigen::Vector3f& a, const Eigen::Vector3f& b)
{
	return a.cross(b);
}


bool CPFProjection::TheSame(float a, float b){

	return fabs(a - b) < 0.001;
	
}


/*
This is just a function to check if all features for testing
are enabled. The automatic test for this class does not run if the features are disabled.
Add the preprocessor symbol _WITH_TESTING, to enable testing.
@return true, if testing is enabled, otherwise false.
*/
//static 
bool CPFProjection::TestEnabled(void)
{
#ifdef _WITH_TESTING
	return true;
#else
	return false;
#endif
}