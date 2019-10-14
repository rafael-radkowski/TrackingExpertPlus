#include "FDTools.h"


using namespace texpert;

// static
float FDTools::angleBetween(const Eigen::Vector3f& a, const Eigen::Vector3f& b)
{
	Eigen::Vector3f a_norm = a.normalized();
	Eigen::Vector3f b_norm = b.normalized();
	Vector3f c = a_norm.cross(b_norm);
	return atan2f(c.norm(), a_norm.dot(b_norm));
}


//static 
Affine3f FDTools::getRefFrame(Vector3f& p, Vector3f& n)
{

	Vector3f axis = n.cross(Eigen::Vector3f::UnitX());

	if (axis(0) == 0.0f && axis(1) == 0.0f && axis(2) == 0.0f) {
		axis = Eigen::Vector3f::UnitX();
	}
	else {
		axis.normalize();
	}

	// create an angle axis transformation that rotates A degree around the axis.
	Eigen::AngleAxisf rot(angleBetween(n, Eigen::Vector3f::UnitX()), axis);

	Eigen::Affine3f m = rot * Eigen::Translation3f(-p);
	Eigen::Affine3f T = Eigen::Affine3f(rot * Eigen::Translation3f(-p));

	// Multiply the point with the transformatiom matrix. 
	// -p moves the point into the origin of its own new coordinate frame.
	// The result is a 4x4 matrix - and x-axis aligned coordinate frame for the 
	// point p. 
	return Eigen::Affine3f(rot * Eigen::Translation3f(-p));
}




PPFDiscreet FDTools::DiscretizePPF(const Vector3f& p1, const Vector3f& n1, const Vector3f& p2, const Vector3f& n2,
										const float distance_step,
										const float angle_step) 
{

	Eigen::Vector3f dir = p2 - p1;

	// Compute the four PPF components
	float f1 = dir.norm();
	float f2 = angleBetween(dir, n1);
	float f3 = angleBetween(dir, n2);
	float f4 = angleBetween(n1, n2);

	// Discretize the feature and return it
	PPFDiscreet ppf;
	ppf[0] = static_cast<uint32_t>(f1 / distance_step);
	ppf[1] = static_cast<uint32_t>(f2 / angle_step);
	ppf[2] = static_cast<uint32_t>(f3 / angle_step);
	ppf[3] = static_cast<uint32_t>(f4 / angle_step);

	return ppf;
}
