#include "CPFTools.h"
#include "CPFProjection.h"

#define M_PI 3.14159265359

namespace nsCPFTools {

	int angle_bins = 12;

}

using namespace texpert;
using namespace nsCPFTools;

// static
float CPFTools::AngleBetween(const Eigen::Vector3f& a, const Eigen::Vector3f& b)
{
	Eigen::Vector3f a_norm = a.normalized();
	Eigen::Vector3f b_norm = b.normalized();
	Eigen::Vector3f c = a_norm.cross(b_norm);
	return atan2f(c.norm(), a_norm.dot(b_norm));
}


//static 
Eigen::Affine3f CPFTools::GetRefFrame(const Eigen::Vector3f& p, const Eigen::Vector3f& n)
{

	Eigen::Vector3f axis = n.cross(Eigen::Vector3f::UnitX());

	if (axis(0) == 0.0f && axis(1) == 0.0f && axis(2) == 0.0f) {
		axis = Eigen::Vector3f::UnitX();
	}
	else {
		axis.normalize();
	}

	// create an angle axis transformation that rotates A degree around the axis.
	Eigen::AngleAxisf rot(AngleBetween(n, Eigen::Vector3f::UnitX()), axis);

	Eigen::Affine3f m = rot * Eigen::Translation3f(-p);
	Eigen::Affine3f T = Eigen::Affine3f(rot * Eigen::Translation3f(-p));

	Eigen::Vector3f p2 = T * p;

	// Multiply the point with the transformatiom matrix. 
	// -p moves the point into the origin of its own new coordinate frame.
	// The result is a 4x4 matrix - and x-axis aligned coordinate frame for the 
	// point p. 
	return Eigen::Affine3f(rot * Eigen::Translation3f(-p));
}




uint32_t CPFTools::DiscretizeCurvature(const Eigen::Vector3f& p1, const Eigen::Vector3f& n1, const PointCloud& pc, const Matches& matches, const float range ) 
{

	int count = 0;
	float angle_global = 0;
	for (int i = 0; i < 21; i++) {
		if( matches.matches[i].distance > 0.0){
			int id = matches.matches[i].second;
			angle_global += AngleBetween(n1, pc.normals[id]) * range;
			count++;
			//if(count == 10)break;
		}
			
	}

	return static_cast<uint32_t>(angle_global / count);
}


/*
Test process to get the discretized curvatures for the point set.
*/
static uint32_t DiscretizeCurvatureDev(const Eigen::Vector3f& p1, const Eigen::Vector3f& n1, const PointCloud& pc, const MyMatches& matches, const float range)
{


}


float max_ang_value = 0.0;
float min_ang_value = 10000000.0;

//static 
CPFDiscreet CPFTools::DiscretizeCPF(const std::uint32_t& c0, const std::uint32_t& c1, const Eigen::Vector3f& p0, const Eigen::Vector3f& p1)
{
	CPFDiscreet cpf;
	Eigen::Vector3f p01;
	//float ang =  p0.normalized().dot(p1.normalized()); // [-1,1]
	float ang = AngleBetween(p0,p1);

	float ang_deg =  (ang +M_PI);
	cpf[0] = c0;
	cpf[1] = c1;
	cpf[2] = ((ang +1) * angle_bins/2.0 );
	cpf[3] = 0;//c0-c1;

	if(ang > max_ang_value)
		max_ang_value = ang;
	if(ang < min_ang_value)
		min_ang_value = ang;

	return cpf;
}


//static 
void CPFTools::GetMaxMinAng(float& max, float& min)
{
	max = max_ang_value;
	min = min_ang_value;
}

//static 
void CPFTools::Reset(void)
{
	max_ang_value = 0.0;
	min_ang_value = 10000000.0;
}


//static 
void CPFTools::SetParam(CPFParam& param)
{
	angle_bins = param.angle_bins;
}



/*
Test process to get the discretized curvatures for the point set.
This is a naive implementation to verify the "optimized" implementation
*/
//static 
uint32_t CPFTools::DiscretizeCurvatureNaive(const Eigen::Vector3f& p1, const Eigen::Vector3f& n1, const PointCloud& pc, const MyMatches& matches, const float range)
{

	int count = 0;
	float angle_global = 0;

	Eigen::Matrix3f tpt, transform;
	CPFProjection::GetTangentPlaneTransform(n1, tpt, transform);

	Eigen::Affine3f frame =  CPFTools::GetRefFrame( p1, n1);
	Eigen::Matrix3f rot = frame.rotation();
	
	
	float cx = 0.0f;
	float cy = 0.0f;

	for (int i = 0; i < 21; i++) {
		if (matches.matches[i].distance > 0.0) {
			int id = matches.matches[i].second;
			float d = sqrt(matches.matches[i].distance) *  100000.0f;
			//angle_global += AngleBetween(n1, pc.normals[id]) * range;

			Eigen::Vector3f proj = CPFProjection::ProjectVectorToPlane(pc.normals[id], tpt);

			Eigen::Vector3f projection_local = transform * proj;

			Eigen::Vector3f xproj = CPFProjection::ProjectVectorToVector(proj, Eigen::Vector3f(rot(0), rot(1), rot(2)));
			Eigen::Vector3f yproj = CPFProjection::ProjectVectorToVector(proj, Eigen::Vector3f(rot(3), rot(4), rot(5)));

			cx += (xproj.norm());
			cy += (yproj.norm() );  // * 100.0f to get from m to mm

			count++;
			//if(count == 10)break;
		}

	}
	float c = std::max(cx, cy) * range;
	 
	uint32_t cc = static_cast<uint32_t>(c / count);

	return cc;
}