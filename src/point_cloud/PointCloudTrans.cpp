#include "PointCloudTrans.h"

namespace ns_PointCloudTrans
{
	MatrixConv* mat_conv = MatrixConv::getInstance();
}

using namespace ns_PointCloudTrans;

//static
Eigen::Vector3d PointCloudTrans::Transform(Eigen::Affine3f& Rt, Eigen::Vector3d& point)
{
	Eigen::Vector4f pt = Eigen::Vector4f(point.x(), point.y(), point.z(), 1);
	Eigen::Matrix4f mat;
	mat_conv->Affine3f2Matrix4f(Rt, mat);

	pt = mat * pt;
	return Eigen::Vector3d(pt.x(), pt.y(), pt.z());
}

//static
vector<Eigen::Vector3d> PointCloudTrans::Transform(Eigen::Affine3f& Rt, vector<Eigen::Vector3d>& points)
{
	vector<Eigen::Vector3d> results = vector<Eigen::Vector3d>();

	for (int i = 0; i < points.size(); i++)
	{
		results.push_back(Transform(Rt, points.at(i)));
	}

	return results;
}

//static
void PointCloudTrans::TransformInPlace(Eigen::Affine3f& Rt, Eigen::Vector3d& point)
{
	Eigen::Vector4f pt = Eigen::Vector4f(point.x(), point.y(), point.z(), 1);
	Eigen::Matrix4f mat;
	mat_conv->Affine3f2Matrix4f(Rt, mat);

	pt = mat * pt;
	point = Eigen::Vector3d(pt.x(), pt.y(), pt.z());
}

//static
void PointCloudTrans::TransformInPlace(Eigen::Affine3f& Rt, vector<Eigen::Vector3d>& points)
{
	for (int i = 0; i < points.size(); i++)
	{
		TransformInPlace(Rt, points.at(i));
	}
}