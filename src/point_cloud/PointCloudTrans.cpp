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

//static
void PointCloudTrans::TransformInPlace(Eigen::Affine3f& Rt, Eigen::Vector3f& point)
{
	Eigen::Vector4f pt = Eigen::Vector4f(point.x(), point.y(), point.z(), 1);
	Eigen::Matrix4f mat;
	mat_conv->Affine3f2Matrix4f(Rt, mat);

	pt = mat * pt;
	point = Eigen::Vector3f(pt.x(), pt.y(), pt.z());
}

//static
void PointCloudTrans::TransformInPlace(Eigen::Affine3f& Rt, vector<Eigen::Vector3f>& points)
{
	for (int i = 0; i < points.size(); i++)
	{
		TransformInPlace(Rt, points.at(i));
	}
}

//static
void PointCloudTrans::getTransformFromPosition(Eigen::Vector3f& accum_t, Eigen::Matrix3f& accum_R, Eigen::Vector3f& centroid, Eigen::Affine3f& init_affine, Eigen::Matrix4f& result)
{
	result = Eigen::Matrix4f::Identity();

	Eigen::Affine3f transform(Eigen::Translation3f(-centroid.x(), -centroid.y(), -centroid.z()));
	Eigen::Matrix4f centInv = transform.matrix();

	Eigen::Affine3f transform2(Eigen::Translation3f(centroid.x(), centroid.y(), centroid.z()));
	Eigen::Matrix4f cent = transform2.matrix();

	Eigen::Affine3f transform3(Eigen::Translation3f(accum_t.x(), accum_t.y(), accum_t.z()));
	Eigen::Matrix4f t2 = transform3.matrix();

	Eigen::Matrix4f R2 = Eigen::Matrix4f::Identity();
	R2.block<3, 3>(0, 0) = accum_R;

	Eigen::Affine3f m;
	m = Eigen::Translation3f(init_affine.translation());
	Eigen::Matrix4f ti;
	ti = m.matrix();

	Eigen::Matrix4f Ri;
	Ri = Eigen::Matrix4f::Identity();
	Ri.block(0, 0, 3, 3) = init_affine.rotation().matrix();

	//result = t2 * cent * R2 * centInv;

	result = Ri.transpose() * (t2 * cent * R2 * centInv) * ti.transpose();
}