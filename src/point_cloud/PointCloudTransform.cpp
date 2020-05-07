#include "PointCloudTransform.h"


using namespace texpert;


const float deg2rad = 3.141592653589793238463 / 180.0;

/*
Move the point cloud points along a translation vector
@param pc_src -  pointer to the source point cloud of type PointCloud
@param translation - vec 3 with the translation in x, y, z/
@return  true - if successful. 
*/
//static 
bool PointCloudTransform::Translate(PointCloud* pc_src,  Eigen::Vector3f  translation)
{
	assert(pc_src != NULL );

	size_t size = pc_src->size();

	vector<Eigen::Vector3f>::iterator itr = pc_src->points.begin();
	while (itr != pc_src->points.end()) {
		(*itr) = (*itr) + translation;
		itr++;
	}
	
	//for_each(pc_src->points.begin(), pc_src->points.end(), [&](Eigen::Vector3f& t) {
	//	t = t + translation;
	//}
	
	return true;
}


/*
Rotate the point cloud points around their origin
@param pc_src -  pointer to the source point cloud of type PointCloud
@param translation - vec 3 with the Euler angles for a rotation arond x, y, z.
@return  true - if successful. 
*/
//static 
bool PointCloudTransform::Rotate(PointCloud* pc_src, Eigen::Vector3f euler_angles)
{
	assert(pc_src != NULL );

	// get centroid
	Eigen::Vector3f centroid = PointCloudUtils::CalcCentroid(pc_src);

	// Move all points to the centroid
	Translate(pc_src, -centroid);

	// assemble rotation matrices;
	Eigen::Affine3f m;
	m = Eigen::Affine3f(Eigen::AngleAxisf(deg2rad * euler_angles.x(), Eigen::Vector3f::UnitX()) )*
    Eigen::Affine3f(Eigen::AngleAxisf(deg2rad *euler_angles.y(), Eigen::Vector3f::UnitY())) *
    Eigen::Affine3f(Eigen::AngleAxisf(deg2rad *euler_angles.z(), Eigen::Vector3f::UnitZ()));

	Eigen::Affine3f m_rot = m;
	m_rot.data()[3] = 0;
	m_rot.data()[7] = 0;
	m_rot.data()[11] = 0;
	m_rot.data()[15] = 0;

	// rotate all points
	int c = 0;
	size_t size = pc_src->size();
	vector<Eigen::Vector3f>::iterator itr = pc_src->points.begin();
	while (itr != pc_src->points.end()) {
		(*itr) = m * (*itr);
		pc_src->normals[c] = m_rot * pc_src->normals[c];
		itr++;
		c++;
	}


	// Move all points back
	Translate(pc_src, centroid);

	return true;
}


/*
Transforms the point cloud points 
@param pc_src -  pointer to the source point cloud of type PointCloud
@param translation - vec 3 with the translation in x, y, z
@param rotation - vec 3 with the Euler angles for a rotation arond x, y, z.
@param around_centroid - moves the entire point cloud set to its centroid before rotating. 
			It rotates it in place otherwise. 
@return  true - if successful. 
*/
//static 
bool PointCloudTransform::Transform(PointCloud* pc_src, Eigen::Vector3f translation, Eigen::Vector3f  euler_angles, bool around_centroid)
{
	assert(pc_src != NULL );

	// get centroid
	Eigen::Vector3f centroid = PointCloudUtils::CalcCentroid(pc_src);

	if(!around_centroid) centroid = Eigen::Vector3f(0,0,0);

	// Move all points to the centroid
	Translate(pc_src, -centroid);

	// assemble rotation matrices;
	Eigen::Affine3f m;
	m = Eigen::Affine3f(Eigen::AngleAxisf(deg2rad * euler_angles.x(), Eigen::Vector3f::UnitX()) )*
    Eigen::Affine3f(Eigen::AngleAxisf(deg2rad *euler_angles.y(), Eigen::Vector3f::UnitY())) *
    Eigen::Affine3f(Eigen::AngleAxisf(deg2rad *euler_angles.z(), Eigen::Vector3f::UnitZ()));

	Eigen::Affine3f m_rot = m;
	m_rot.data()[3] = 0;
	m_rot.data()[7] = 0;
	m_rot.data()[11] = 0;
	m_rot.data()[15] = 0;

	// rotate all points
	int c = 0;
	size_t size = pc_src->size();
	vector<Eigen::Vector3f>::iterator itr = pc_src->points.begin();
	while (itr != pc_src->points.end()) {
		(*itr) = m * (*itr);
		pc_src->normals[c] = m_rot * pc_src->normals[c];
		itr++;
		c++;
	}


	// Move all points back
	Translate(pc_src, centroid + translation);

	return true;
}