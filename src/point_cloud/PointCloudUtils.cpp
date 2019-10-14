#include "PointCloudUtils.h"

using namespace texpert;


/*
Move the point cloud points along a translation vector
@param pc_src -  pointer to the source point cloud of type PointCloud
@param pc_dst - pointer to the destination point cloud of type PointCloud
@param translation - vec 3 with the translation in x, y, z/
@return  true - if successful. 
*/
//static 
Eigen::Vector3f PointCloudUtils::CalcCentroid(PointCloud* pc_src)
{
	if (pc_src == NULL)return Eigen::Vector3f (0, 0, 0);

	
	size_t size = pc_src->size();
	Eigen::Vector3f sum(0, 0, 0);
	for_each(pc_src->points.begin(), pc_src->points.end(), [&](Eigen::Vector3f t) {
		sum += t;
	});

	/*
	
	vector<Eigen::Vector3f>::iterator itr = pc_src->points.begin();

	Eigen::Vector3f sum(0, 0, 0);

	while (itr != pc_src->points.end()) {
		sum += (*itr);
		itr++;
	}
	*/
	sum /= size;

	return sum;
}


