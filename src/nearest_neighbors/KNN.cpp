#include "KNN.h"

using namespace  texpert;

KNN::KNN() {

	_kdtree = new Cuda_KdTree();
	_ready = false;

	_refPoint = NULL;
	_testPoint = NULL;

}
KNN::~KNN(){
	delete _kdtree;
}


/*
Set the reference point cloud. 
This one goes into the kd-tree as soon as it is set. 
@param pc - reference to the point cloud model
*/
bool KNN::populate(PointCloud& pc) {

	assert(_kdtree);

	_rpoints.clear();

	vector<Eigen::Vector3f> p = pc.points;
	size_t size = pc.points.size();

	for (int i = 0; i < size; i++) {
		Cuda_Point k = Cuda_Point(p[i].x(), p[i].y(), p[i].z());
		k._id = i;
		_rpoints.push_back(k);
	}

	if (_rpoints.size() == 0) return false;
	_kdtree->initialize(_rpoints);
	_refPoint = &pc;

	// check if ready
	_ready = ready();

	return true;
}

/*
Set the test model, this is tested agains the 
reference model in the kd-tree
@param pc - reference to the point cloud model
*/
//bool KNN::setTestModel(PointCloud& pc)
//{
//	_tpoints.clear();
//
//	vector<Eigen::Vector3f>& p = pc.points ;
//
//	for (int i = 0; i < pc.points.size(); i++) {
//		Cuda_Point k = Cuda_Point(p[i].x(), p[i].y(), p[i].z());
//		k._id = i;
//		_tpoints.push_back(k);
//	}
//
//	_testPoint = &pc;
//
//	// check if ready
//	_ready = ready();
//
//	return true;
//}


/*
Start the knn search and return matches.
@param k - the number of matches to return
@param matches - reference to the matches
*/
int KNN::knn(PointCloud& pc, int k,  vector<Matches>& matches)
{
	// copy all models into the cuda structure. 

	_tpoints.clear();

	vector<Eigen::Vector3f>& p = pc.points ;

	for (int i = 0; i < pc.points.size(); i++) {
		Cuda_Point k = Cuda_Point(p[i].x(), p[i].y(), p[i].z());
		k._id = i;
		_tpoints.push_back(k);
	}

	_testPoint = &pc;

	// check if ready
	_ready = ready();


	assert(_kdtree);

	if (!_ready) return -1;
	if (_tpoints.size() == 0) return -1;

	_kdtree->knn(_tpoints, matches, 1);

	return 1;
}


/*
Check if this class is ready to run.
The kd-tree and the test points - both need to have points
@return - true, if it can run. 
*/
bool KNN::ready(void)
{
	if (_rpoints.size() > 0 && _tpoints.size() > 0)
		return true;

	return false;
}



/*
Reset the tree
*/
int KNN::reset(void) {

	assert(_kdtree);
	_kdtree->resetDevTree();

	return 1;
}
	