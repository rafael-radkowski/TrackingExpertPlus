#include "ICP.h"



using namespace  texpert;

ICP::ICP() {

	_max_error = 0.001;
	_max_iterations = 40;

	_knn = new KNN();

}

ICP::~ICP(){

	delete _knn;
}


/*
Set the reference point cloud. 
This one goes into the kd-tree as soon as it is set. 
@param pc - reference to the point cloud model
*/
bool ICP::setCameraData(PointCloud& pc) {

	_cameraPoints = pc;

	_knn->reset();
	_knn->populate(pc);
	return true;
}

/*
Set the test model, this is tested agains the 
reference model in the kd-tree
@param pc - reference to the point cloud model
*/
bool  ICP::compute(PointCloud& pc, Pose initial_pose, Eigen::Matrix4f& result_pose, float& rms)
{
	Matrix4f overall = Matrix4f::Identity();
	Matrix4f result = Matrix4f::Identity();

	rms = 100000000.0;

	// remember the test points
	_testPoints = pc;

	// copy test points into a new struct
	vector<Vector3f> in0, nn0;
	std::copy(_testPoints.points.begin(), _testPoints.points.end(), back_inserter(in0));
	std::copy(_testPoints.normals.begin(), _testPoints.normals.end(), back_inserter(nn0));

	// Apply the initial rotation to all copied test points
	Matrix3f R = initial_pose.t.rotation();
	Vector3f t = initial_pose.t.translation();
	for_each(in0.begin(), in0.end(), [&](Vector3f& p){p = (R * p) + t;});
	for_each(nn0.begin(), nn0.end(), [&](Vector3f& n){n = (R * n);});

	// prepare for knn
	_testPointsProcessing.points = in0;
	_testPointsProcessing.normals = nn0;
	_testPointsProcessing.size();

	// reserve memory for all aligning points. 
	std::vector<Eigen::Vector3f> matching_points;
	matching_points.resize(_testPointsProcessing.size());

	// the loop expects that both vectors are already index aligned. 
	int itr = 0;
	for (int i = 0; i < _max_iterations; i++)
	{

		// search for nearest neighbors
		std::vector<Matches> local_matches;
		_knn->knn(_testPointsProcessing, 1, local_matches);

		// get a vector with all the matching points. 
		for_each(local_matches.begin(), local_matches.end(), [&](Matches m )
		{
			 matching_points[ m.matches[0].first ] = _cameraPoints.points[m.matches[0].second];
		});

		// Check if sufficient points are available to register the points
		if (matching_points.size() < 12) {
			return false;
		}

		//  Calculate the rotation delta
		Matrix3f R = ICPTransform::CalcRotationArun(_testPointsProcessing.points, _cameraPoints.points);
		Vector3f t = ICPTransform::CalculateTranslation(_testPointsProcessing.points, _cameraPoints.points);
		Matrix3f R_inv = R;// Matrix3f::Identity();

		// maintaining row-major
		result(0) = R_inv(0);
		result(1) = R_inv(1);
		result(2) = R_inv(2);

		result(4) = R_inv(3);
		result(5) = R_inv(4);
		result(6) = R_inv(5);

		result(8) = R_inv(6);
		result(9) = R_inv(7);
		result(10) = R_inv(8);

		result(12) = t.x();
		result(13) = t.y();
		result(14) = t.z();

		// update the points
		/// TODO: performance teste. Which function is faster for_each vs. std::transform
		// p' = (R * p) + t;
		for_each(_testPointsProcessing.points.begin(), _testPointsProcessing.points.end(), [&](Vector3f& p){p = (R * p) + t;});
	
		overall =  result * overall;

		itr++;
		rms = ICPTransform::CheckRMS(_testPointsProcessing.points,  matching_points);

		if(_verbose && _verbose == 2)
			cout << "[ICP] - RMS: " << rms << " at " << itr << "." << endl;

		if (rms < _max_error) break;

		
	}
	
	result_pose = initial_pose.t.matrix() * overall;

	if(_verbose && _verbose == 2)
		MatrixUtils::PrintMatrix4f(result_pose);

	if(_verbose)
		cout << "[INFO] -ICP RMS: " << rms << " with " << itr << " interations." << endl;

	return true;



}



/*
Set the amount of output plotted to the dialot. 
@param verbose - true enables outputs. 
@param int - level 1 or 2 sets the amount of outputs.
		1 gives just basis data
		2 prints data per ICP iterations
*/
bool ICP::setVerbose(bool verbose, int verbose_level)
{
	_verbose = verbose;
	_verbose_level =  std::max(1 ,std::min(2, verbose_level));
	return true;
}



/*
Check if this class is ready to run.
The kd-tree and the test points - both need to have points
@return - true, if it can run. 
*/
bool ICP::ready(void)
{
	

	return false;
}



