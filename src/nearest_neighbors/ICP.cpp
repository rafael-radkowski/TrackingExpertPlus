#include "ICP.h"



using namespace  texpert;

ICP::ICP() {

	_max_error = 0.001;
	_max_iterations = 40;

	_knn = new KNN();

	_outlier_reject.setMaxNormalVectorAngle(25.0);
	_outlier_reject.setMaxThreshold(1.5);

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
	
	if (pc.size() <= 0) {
		std::cout << "[ERROR] ICP: no camera points given. " << std::endl;
		return false;
	}

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
//#define ICPTRANSTEST
#ifdef ICPTRANSTEST
// A test to test the transformations only assuming the nearest neighbors are perfet. 
// note that the two point clouds must be equal for this test. 
	return test_transformation(pc, initial_pose,  result_pose,  rms);
#endif
	
	// remember the test points
	_testPoints = pc;

	if (!ready()) {
		return false;
	}

	Matrix4f overall = Matrix4f::Identity();
	Matrix4f result = Matrix4f::Identity();

	rms = 100000000.0;


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
	// matching_points are camera points that were selected as nearest neighbosr
	// accepted_points contains the model points that survived the outlier test. 
	std::vector<Eigen::Vector3f> matching_points, accepted_points;
	matching_points.reserve(_testPointsProcessing.size());
	accepted_points.reserve(_testPointsProcessing.size());

	if(_verbose && _verbose_level == 2)
			cout << "\n[ICP] - Start to register." << endl;

	// the loop expects that both vectors are already index aligned. 
	int itr = 0;
	// search for nearest neighbors
	std::vector<Matches> local_matches;
	for (int i = 0; i < _max_iterations; i++)
	{

		local_matches.clear();
		local_matches.reserve(_testPointsProcessing.size());

		// find nearest neighbors
		_knn->knn(_testPointsProcessing, 1, local_matches);

		matching_points.clear();
		accepted_points.clear();

		// Reject nearest neighbors that are most likely outliers. 
		// get a vector with all the matching points. 
		for_each(local_matches.begin(), local_matches.end(), [&](Matches m )
		{
			if( _outlier_reject.testDistanceAngle(_testPointsProcessing.points[m.matches[0].first ], _cameraPoints.points[m.matches[0].second],
												  _testPointsProcessing.normals[m.matches[0].first ], _cameraPoints.normals[m.matches[0].second]))
			{
				 matching_points.push_back(_cameraPoints.points[m.matches[0].second]);
				 accepted_points.push_back(_testPointsProcessing.points[m.matches[0].first ] );
			}
		});

		// Check if sufficient points are available to register the points
		if (matching_points.size() < 12) {
			cout << "insufficient points" << endl;
			return false;
		}

		//  Calculate the rotation delta
		Matrix3f R = ICPTransform::CalcRotationArun(accepted_points, matching_points);
		Vector3f t = ICPTransform::CalculateTranslation(accepted_points, matching_points);
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

		// update the point transformation
		/// TODO: performance teste. Which function is faster for_each vs. std::transform
		// p' = (R * p) + t;
		for_each(accepted_points.begin(), accepted_points.end(), [&](Vector3f& p){p = (R * p) + t;});

		// transform the original points and normal vectors
		for_each(_testPointsProcessing.points.begin(), _testPointsProcessing.points.end(), [&](Vector3f& p){p = (R * p) + t;});
		for_each(_testPointsProcessing.normals.begin(), _testPointsProcessing.normals.end(), [&](Vector3f& p){p = (R * p);});
	
		overall =  result * overall;

		itr++;
		rms = ICPTransform::CheckRMS(accepted_points,  matching_points);

		if(_verbose && _verbose_level == 2)
			cout << "[ICP] - RMS: " << rms << " at " << itr << "." << endl;

		if (rms < _max_error) break;

		
	}
	
	result_pose = initial_pose.t.matrix() * overall;

	if(_verbose && _verbose_level == 2)
		MatrixUtils::PrintMatrix4f(result_pose);

	if(_verbose)
		cout << "[INFO] - ICP terminated with RMS: " << rms << " with " << itr << " interations." << endl;

	return true;



}



bool ICP::test_transformation(PointCloud& pc, Pose initial_pose, Eigen::Matrix4f& result_pose, float& rms)
{
	// remember the test points
	_testPoints = pc;

	if (!ready()) {
		return false;
	}

	Matrix4f overall = Matrix4f::Identity();
	Matrix4f result = Matrix4f::Identity();

	rms = 100000000.0;


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

	if(_verbose && _verbose_level == 2)
			cout << "\n[ICP] - Start to register." << endl;

	// the loop expects that both vectors are already index aligned. 
	int itr = 0;
	// search for nearest neighbors
	std::vector<Matches> local_matches;
	for (int i = 0; i < _max_iterations; i++)
	{

		

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

		// update the point transformation
		/// TODO: performance teste. Which function is faster for_each vs. std::transform
		// p' = (R * p) + t;
		// transform the original points and normal vectors
		for_each(_testPointsProcessing.points.begin(), _testPointsProcessing.points.end(), [&](Vector3f& p){p = (R * p) + t;});
		for_each(_testPointsProcessing.normals.begin(), _testPointsProcessing.normals.end(), [&](Vector3f& p){p = (R * p);});
	
		overall =  result * overall;

		itr++;
		rms = ICPTransform::CheckRMS(_testPointsProcessing.points,  _cameraPoints.points);

		if(_verbose && _verbose_level == 2)
			cout << "[ICP] - RMS: " << rms << " at " << itr << "." << endl;

		if (rms < _max_error) break;

		
	}
	
	result_pose = initial_pose.t.matrix() * overall;

	if(_verbose && _verbose_level == 2)
		MatrixUtils::PrintMatrix4f(result_pose);

	if(_verbose)
		cout << "[INFO] - ICP terminated with RMS: " << rms << " with " << itr << " interations." << endl;

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
	if (_testPoints.size() <= 0 || _cameraPoints.size() <= 0) {
		std::cout << "[ERROR] ICP: no points given. " << std::endl;
		return false;
	}
	return true;
}


/*
Set a minimum error as termination criteria.
@param error - a float value > 0.0
*/
void ICP::setMinError(float error)
{
	_max_error  =  std::max(0.0f, error);
}


/*
Set the number of maximum iterations. 
@param max_iterations - integer number with the max. number of iterations.
	The number must be in the range [1, 1000] 
*/
void ICP::setMaxIterations(int max_iterations)
{
	_max_iterations = std::min(1000, std::max(1, max_iterations));
}
