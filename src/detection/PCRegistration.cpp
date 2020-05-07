#include "PCRegistration.h"



using namespace texpert;

PCRegistratation::PCRegistratation()
{

	// init matching and icp
	_fm = new FDMatching();

	// init feature matching
	setRegistrationParams(_params);

	// inverts the pose so that the object is given with respect to the camera
	// coordinate frame. 
	_fm->invertPose(false);
	_fm->setVerbose(true);


	// Initiate ICP
	_icp = new ICP();
	_icp->setVerbose(false);
}

PCRegistratation::~PCRegistratation()
{
	delete _fm;
	delete _icp;

}


/*
Add a set of reference objects to the registration process. 
The function will immediatelly extract the descriptors from the point cloud. 
So this function is blocking until all parameters have been extracted. 
@param reference_point_cloud -  a point cloud of the reference object. 
	Note that the function creates a local copy of the reference point cloud. 
	So previous sets can be released. 
@return - true, if the process was exectued successfully. 
*/
bool PCRegistratation::addReferenceObject(PointCloud& reference_point_cloud)
{
	// store the reference point cloud
	_reference_point_clouds.push_back(reference_point_cloud);
	_rms_error.push_back(100000.0);

	// extract the feature map for this point cloud. 
	bool ret = _fm->extract_feature_map(&reference_point_cloud.points, &reference_point_cloud.normals);

	return ret;
}


/*
Process the current camera frame and match all reference objects with
possible counterparts in the camera point cloud. 
The function will immediately start the process. 
@param camera_point_cloud - the location of the current point cloud to process. 
@return true, if all steps were exectued successfully. 
*/
bool  PCRegistratation::process(PointCloud& camera_point_cloud)
{
	bool ret = true;

	// clear all poses
	_poses.clear();
	_gl_poses.clear();

	// no data to match
	if(_reference_point_clouds.size() <= 0) return false;

	if(_rms_error[0] > 3.0){

		// detect the object
		ret = _fm->searchIn(&camera_point_cloud.points, &camera_point_cloud.normals, _poses);

		cout << "[INFO] - Acquired new global pose " << endl;
	
		//--------- debug -- /
	}else{
		Pose p;
		p.t = Matrix4f::Identity();
		_poses.push_back(p);
	}

	// -----------------------------------------
	// Refine the pose for each object
	int index = 0;
	for(auto p : _poses){
		// final result
		Eigen::Matrix4f mat;

		// Set the camera data and run ICP with the initial pose
		_icp->setCameraData( camera_point_cloud);
		 bool has_pose = _icp->compute(_reference_point_clouds[index] , p, mat, _rms_error[index]);

		 if(has_pose){
			 // get the pose and transform it into a opengl pose
			_poses[index].t = mat;
			glm::mat4 m = MatrixUtils::Affine3f2Mat4(_poses[index].t);
			_gl_poses.push_back(m);
		}

		//MatrixUtils::PrintGlm4(m);
		index++;
	}


	return ret;
}


/*!
Return the poses. One pose per object, index aligned. 
The pose matrix is a identity matrix if no pose could be found.
@return - vector with Pose objects. 
*/
std::vector<Pose> PCRegistratation::getPoses(void)
{
	return _poses;
}


/*!
Return the pose as an 4x4 matrix for OpenGL. 
@return - vector with 4x4 matrices containing the object pose
*/
std::vector<glm::mat4> PCRegistratation::getGlPoses(void)
{
	return _gl_poses;
}

/*
The class implements multiple descriptor types
Set the descriptor type for registration. 
@param type -  a enum value of type DescriptorType, CPF or PPF
*/
void  PCRegistratation::setProcessingType(DescriptorType type)
{

	
}


/*
Set the registration parameters.
@param params - a struct of type PCRegParams containing registration parameters
*/
void PCRegistratation::setRegistrationParams(PCRegParams params)
{
	_params = params;
	
	_fm->setAngleStep(_params.ppf_angle_step);
	_fm->setDistanceStep(_params.ppf_distance_step);
	_fm->setClusteringThreshold(_params.ppf_cluster_distance_th,
								_params.ppf_cluster_angle_th);
}