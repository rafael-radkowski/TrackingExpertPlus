#pragma once


// stl
#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include <chrono>


// local
#include "Types.h"				// point cloud types
#include "FDMatching.h"			// feature descriptor matching
#include "Utils.h"		
#include "LogReaderWriter.h"
#include "ICP.h"		// ICP


namespace texpert{

typedef enum {
	CPD,    // curvature pair features
	PPF		// point pair features - for performance comparison only
}DescriptorType;


/*
Registration parameters
*/
typedef struct PCRegParams{

	// cpf


	// ppf
	float	ppf_angle_step;
	float	ppf_distance_step;
	float	ppf_cluster_distance_th; 
	float	ppf_cluster_angle_th;

	// icp
	float	icp_min_error;

	PCRegParams()
	{
		ppf_angle_step = 3.0;
		ppf_distance_step = 0.3;
		ppf_cluster_distance_th = 0.8;
		ppf_cluster_angle_th = 12.0;
	}



}PCRegParams;  



class PCRegistratation
{
public:
	
	PCRegistratation();
	~PCRegistratation();


	/*
	Add a set of reference objects to the registration process. 
	The function will immediatelly extract the descriptors from the point cloud. 
	So this function is blocking until all parameters have been extracted. 
	@param reference_point_cloud -  a point cloud of the reference object. 
 		So previous sets can be released. 
	@return - true, if the process was exectued successfully. 
	*/
	bool addReferenceObject(PointCloud& reference_point_cloud);


	/*
	Process the current camera frame and match all reference objects with
	possible counterparts in the camera point cloud. 
	The function will immediately start the process. 
	@param camera_point_cloud - the location of the current point cloud to process. 
	@return true, if all steps were exectued successfully. 
	*/
	bool process(PointCloud& camera_point_cloud);


	/*!
	Return the poses. One pose per object, index aligned. 
	The pose matrix is a identity matrix if no pose could be found.
	@return - vector with Pose objects. 
	*/
	std::vector<Pose>	getPoses(void);


	/*!
	Return the pose as an 4x4 matrix for OpenGL. 
	@return - vector with 4x4 matrices containing the object pose
	*/
	std::vector<glm::mat4> getGlPoses(void);



	/*
	The class implements multiple descriptor types
	Set the descriptor type for registration. 
	@param type -  a enum value of type DescriptorType, CPF or PPF
	*/
	void setProcessingType(DescriptorType type);


	/*
	Set the registration parameters.
	@param params - a struct of type PCRegParams containing registration parameters
	*/
	void setRegistrationParams(PCRegParams params);



private:

	// vector to maintain the reference point clouds
	std::vector<PointCloud>			_reference_point_clouds;


	// The one and only environment point cloud
	PointCloud						_camera_point_cloud;


	//------------------------------------------------------------------------
	// matching
	FDMatching*						_fm;

	// icp refinement
	ICP*							_icp;

	// all parameters for the registration process
	PCRegParams						_params;

	// the object poses. 
	std::vector<Pose>				_poses;
	std::vector<glm::mat4>			_gl_poses;
	std::vector<float>				_rms_error;
};


} //namespace texpert{