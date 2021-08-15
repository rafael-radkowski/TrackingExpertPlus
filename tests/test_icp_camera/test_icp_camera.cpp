/*
@file test_icp_camera.cpp

This file implements an ICP test to determine whether or not the ICP algorithm for 
TrackingExpert+ is running correctly. It works with a set of point cloud recorded with an 
Azure Kinect camera. The point set contains a Stanford bunny model. ICP registers
a reference model with this bunny. 

The scene contains the following point sets:
Red points: camera point cloud
Green point set: reference point set at its original position.
Yellow point set: evaluation point set at the ICP pose after ICP terminates. 

The file runs one test set automatically. 

Manual operations:
- To change the camera point set, press '='. The code will load the next point set in the list. 
- To start ICP for the current point set, press the key 'a'. This will run all ICP iterations automatically. 
- To start a manual ICP run for the current point set, press 's'. To step to the next iteration, press 'space'

Graphics keyboard layout:
- n - enable or disable normal vector rendering
- r - enable or disable the reference model rendering.
- c - show the nearest neighbors between both point clouds. 

Rafael Radkowski
Iowa State University
rafael@iastate.edu
January 2020
MIT License
-----------------------------------------------------------------------------------------------------------------------------
Last edited:



*/

// STL
#include <iostream>
#include <string>
#include <Windows.h>
#include <fstream>


// GLM include files
#define GLM_FORCE_INLINE
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtx/transform.hpp>  // transformation
#include <glm/gtx/quaternion.hpp> // quaternions


// TrackingExpert
#include "trackingx.h"
#include "graphicsx.h"
#include "ReaderWriterOBJ.h"
#include "ReaderWriterPLY.h"
#include "ReaderWriterUtil.h"
#include "ReadFiles.h"
#include "ICP.h"  // the ICP class to test
//#include "MatrixTransform.h"
#include "GLLineRenderer.h"
#include "MatrixConv.h"
#include "KinectAzureCaptureDevice.h"

using namespace texpert;



// The OpenGL window
isu_ar::GLViewer* window;

// OpenGL point cloud objects showing the point cloud
isu_ar::GLPointCloudRenderer* gl_camera_point_cloud;
isu_ar::GLPointCloudRenderer* gl_reference_point_cloud;
isu_ar::GLPointCloudRenderer* gl_reference_eval; 

// a debug visualization that allows one to render lines between points. 
isu_ar::GLLineRenderer*			gl_knn_lines;

// helper variable to enable/disable the gl renderer. 
int							gl_normal_rendering = 0;
int							gl_ref_rendering = 1;
int							gl_nn_rendering = 0;

// Helper variables to set the point cloud sampling. 
SamplingParam		sampling_param;
SamplingMethod		sampling_method;


// The reference point cloud.
// The first one is the point cloud for all ICP purposes.
// The second one is the raw point cloud as loaded. 
PointCloud			pc_ref;
PointCloud			pc_ref_as_loaded;

// The test point cloud
// The first one is the point cloud for all ICP purposes.
// The second one is the raw point cloud as loaded. 
PointCloud			pc_camera;
PointCloud			pc_camera_as_loaded;

// THe evaluation point cloud. The ICP results are applied to this point set. 
// The first one is the point cloud for all ICP purposes.
// The second one is the raw point cloud as loaded. 
PointCloud			pc_eval;
PointCloud			pc_eval_as_loaded;


// variable to store the fiel. 
std::string			ref_file = "../data/stanford_bunny_pc.obj";
std::vector<std::string> files;

MatrixConv* conv = MatrixConv::getInstance();

PointCloudProducer* cam_cloud;
KinectAzureCaptureDevice* cam;
bool usingCam = false;


//--------------------------------------------------------
// ICP
texpert::ICP*		icp;


// icp variables to run the test. 
std::vector< Eigen::Vector3f>	initial_pos;
std::vector< Eigen::Vector3f>	initial_rot;
std::vector< float>				ground_truth_rms;
int								current_set = 0;
int								run_test = 0;


// for test results. 
Eigen::Matrix4f pose_result;
float rms = 1000.0;
float overall_rms = 0.0;
int N = 0;
string date = "";


int sequence_test_case = 0;


//--------------------------------------------------------------------
// Function prototypes.
void runTest(void);
void runTestManual(void);
void startTestManual(void);
float startAutoICP(void);
void loadNewObject(void);
void runSequenceTest(void);
void startAutoICP2(void);
void findModel(void);
/*
Keyboard callback for keyboard interaction. 
*/
void keyboard_callback( int key, int action) {


	cout << key << " : " << action << endl;
	switch (action) {
	case 0:  // key up
	
		switch (key) {
		case 44: //,
		{
			usingCam = !usingCam;
			icp->setVerbose(!usingCam, 0);
			break;
		}
		case 87: // w
		{
			runTest();
			break;
			} 
		case 78: // n
			{
				gl_normal_rendering = (++gl_normal_rendering)%2;
				gl_camera_point_cloud->enableNormalRendering((gl_normal_rendering==1)? true :false);
				gl_reference_point_cloud->enableNormalRendering((gl_normal_rendering==1)? true :false);
				gl_reference_eval->enableNormalRendering((gl_normal_rendering==1)? true :false);

			break;
			}

		case 82: // r
			{
				gl_ref_rendering = (++gl_ref_rendering)%2;
				gl_reference_point_cloud->enablePointRendering((gl_ref_rendering==1)? true :false);
		
			break;
			}
		case 67: // c
			{
				gl_nn_rendering = (++gl_nn_rendering)%2;
				gl_knn_lines->enableRenderer((gl_nn_rendering==1)? true :false);
		
			break;
			}
		case 69: // 4
			{
				gl_nn_rendering = (++gl_nn_rendering)%2;
				gl_knn_lines->enableRenderer((gl_nn_rendering==1)? true :false);
		
			break;
			}
		case 83: // s
			{
				startTestManual();
		
			break;
			}
		case 65: // a
			{
				startAutoICP();
		
			break;
			}
		case 32: // space
			{
				runTestManual();
				break;
			}
		case 79: // o
		{
			sequence_test_case = 0;
			break;
		}
		case 80: // p
		{
			sequence_test_case++;
			break;
		}
		case 91: // [
		{
			startAutoICP2();
			break;
		}
		case 61: // =
			{
				run_test++;
				if(run_test>4) run_test = 0;

				std::cout << "Current test: " << run_test << std::endl;
				loadNewObject();
				break;
			}
		}
		break;
		
		case 1: // key down

			break;
	}
}



/*
Function to start a manual ICP run. 
It set the start condition for the currently loaded model 
*/
void startTestManual(void)
{
	if(current_set >= initial_pos.size()) current_set = 0;

	std::cout << "[INFO] - START MANUAL ICP " << std::endl; ;

	// set the ICP max iteration to 1. The function along wiht runTestManual
	// allows a user to step through all ICP iterations manually.
	// This requires a max iteration of 1 to only run one and to terminate after 1. 


	//icp->total = Eigen::Matrix4f::Identity();

	icp->setMaxIterations(1);



	// update all model with the loaded version.
	current_set = run_test;
	pc_ref = pc_ref_as_loaded;
	pc_eval = pc_ref_as_loaded;


	// Move the point cloud to it initial position position. 

	PointCloudTransform::Transform(&pc_ref, pose_result );
	PointCloudTransform::Transform(&pc_eval, pose_result);

	pose_result = Eigen::Matrix4f::Identity();

	
	// reset the graphics model matrix. 

	gl_reference_eval->setModelmatrix(glm::mat4(1.0));



	if(current_set >= initial_pos.size()) current_set = 0;
}

/*
The function runs the ICP test manually. 
THe user can run through all ICP iterations. 
*/
void runTestManual(void)
{
	
	// Move the point cloud to the last pose 

	PointCloudTransform::Transform(&pc_ref, pose_result );
	PointCloudTransform::Transform(&pc_eval, pose_result );

	// Run ICP with the new poise

	Pose pose;
	pose.t =  Eigen::Matrix4f::Identity();
	icp->compute(pc_ref, pose, pose_result, rms);
	
	
	// Update teh graphic reference model
	glm::mat4 ref_mat;
	conv->Matrix4f2Mat4(icp->Rt(), ref_mat);
	gl_reference_eval->setModelmatrix(ref_mat);
	

	// Update the lines between the point clouds showing the nearest neighbors

	gl_knn_lines->updatePoints(pc_ref.points, pc_camera.points , icp->getNN());
	


	//std::cout << "[INFO] - glm x: " << m[3][0] << ", y: " << m[3][1] << ", z: " <<  m[3][2] << std::endl;
	//std::cout << "[INFO] - translation x: " << pose_result(12) << ", y: " << pose_result(13) << ", z: " <<  pose_result(14) << std::endl;
	

	// Output the pose

	std::cout << "[INFO] - Pose\n";
	std::cout << pose_result << std::endl;

	
	
}

/*
Start an ICP auto run.
This function runs the complete ICP process until it terminates for the currently selected model. 

*/
float startAutoICP(void)
{ 

	std::cout << "[INFO] - START AUTO ICP " << std::endl; 

	// set the max iteration to 100. 

	icp->setMaxIterations(200);

	// Move the point cloud to its start position and orientation

	PointCloudTransform::Transform(&pc_ref, pose_result);
	pc_eval = pc_ref;

	Pose pose;
	pose.t = Eigen::Matrix4f::Identity();

	// run ICP
	icp->compute(pc_ref, pose, pose_result, rms);

	// Update the graphics model matrix and the lines between the nearest neighbors. 
	glm::mat4 ref_mat;
	conv->Matrix4f2Mat4(icp->Rt(), ref_mat);

	gl_reference_eval->setModelmatrix(ref_mat);
	gl_knn_lines->updatePoints(pc_ref.points, pc_camera.points , icp->getNN());

	//std::cout << "[INFO] - Pose delta\n";
	//std::cout << icp->Rt() << std::endl;

	return rms;
}



void runTest(void)
{
	
	//thread_1.lock();

	int error_count = 0;

	Sleep(100);



	while(run_test<5 ){

		// load a new object
		loadNewObject();

		// run ICP
		startAutoICP();

		run_test++;

		Sleep(300);
				
	}


	//std::cout << "\n[INFO] - All " <<  initial_pos.size() << " completed with a mean rms: " << overall_rms << " , " << error_count <<  " false results" << std::endl;
	//if(current_set >= initial_pos.size()) current_set = 0;

	// set the variable to the last value. 
	run_test = 4;

	//std::cout << "\n[INFO] - All " <<  initial_pos.size() << " completed with a mean rms: " << overall_rms << " , " << error_count <<  " false results" << std::endl;



	//thread_1.unlock();
}


/*
Load a new object from a file and update the ICP camera dataset. 
The function takes a file from the file list, loads the data, 
and updates the ICP algorithm with the new model. 
*/
void loadNewObject(void) {
	
	std:string camera_file = files[run_test];
	ReaderWriterUtil::Read(camera_file, pc_camera_as_loaded.points, pc_camera_as_loaded.normals, true, false);
	
	// sampling is disabled since the test pointclouds are already downsampled. 
	//Sampling::Run(pc_camera_as_loaded, pc_camera_as_loaded);
	pc_camera = pc_camera_as_loaded;

	// update the ICP camera dataset. 
	icp->setCameraData(pc_camera_as_loaded);
}

/*
	Use the camera point cloud to estimate model pose.
*/
void findModel(void) {
	cam_cloud->process();
	pc_camera = pc_camera_as_loaded;
	icp->setCameraData(pc_camera_as_loaded);

	//Low iteration count to make viewport framerate more bearable
	icp->setMaxIterations(20);

	// Move the point cloud to its start position and orientation

	PointCloudTransform::Transform(&pc_ref, pose_result);
	pc_eval = pc_ref;

	Pose pose;
	pose.t = Eigen::Matrix4f::Identity();

	// run ICP (slow framerate at high iteration count.  Pointcloud size problem?)
	icp->compute(pc_ref, pose, pose_result, rms);

	// Update the graphics model matrix and the lines between the nearest neighbors. 
	glm::mat4 ref_mat;
	conv->Matrix4f2Mat4(icp->Rt(), ref_mat);

	gl_reference_eval->setModelmatrix(ref_mat);
	//gl_knn_lines->updatePoints(pc_ref.points, pc_camera.points, icp->getNN());
}


/*
The main render and processing loop. 
@param pm - projection matrix
@param vm - camera view matrix. 
*/
void render_loop(glm::mat4 pm, glm::mat4 vm) {


	//-----------------------------------------------------------
	// Rendering

	gl_reference_point_cloud->draw(pm, vm);

	gl_camera_point_cloud->draw(pm, vm);

	gl_reference_eval->draw(pm, vm);

	gl_knn_lines->draw(pm, vm);


	runSequenceTest();

	//The camera point cloud loop
	if (usingCam)
	{
		findModel();
	}

	
	Sleep(25);
}



void runSequenceTest(void) {


	switch(sequence_test_case)
	{
		case 0:
		{
			icp->setMaxIterations(1);
			icp->setVerbose(true, 0);
			icp->setRejectMaxAngle(25.0);
			icp->setRejectMaxDistance(0.5);
			icp->setRejectionMethod(ICPReject::DIST);


			pc_ref = pc_ref_as_loaded;
			pc_eval = pc_ref;
			pc_camera = pc_camera_as_loaded;
			Eigen::Vector3f t(0.1, -0.0, 0.600);
			Eigen::Vector3f R(45.0, 0.0, 180.0);
			PointCloudTransform::Transform(&pc_ref, t, R);
			PointCloudTransform::Transform(&pc_eval, t, R);



			icp->setCameraData(pc_camera);

			gl_camera_point_cloud->updatePoints();
			gl_reference_eval->updatePoints();
			gl_reference_point_cloud->updatePoints();
			gl_reference_eval->setModelmatrix(glm::mat4(1));
			gl_reference_point_cloud->setModelmatrix(glm::mat4(1));

			pose_result = Eigen::Matrix4f::Identity();
			sequence_test_case = 1;
			break;
		}
		case 1:

			break;
		case 2:
		{
			icp->setRejectionMethod(ICPReject::DIST_ANG);
			PointCloudTransform::Transform(&pc_ref, icp->Rt2());
			PointCloudTransform::Transform(&pc_eval, icp->Rt2());
			gl_reference_eval->updatePoints();
			gl_reference_point_cloud->updatePoints();

			gl_reference_eval->setModelmatrix(glm::mat4(1));
			gl_reference_point_cloud->setModelmatrix(glm::mat4(1));

			Eigen::Vector3f t(0.05, 0.0, -0.0);
			Eigen::Vector3f R(0.05, 0.0, -0.0);
			PointCloudTransform::Transform(&pc_camera, t, R);
			icp->setCameraData(pc_camera);
			gl_camera_point_cloud->updatePoints();
			sequence_test_case = 3;
			break;	
		}
		case 3: // idle



			break;
		case 4:
		{
			icp->setRejectionMethod(ICPReject::DIST_ANG);
			PointCloudTransform::Transform(&pc_ref, icp->Rt2());
			PointCloudTransform::Transform(&pc_eval, icp->Rt2());
			gl_reference_eval->updatePoints();
			gl_reference_point_cloud->updatePoints();

			gl_reference_eval->setModelmatrix(glm::mat4(1));
			gl_reference_point_cloud->setModelmatrix(glm::mat4(1));

			Eigen::Vector3f t(0.02, 0.01, -0.0);
			Eigen::Vector3f R(0.02, 0.00, -0.0);
			PointCloudTransform::Transform(&pc_camera, t, R);
			icp->setCameraData(pc_camera);
			gl_camera_point_cloud->updatePoints();
			sequence_test_case = 5;
			break;
		}
		case 5: // idle
			break;
	}

}


void startAutoICP2(void)
{

	icp->setMaxIterations(200);

	Pose pose;
	pose.t = Eigen::Matrix4f::Identity();
	icp->compute(pc_ref, pose, pose_result, rms);


	// Update teh graphic reference model

	glm::mat4 ref_mat;
	conv->Matrix4f2Mat4(icp->Rt(), ref_mat);

	gl_reference_eval->setModelmatrix(ref_mat);
	gl_reference_point_cloud->setModelmatrix(ref_mat);

	// Update the lines between the point clouds showing the nearest neighbors

	gl_knn_lines->updatePoints(pc_ref.points, pc_camera.points, icp->getNN());

	



	// Output the pose

	std::cout << "[INFO] - Pose\n";
	std::cout << pose_result << std::endl;

}



int main(int argc, char** argv)
{
	
	std::cout << "ICP Camera data test.\n" << std::endl;
	std::cout << "This application implements an ICP test to determine whether or not the ICP algorithm for" << std::endl;
	std::cout << "TrackingExpert+ is running correctly. It works with a set of point cloud recorded with an " << std::endl;
	std::cout << "Azure Kinect camera. The point set contains a Stanford bunny model. ICP registers" << std::endl;
	std::cout << "a reference model with this bunny.\n " << std::endl;
	std::cout << "Note that the test files are stored in ../data/test/stanford_bunny_desk.\n " << std::endl;
	
	std::cout << "Rafael Radkowski\nIowa State University\nrafael@iastate.edu" << std::endl;
	std::cout << "-----------------------------------------------------------------------------------------\n" << std::endl;
	std::cout << "Keyboard layout" << std::endl;

	std::cout << "= \tTo change the camera point set currently loaded." << std::endl;
	std::cout << "a \tTo start ICP for the current point set." << std::endl;
	std::cout << "s \tTo start a manual ICP run for the current point set." << std::endl;
	std::cout << "space \tTo step to the next iteration; works only with a manual run started with 's'." << std::endl;
	std::cout << "n \tenable or disable normal vector rendering." << std::endl;
	std::cout << "r \tenable or disable the reference model rendering." << std::endl;
	std::cout << "c \tshow the nearest neighbors between both point clouds. " << std::endl;
	std::cout << ", \ttoggle running ICP off of the live camera point cloud feed. " << std::endl;

	std::cout << "\n\n" << std::endl;

	/*------------------------------------------------------------------------
	Read the data for this test. The files are stored in "../data/test/stanford_bunny_desk"
	The function reads automatically all files of type ply. 
	*/

	std::string data_path = "../data/test/stanford_bunny_desk";
	bool ret = ReadFiles::GetFileList(data_path, "ply", files);

	if (!ret) {
		std::cout << "[ERROR] - Did not find any files in " << data_path << endl;
		return 1;
	}


	// get the current data and time 
	date = TimeUtils::GetCurrentDateTime();

	
	/*------------------------------------------------------------------------
	The initial pose for the test files in ../data/test/stanford_bunny_desk"
	*/

	initial_pos.push_back(Eigen::Vector3f(0.1, -0.0, 0.60)); initial_rot.push_back(Eigen::Vector3f(45.0, 0.0, 180.0)); ground_truth_rms.push_back(0.0015492);
	initial_pos.push_back(Eigen::Vector3f(0.1, -0.0, 0.60)); initial_rot.push_back(Eigen::Vector3f(45.0, 0.0, 180.0)); ground_truth_rms.push_back(0.0015501);
	initial_pos.push_back(Eigen::Vector3f(0.1, 0.0, 0.6)); initial_rot.push_back(Eigen::Vector3f(45.0, 0.0, 180.0)); ground_truth_rms.push_back(0.0015498);
	initial_pos.push_back(Eigen::Vector3f(0.1, 0.00, 0.65)); initial_rot.push_back(Eigen::Vector3f(45.0, 0.0, 180.0));ground_truth_rms.push_back(0.0015492);
	initial_pos.push_back(Eigen::Vector3f(0.0, -0.0, 0.65)); initial_rot.push_back(Eigen::Vector3f(45.0, 0.0, 180.0)); ground_truth_rms.push_back(0.0015492);
	initial_pos.push_back(Eigen::Vector3f(0.05, 0.2, 0.05)); initial_rot.push_back(Eigen::Vector3f(0.0, 0.0, 0.0));ground_truth_rms.push_back(0.0015501);
	initial_pos.push_back(Eigen::Vector3f(0.0, 0.2, 0.0)); initial_rot.push_back(Eigen::Vector3f(10.0, 0.0, 0.0));ground_truth_rms.push_back(0.0015504);
	initial_pos.push_back(Eigen::Vector3f(0.2, 0.0, 0.0)); initial_rot.push_back(Eigen::Vector3f(25.0, 0.0, 0.0));ground_truth_rms.push_back(0.0015504);
	initial_pos.push_back(Eigen::Vector3f(0.0, 0.0, 0.2)); initial_rot.push_back(Eigen::Vector3f(0.0, 25.0, 20.0));ground_truth_rms.push_back(0.0015504);
	initial_pos.push_back(Eigen::Vector3f(0.2, 0.2, 0.0)); initial_rot.push_back(Eigen::Vector3f(20.0, 10.0, -10.0));ground_truth_rms.push_back(0.0015483);
	initial_pos.push_back(Eigen::Vector3f(0.0, 0.2, 0.2)); initial_rot.push_back(Eigen::Vector3f(10.0, 10.0, 0.0));ground_truth_rms.push_back(0.0015492);
	initial_pos.push_back(Eigen::Vector3f(0.0, 0.2, 0.2)); initial_rot.push_back(Eigen::Vector3f(2.0, 45.0, 10.0));ground_truth_rms.push_back(0.00247738);




	/*------------------------------------------------------------------------
	Load the first object for the test
	*/
	sampling_method = SamplingMethod::UNIFORM;
	sampling_param.grid_x = 0.01;
	sampling_param.grid_y = 0.01;
	sampling_param.grid_z = 0.01;
	sampling_param.uniform_step = 7;
	Sampling::SetMethod(sampling_method, sampling_param);

	ReaderWriterOBJ::Read(ref_file, pc_ref_as_loaded.points, pc_ref_as_loaded.normals, false, false);
	Sampling::Run(pc_ref_as_loaded, pc_ref_as_loaded);
	pc_ref = pc_ref_as_loaded;


	// Move the point cloud to a different position. 
	PointCloudTransform::Transform(&pc_ref,  initial_pos[run_test], initial_rot[run_test], false);


	
	ReaderWriterOBJ::Read(ref_file, pc_eval_as_loaded.points, pc_eval_as_loaded.normals, false, false);
	Sampling::Run(pc_eval_as_loaded, pc_eval_as_loaded);
	pc_eval = pc_eval_as_loaded;
	 
	PointCloudTransform::Transform(&pc_eval, initial_pos[run_test], initial_rot[run_test], false);
	
	/*------------------------------------------------------------------------
	Load the second object for the test. 
	*/
	std:string camera_file = files[run_test];

	camera_file = "../data/test/Azure_Kinect_model_1_2020-06-18_05-09-00_pc.ply";
	ReaderWriterUtil::Read(camera_file, pc_camera_as_loaded.points, pc_camera_as_loaded.normals, true, false);
	
	sampling_param.grid_x = 0.01;
	sampling_param.grid_y = 0.01;
	sampling_param.grid_z = 0.01;
	Sampling::SetMethod(sampling_method, sampling_param);
	Sampling::Run(pc_camera_as_loaded, pc_camera_as_loaded);
	pc_camera = pc_camera_as_loaded;


		
	/*------------------------------------------------------------------------
	Crete an ICP instance
	*/
	Pose pose; 
	pose.t =Eigen::Affine3f::Identity();
	
	icp = new texpert::ICP();
	icp->setMinError(0.00000001);
	icp->setMaxIterations(1);
	icp->setVerbose(true, 0);
	icp->setRejectMaxAngle(45.0);
	icp->setRejectMaxDistance(0.1);
	icp->setRejectionMethod(ICPReject::DIST_ANG);
	icp->setCameraData(pc_camera_as_loaded);
	icp->compute(pc_ref, pose, pose_result, rms);
	


	/*-------------------------------------------------------------------------------------
	create the renderer.
	The renderer executes the main loop in this demo. 
	*/
	glm::mat4 vm = glm::lookAt(glm::vec3(0.0f, 0.0, -0.5f), glm::vec3(0.0f, 0.0f, 0.5), glm::vec3(0.0f, 1.0f, 0.0f));
	window = new isu_ar::GLViewer();
	window->create(1280, 1280, "Tracking test");
	window->addRenderFcn(render_loop);
	window->addKeyboardCallback(keyboard_callback);
	window->setViewMatrix(vm);
	window->setClearColor(glm::vec4(1, 1, 1, 1));
	window->enableCameraControl(true);

	/*
	Create the 3D render content
	*/
	Eigen::Affine3f a_mat;
	a_mat = pose_result;
	glm::mat4 a_glmat;
	conv->Affine3f2Mat4(a_mat, a_glmat);

	// Point cloud showing the reference model.
	gl_reference_point_cloud = new	isu_ar::GLPointCloudRenderer(pc_ref.points, pc_ref.normals);
	gl_reference_point_cloud->setPointColor(glm::vec3(0.0,1.0,0.0));
	gl_reference_point_cloud->setNormalColor(glm::vec3(0.0,0.8,0.8));
	gl_reference_point_cloud->setNormalGfxLength(0.02f);

	// point clodu showing an evaluation model
	gl_reference_eval = new	isu_ar::GLPointCloudRenderer(pc_eval.points, pc_eval.normals);
	gl_reference_eval->setPointColor(glm::vec3(1.0,1.0,0.0));
	gl_reference_eval->setNormalColor(glm::vec3(0.5,0.8,0.8));
	gl_reference_eval->setNormalGfxLength(0.02f);
	gl_reference_eval->enablePointRendering(true);
	gl_reference_eval->setModelmatrix(a_glmat);

	// Point cloud showing the camera data. 
	gl_camera_point_cloud = new	isu_ar::GLPointCloudRenderer(pc_camera.points, pc_camera.normals);
	gl_camera_point_cloud->setPointColor(glm::vec3(1.0,0.0,0.0));
	gl_camera_point_cloud->setNormalColor(glm::vec3(0.8,0.5,0.0));
	gl_camera_point_cloud->setNormalGfxLength(0.02f);
	gl_camera_point_cloud->enablePointRendering(true);


	// line renderer for debugging nearest neighbors
	gl_knn_lines = new isu_ar::GLLineRenderer(pc_ref.points, pc_camera.points, icp->getNN());
	gl_knn_lines->updatePoints();

	// Camera point cloud producer
	cam = new KinectAzureCaptureDevice(0, KinectAzureCaptureDevice::Mode::RGBIRD, false);
	cam_cloud = new PointCloudProducer(*cam, pc_camera_as_loaded);

	sampling_param.uniform_step = 10;
	cam_cloud->setSampingMode(sampling_method, sampling_param);

	//thread_1.lock(); // block the start until the render window is up
	//std::thread test_run_static(runTest);
	

	//Sleep(100);
	// start the window along with  point cloud processing
	window->start();



	// delete all instances. 
	//test_run_static.detach();
	delete window;
	

}




