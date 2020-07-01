/*
@file icp_test.cpp

This file implements an ICP test to determine whether or not the ICP algorithm for 
TrackingExpert+ is running correctly. This test is a simple base test. It works with two 
identical objects and matches object A to object B. 
All objects are loaded from a file.

The scene contains the following point sets:
Red points: camera point cloud, or the simulated camera point cloud. The object does not move and remains at its global position. 
Green point set: reference point set at its original position.
Yellow point set: evaluation point set at the ICP pose after ICP terminates. 

The file runs one test set automatically. 

Manual operations:
- Press 'space' to run through all test steps manually. 

Graphics keyboard layout:
- n - enable or disable normal vector rendering


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
#include <thread>
#include <mutex>

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
#include "ICP.h"  // the ICP class to test

using namespace texpert;

// The OpenGL window
isu_ar::GLViewer* window;

// OpenGL point cloud objects showing the point cloud
isu_ar::GLPointCloudRenderer* gl_camera_point_cloud;
isu_ar::GLPointCloudRenderer* gl_reference_point_cloud;
isu_ar::GLPointCloudRenderer* gl_reference_eval; // evaluation point cloud for visual evaluation


// helper variable to enable/disable the gl renderer. 
int							gl_normal_rendering = 0;

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

// the two test files to load. 
std::string			ref_file = "../data/stanford_bunny_pc.obj";
std::string			camera_file = "../data/stanford_bunny_pc.obj";

//////////////////////////////////////////////////////////////////////
// ICP
texpert::ICP*		icp;

// IPC parameters and parameters for testing. 
std::vector< Eigen::Vector3f>	initial_pos;
std::vector< Eigen::Vector3f>	initial_rot;
std::vector< float>				ground_truth_rms;
int								current_set = 0;

Eigen::Matrix4f pose_result;
int N = 0;

int file_counter = 1;
string date = "";


//--------------------------------------------------------------------------------
// Function prototypes

float runTest(void);
void runAllTests(void);


/*
Keyboard callback function
*/
void keyboard_callback( int key, int action) {


	//cout << key << " : " << action << endl;
	switch (action) {
	case 0:  // key up
	
		switch (key) {
		case 87: // w
		{
			
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
		case 32: // space
			{
				runTest();
				break;
			}
		}
		break;
		
		case 1: // key down

			break;
	}
}


/*
Compare two identical point sets. 
The function compares the two point sets and determines the rms.
Note that this function assumes that the point sets a and b are of the same object and
the same point configuration. Only the global pose is different. 
Thus, do not use the function for two different oobjects. 
*/
float comparePointSets(PointCloud& a, PointCloud& b) {

	if (a.size() != b.size()) {
		std::cout << "[ERROR] Points sets a and b are of different size -> " << a.size() << " vs. " << b.size() << std::endl;
		return 100000.0;
	}

	float dist = 0.0;
	for (int i = 0; i < a.size(); i++) {
		dist += (a.points[i] - b.points[i]).norm();
	}
	dist /= a.size();

	return dist;

}


/*
This function run the test manually, each time the user presses the space button. 
*/
float runTest(void)
{
	if(current_set >= initial_pos.size()) current_set = 0;

	cout << "\n";
	pc_ref = pc_ref_as_loaded;

	//---------------------------------------------------------------------------
	// Move the point cloud to a different position. 
	PointCloudTransform::Transform(&pc_ref, initial_pos[current_set], initial_rot[current_set] );
	pc_ref.centroid0 = PointCloudUtils::CalcCentroid(&pc_ref);

	//---------------------------------------------------------------------------
	// Apply ICP
	float icp_rms = 0.0;
	Pose pose;
	pose.t =  Eigen::Matrix4f::Identity();
	icp->compute(pc_ref, pose, pose_result, icp_rms);
	
	//---------------------------------------------------------------------------
	// Apply the ICP pose to the initial point cloud
	Eigen::Vector3f t = icp->t();
	Eigen::Matrix3f R = icp->R().inverse();


	PointCloud test_cloud = pc_ref;
	for_each(test_cloud.points.begin(), test_cloud.points.end(), [&](Vector3f& p){p = (R * (p - test_cloud.centroid0)) + (t + test_cloud.centroid0);});		
	

	//---------------------------------------------------------------------------
	// Compare the centroid of both point clouds and the RMS

	float rms =  comparePointSets(pc_camera_as_loaded, test_cloud);
	
	if (rms > 0.025) {
		std::cout << "[ERROR] - test: " << N+1 << ", rms is " << rms << endl;
	}else
	{
		std::cout << "[SUCCESS] - test: " << N+1 << " successful, with rms is " << rms << endl;
	}


	Eigen::Affine3f a_mat;
	a_mat = pose_result;
	gl_reference_eval->setModelmatrix(MatrixUtils::ICPRt3Mat4( icp->Rt()));

	current_set++;
	N++;
	

	std::cout << "[INFO] - test: " << N << " case " << current_set-1 << ", rms: " << rms << std::endl;

	if(current_set >= initial_pos.size()) current_set = 0;

	return rms;
}


/*
Run all the tests automatically. 
*/
void runAllTests(void)
{
	

	int error_count = 0;
	int count = 0;
	float overall_rms = 0.0;

	while(count < initial_pos.size()){

		cout << "\n";
		pc_ref = pc_ref_as_loaded;

		// run one test 
		overall_rms += runTest();


		count++;
		Sleep(200);
				
	}
	overall_rms /= (count);

	std::cout << "\n[INFO] - All " <<  initial_pos.size() << " completed with a mean rms: " << overall_rms << " , " << error_count <<  " false results" << std::endl;
	
}


/*
The main render and processing loop. 
@param pm - projection matrix
@param vm - camera view matrix. 
*/
void render_loop(glm::mat4 pm, glm::mat4 vm) {

	//thread_1.unlock();

	//-----------------------------------------------------------
	// Rendering
	gl_reference_point_cloud->draw(pm, vm);

	gl_camera_point_cloud->draw(pm, vm);

	gl_reference_eval->draw(pm, vm);

	
	Sleep(25);
}




int main(int argc, char** argv)
{
	std::cout << "ICP test.\n" << std::endl;
	std::cout << "This application implements a simple ICP test to determine whether or not the ICP algorithm for" << std::endl;
	std::cout << "TrackingExpert+ is running correctly. It works with one set of points and registers the set of  " << std::endl;
	std::cout << "points agains each other. \n" << std::endl;
	std::cout << "Note that the test files are stored in ../data/stanford_bunny_pc.obj.\n " << std::endl;
	
	std::cout << "Rafael Radkowski\nIowa State University\nrafael@iastate.edu" << std::endl;
	std::cout << "-----------------------------------------------------------------------------------------\n" << std::endl;
	std::cout << "Keyboard layout" << std::endl;

	std::cout << "space \tTo step through all tests manually after the automatic run terminates." << std::endl;
	std::cout << "n \tenable or disable normal vector rendering." << std::endl;

	std::cout << "\n\n" << std::endl;

	bool err = false;


	date = TimeUtils::GetCurrentDateTime();

	initial_pos.push_back(Eigen::Vector3f(0.0, 0.2, 0.0)); initial_rot.push_back(Eigen::Vector3f(0.0, 0.0, 0.0)); ground_truth_rms.push_back(0.0015492);
	initial_pos.push_back(Eigen::Vector3f(0.2, 0.0, 0.0)); initial_rot.push_back(Eigen::Vector3f(0.0, 0.0, 0.0)); ground_truth_rms.push_back(0.0015501);
	initial_pos.push_back(Eigen::Vector3f(0.0, 0.0, 0.2)); initial_rot.push_back(Eigen::Vector3f(0.0, 0.0, 0.0)); ground_truth_rms.push_back(0.0015498);
	initial_pos.push_back(Eigen::Vector3f(0.2, 0.15, 0.0)); initial_rot.push_back(Eigen::Vector3f(0.0, 0.0, 0.0));ground_truth_rms.push_back(0.0015492);
	initial_pos.push_back(Eigen::Vector3f(0.0, 0.2, 0.15)); initial_rot.push_back(Eigen::Vector3f(0.0, 0.0, 0.0));ground_truth_rms.push_back(0.0015492);
	initial_pos.push_back(Eigen::Vector3f(0.05, 0.2, 0.05)); initial_rot.push_back(Eigen::Vector3f(0.0, 0.0, 0.0));ground_truth_rms.push_back(0.0015501);
	initial_pos.push_back(Eigen::Vector3f(0.0, 0.2, 0.0)); initial_rot.push_back(Eigen::Vector3f(10.0, 0.0, 0.0));ground_truth_rms.push_back(0.0015504);
	initial_pos.push_back(Eigen::Vector3f(0.2, 0.0, 0.0)); initial_rot.push_back(Eigen::Vector3f(25.0, 0.0, 0.0));ground_truth_rms.push_back(0.0015504);
	initial_pos.push_back(Eigen::Vector3f(0.0, 0.0, 0.2)); initial_rot.push_back(Eigen::Vector3f(0.0, 25.0, 20.0));ground_truth_rms.push_back(0.0015504);
	initial_pos.push_back(Eigen::Vector3f(0.2, 0.1, 0.0)); initial_rot.push_back(Eigen::Vector3f(20.0, 10.0, -10.0));ground_truth_rms.push_back(0.0015483);
	initial_pos.push_back(Eigen::Vector3f(0.0, 0.2, 0.2)); initial_rot.push_back(Eigen::Vector3f(10.0, 10.0, 0.0));ground_truth_rms.push_back(0.0015492);
	initial_pos.push_back(Eigen::Vector3f(0.0, 0.2, 0.2)); initial_rot.push_back(Eigen::Vector3f(2.0, 20.0, 10.0));ground_truth_rms.push_back(0.00247738);




	/*------------------------------------------------------------------------
	Load the first object for the test
	*/
	sampling_method = SamplingMethod::UNIFORM;
	sampling_param.grid_x = 0.01;
	sampling_param.grid_y = 0.01;
	sampling_param.grid_z = 0.01;
	Sampling::SetMethod(sampling_method, sampling_param);

	ReaderWriterOBJ::Read(ref_file, pc_ref_as_loaded.points, pc_ref_as_loaded.normals, false, false);
	Sampling::Run(pc_ref_as_loaded, pc_ref_as_loaded);
	pc_ref = pc_ref_as_loaded;

	pc_ref.centroid0 = PointCloudUtils::CalcCentroid(&pc_ref);



	// Move the point cloud to a different position. 
	PointCloudTransform::Transform(&pc_ref,  Eigen::Vector3f(0.0, 0.2, 0.0), Eigen::Vector3f(0.0, 0.0, 0.0));
	

	/*------------------------------------------------------------------------
	Load the second object for the test. 
	*/
	ReaderWriterOBJ::Read(camera_file, pc_camera_as_loaded.points, pc_camera_as_loaded.normals, false, false);
	//LoaderObj::Read(camera_file, &pc_camera_as_loaded.points, &pc_camera_as_loaded.normals, false, true);
	Sampling::Run(pc_camera_as_loaded, pc_camera_as_loaded);
	pc_camera_as_loaded.centroid0 = PointCloudUtils::CalcCentroid(&pc_camera_as_loaded);
	pc_camera = pc_camera_as_loaded;

		
	/*------------------------------------------------------------------------
	Crete an ICP instance
	*/
	Pose pose;
	pose.t = Eigen::Matrix4f::Identity();
	float icp_rms = 0.0;
	

	icp = new texpert::ICP();
	icp->setMinError(0.00000001);
	icp->setMaxIterations(25);
	icp->setRejectionMethod(ICPReject::DIST_ANG);
	icp->setVerbose(true, 1);
	icp->setRejectMaxDistance(0.2);
	icp->setRejectMaxAngle(65.0);
	icp->setCameraData(pc_camera_as_loaded);
	icp->compute(pc_ref, pose, pose_result, icp_rms);
	


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


	/*
	Create the 3D render content
	*/
	gl_reference_point_cloud = new	isu_ar::GLPointCloudRenderer(pc_ref.points, pc_ref.normals);
	gl_reference_point_cloud->setPointColor(glm::vec3(0.0,1.0,0.0));
	gl_reference_point_cloud->setNormalColor(glm::vec3(0.0,0.8,0.8));
	gl_reference_point_cloud->setNormalGfxLength(0.005f);

	gl_reference_eval = new	isu_ar::GLPointCloudRenderer(pc_ref.points, pc_ref.normals);
	gl_reference_eval->setPointColor(glm::vec3(1.0,1.0,0.0));
	gl_reference_eval->setNormalColor(glm::vec3(0.5,0.8,0.8));
	gl_reference_eval->setNormalGfxLength(0.005f);

	Eigen::Affine3f a_mat;
	a_mat = pose_result;
	gl_reference_eval->setModelmatrix(MatrixUtils::Affine3f2Mat4(a_mat));
	
	

	gl_camera_point_cloud = new	isu_ar::GLPointCloudRenderer(pc_camera.points, pc_camera.normals);
	gl_camera_point_cloud->setPointColor(glm::vec3(1.0,0.0,0.0));
	gl_camera_point_cloud->setNormalColor(glm::vec3(0.8,0.5,0.0));
	gl_camera_point_cloud->setNormalGfxLength(0.005f);


	//thread_1.lock(); // block the start until the render window is up
	std::thread test_run_static(runAllTests);  

	Sleep(100);
	// start the window along with  point cloud processing
	window->start();



	// delete all instances. 
	delete window;
	test_run_static.detach();

}




