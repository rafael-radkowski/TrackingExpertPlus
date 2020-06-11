

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
isu_ar::GLPointCloudRenderer* gl_camera_point_cloud;
isu_ar::GLPointCloudRenderer* gl_reference_point_cloud;
isu_ar::GLPointCloudRenderer* gl_reference_eval; // evaluation point cloud for visual evaluation

int							gl_normal_rendering = 0;


SamplingParam		sampling_param;
SamplingMethod		sampling_method;


// The reference point cloud
PointCloud			pc_ref;
PointCloud			pc_ref_as_loaded;

// The test point cloud
PointCloud			pc_camera;
PointCloud			pc_camera_as_loaded;


std::string			ref_file = "../data/stanford_bunny_pc.obj ";
std::string			camera_file = "../data/stanford_bunny_cam.obj ";

// ICP
texpert::ICP*		icp;

std::vector< Eigen::Vector3f>	initial_pos;
std::vector< Eigen::Vector3f>	initial_rot;
std::vector< float>				ground_truth_rms;
int								current_set = 0;

Eigen::Matrix4f pose_result;
float rms = 1000.0;
float overall_rms = 0.0;
int N = 0;

int file_counter = 1;
string date = "";
std::mutex thread_1;


void runTest(void);
void runTestManual(void);

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
			runTestManual();
			break;
			}
		}
		break;
		
		case 1: // key down

			break;
	}
}


void runTestManual(void)
{
	if(current_set >= initial_pos.size()) current_set = 0;

	cout << "\n";
	pc_ref = pc_ref_as_loaded;
	// Move the point cloud to a different position. 
	PointCloudTransform::Transform(&pc_ref, initial_pos[current_set], initial_rot[current_set] );
	pc_ref.centroid0 = PointCloudUtils::CalcCentroid(&pc_ref);

	Pose pose;
	pose.t =  Eigen::Matrix4f::Identity();
	icp->compute(pc_ref, pose, pose_result, rms);
	current_set++;
	N++;

	//---------------------------------------------------------------------------
	// Apply the ICP pose to the initial point cloud
	Matrix3f R;
	Vector3f t;
		
	R(0) = pose_result(0);
	R(1) = pose_result(1);
	R(2) = pose_result(2);
	R(3) = pose_result(4);
	R(4) = pose_result(5);
	R(5) = pose_result(6);
	R(6) = pose_result(8);
	R(7) = pose_result(9);
	R(8) = pose_result(10);
	t.x() = pose_result(12);
	t.y() = pose_result(13);
	t.z() = pose_result(14);


	PointCloud test_cloud = pc_ref;
	for_each(test_cloud.points.begin(), test_cloud.points.end(), [&](Vector3f& p){p = (R * p) + t;});		
	test_cloud.centroid0 = PointCloudUtils::CalcCentroid(&test_cloud);

	//---------------------------------------------------------------------------
	// Compare the centroid of both point clouds and the RMS

	float centroid_dist =  (pc_camera_as_loaded.centroid0 - test_cloud.centroid0).norm();

	if (centroid_dist > 0.025) {
		std::cout << "[ERROR] - test: " << N+1 << ", centroid distance is " << centroid_dist << endl;
	}else
	{
		std::cout << "[SUCCESS] - test: " << N+1 << " successful, with centroid distance is " << centroid_dist << endl;
	}


	Eigen::Affine3f a_mat;
	a_mat = pose_result;
	gl_reference_eval->setModelmatrix(MatrixUtils::Affine3f2Mat4(a_mat));

	std::cout << "[INFO] - test: " << N << " case " << current_set-1 << ", mean rms: " << overall_rms << std::endl;

	if(current_set >= initial_pos.size()) current_set = 0;
}



void runTest(void)
{
	
	//thread_1.lock();

	int error_count = 0;

	while(current_set < initial_pos.size()){

		cout << "\n";
		pc_ref = pc_ref_as_loaded;

		// -------------------------------------------------------------------------
		// Move the point cloud to a different position. 
		PointCloudTransform::Transform(&pc_ref, initial_pos[current_set], initial_rot[current_set] );

		pc_ref.centroid0 = PointCloudUtils::CalcCentroid(&pc_ref);
		
		// --------------------------------------------------------------------------
		// Compute ICP
		Pose pose;
		pose.t =  Eigen::Matrix4f::Identity();
		icp->compute(pc_ref, pose, pose_result, rms);

		//---------------------------------------------------------------------------
		// Apply the ICP pose to the initial point cloud
		Matrix3f R;
		Vector3f t;
		
		R(0) = pose_result(0);
		R(1) = pose_result(1);
		R(2) = pose_result(2);
		R(3) = pose_result(4);
		R(4) = pose_result(5);
		R(5) = pose_result(6);
		R(6) = pose_result(8);
		R(7) = pose_result(9);
		R(8) = pose_result(10);
		t.x() = pose_result(12);
		t.y() = pose_result(13);
		t.z() = pose_result(14);

		PointCloud test_cloud = pc_ref;
		for_each(test_cloud.points.begin(), test_cloud.points.end(), [&](Vector3f& p){p = (R * p) + t;});		
		test_cloud.centroid0 = PointCloudUtils::CalcCentroid(&test_cloud);

		//---------------------------------------------------------------------------
		// Compare the centroid of both point clouds and the RMS

		float centroid_dist =  (pc_camera_as_loaded.centroid0 - test_cloud.centroid0).norm();

		if (centroid_dist > 0.025) {
			std::cout << "[ERROR] - test: " << N+1 << ", centroid distance is " << centroid_dist << endl;
		}else
		{
			std::cout << "[SUCCESS] - test: " << N+1 << " successful, with centroid distance is " << centroid_dist << endl;
		}


		current_set++;
		N++;


		//---------------------------------------------------------------------------
		// Set the matrix to obtain a visual image for comparisons.

		Eigen::Affine3f a_mat;
		a_mat = pose_result;
		gl_reference_eval->setModelmatrix(MatrixUtils::Affine3f2Mat4(a_mat));

		Sleep(200);
				
	}

	std::cout << "\n[INFO] - All " <<  initial_pos.size() << " completed with a mean rms: " << overall_rms << " , " << error_count <<  " false results" << std::endl;
	//if(current_set >= initial_pos.size()) current_set = 0;

	//thread_1.unlock();
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


class RenderLoop {

public:

	static void fcn(glm::mat4 pm, glm::mat4 vm) {

		//thread_1.unlock();

		//-----------------------------------------------------------
		// Rendering
		gl_reference_point_cloud->draw(pm, vm);

		gl_camera_point_cloud->draw(pm, vm);

		gl_reference_eval->draw(pm, vm);

	
		Sleep(25);
	}
};



int main(int argc, char** argv)
{
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

	

	icp = new texpert::ICP();
	icp->setMinError(0.00000001);
	icp->setMaxIterations(100);
	icp->setRejectionMethod(ICPReject::DIST_ANG);
	icp->setVerbose(true, 1);
	icp->setRejectMaxDistance(0.2);
	icp->setRejectMaxAngle(25.0);
	icp->setCameraData(pc_camera_as_loaded);
	icp->compute(pc_ref, pose, pose_result, rms);
	


	/*-------------------------------------------------------------------------------------
	create the renderer.
	The renderer executes the main loop in this demo. 
	*/
	glm::mat4 vm = glm::lookAt(glm::vec3(0.0f, 0.0, -0.5f), glm::vec3(0.0f, 0.0f, 0.5), glm::vec3(0.0f, 1.0f, 0.0f));
	window = new isu_ar::GLViewer();
	window->create(1280, 1280, "Tracking test");
	window->addRenderFcn(std::bind(RenderLoop::fcn, std::placeholders::_1, std::placeholders::_2));
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
	std::thread test_run_static(runTest);  

	Sleep(100);
	// start the window along with  point cloud processing
	window->start();



	// delete all instances. 
	delete window;
	test_run_static.detach();

}




