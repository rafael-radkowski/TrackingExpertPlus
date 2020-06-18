

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
#include "MatrixTransform.h"
#include "GLLineRenderer.h"

using namespace texpert;



// The OpenGL window
isu_ar::GLViewer* window;
isu_ar::GLPointCloudRenderer* gl_camera_point_cloud;
isu_ar::GLPointCloudRenderer* gl_reference_point_cloud;
isu_ar::GLPointCloudRenderer* gl_reference_eval; // evaluation point cloud for visual evaluation

// a debug visualization that allows one to render lines between points. 
isu_ar::GLLineRenderer*			gl_knn_lines;


int							gl_normal_rendering = 0;
int							gl_ref_rendering = 1;
int							gl_nn_rendering = 0;


SamplingParam		sampling_param;
SamplingMethod		sampling_method;


// The reference point cloud
PointCloud			pc_ref;
PointCloud			pc_ref_as_loaded;

// The test point cloud
PointCloud			pc_camera;
PointCloud			pc_camera_as_loaded;

PointCloud			pc_eval;
PointCloud			pc_eval_as_loaded;


std::string			ref_file = "../data/stanford_bunny_pc.obj";
std::string			camera_file = "";


// ground truth pose data
std::vector<std::pair<Eigen::Vector3f, Eigen::Vector3f>> gt_pose;



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
void startICP(void);
void startAutoICP(void);
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
				startICP();
		
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
		}
		break;
		
		case 1: // key down

			break;
	}
}



void startICP(void)
{
	if(current_set >= initial_pos.size()) current_set = 0;

	std::cout << "Manual ---------------------------------\n";

	icp->setMaxIterations(1);

	pc_ref = pc_ref_as_loaded;
	pc_eval = pc_ref_as_loaded;
	// Move the point cloud to a different position. 
	PointCloudTransform::Transform(&pc_ref, initial_pos[current_set], initial_rot[current_set] );
	PointCloudTransform::Transform(&pc_eval, initial_pos[current_set], initial_rot[current_set]);

	pose_result = Eigen::Matrix4f::Identity();

	/*pose_result = Eigen::Matrix4f::Identity();

	Pose pose;
	pose.t =  Eigen::Matrix4f::Identity();
	icp->compute(pc_ref, pose, pose_result, rms);
	current_set++;


	pose_result = icp->getGfxPose();
	*/
	//Eigen::Affine3f a_mat;
	//a_mat = pose_result;
	gl_reference_eval->setModelmatrix(glm::mat4(1.0));

	/*
	gl_knn_lines->updatePoints(pc_ref.points, pc_camera.points , icp->getNN(), MatrixUtils::Affine3f2Mat4(a_mat));

	std::cout << "[INFO] - translation x: " << pose_result(12) << ", y: " << pose_result(13) << ", z: " <<  pose_result(14) << std::endl;*/


	if(current_set >= initial_pos.size()) current_set = 0;
}

void runTestManual(void)
{
	
	// Move the point cloud to a different position. 
	//PointCloudTransform::Transform(&pc_ref, new_t, Eigen::Vector3f(0.0,0.0,0.0) );
	//PointCloudTransform::Transform(&pc_eval, new_t,  Eigen::Vector3f(0.0,0.0,0.0) );

	PointCloudTransform::Transform(&pc_ref, pose_result );
	PointCloudTransform::Transform(&pc_eval, pose_result );

	Pose pose;
	pose.t =  Eigen::Matrix4f::Identity();
	icp->compute(pc_ref, pose, pose_result, rms);
	
	
	Eigen::Vector3f centroid = PointCloudUtils::CalcCentroid(&pc_eval);
	
	Eigen::Affine3f a_mat;
	
	a_mat = pose_result;
	glm::mat4 m = MatrixUtils::Affine3f2Mat4(a_mat);
	glm::mat4 R = MatrixUtils::Affine3f2Mat4(a_mat);



	//m[0][0] = 1.0; m[1][0] = 0.0; m[2][0] = 0.0;
	//m[0][1] = 0.0; m[1][1] = 1.0; m[2][1] = 0.0;
	//m[0][2] = 0.0; m[1][2] = 0.0; m[2][2] = 1.0;
	 R[3][0] == R[3][1] ==  R[3][2] == 0.0;

	 
	glm::mat4 centInv = glm::translate(glm::vec3(-centroid.x(), -centroid.y(), -centroid.z() ));
	glm::mat4 cent = glm::translate(glm::vec3(centroid.x(), centroid.y(), centroid.z() ));
	glm::mat4 tInv = glm::translate(glm::vec3(-m[3][0],-m[3][1], -m[3][2]));
	glm::mat4 t = glm::translate(glm::vec3(m[3][0],m[3][1], m[3][2]));
	glm::mat4 all = t * cent * R *  centInv;
	gl_reference_eval->setModelmatrix(all);
	//gl_reference_eval->setModelmatrix(glm::mat4(1.0));


	
	gl_knn_lines->updatePoints(pc_ref.points, pc_camera.points , icp->getNN());
	

	std::cout << "[INFO] - glm x: " << m[3][0] << ", y: " << m[3][1] << ", z: " <<  m[3][2] << std::endl;
	std::cout << "[INFO] - translation x: " << pose_result(12) << ", y: " << pose_result(13) << ", z: " <<  pose_result(14) << std::endl;
	std::cout << pose_result << std::endl;
	
	//std::cout << "[INFO] - test: " << N << " case " << current_set-1 << ", mean rms: " << overall_rms << std::endl;

	
}


void startAutoICP(void)
{ 
	if(current_set >= initial_pos.size()) current_set = 0;
	std::cout << "Auto ---------------------------------\n";
	icp->setMaxIterations(100);

	pc_ref = pc_ref_as_loaded;
	pc_eval = pc_ref_as_loaded;

	// Move the point cloud to a different position. 
	PointCloudTransform::Transform(&pc_ref, initial_pos[current_set], initial_rot[current_set] );
	PointCloudTransform::Transform(&pc_eval, initial_pos[current_set], initial_rot[current_set]);
	gl_reference_eval->setModelmatrix(glm::mat4(1.0));
	pose_result = Eigen::Matrix4f::Identity();

	Pose pose;
	pose.t =  Eigen::Matrix4f::Identity();
	icp->compute(pc_ref, pose, pose_result, rms);
	current_set++;


	 /* glm computation: todo -> add to another class
	Eigen::Vector3f tall = icp->t();
	Eigen::Matrix3f Rall = icp->R();
	glm::mat4 Rglm = glm::mat4(1.0);

	for (int i = 0; i < 9; i++) {
		Rglm[i/3][i%3] = Rall(i/3, i%3);
	}
	
	Eigen::Vector3f centroid = PointCloudUtils::CalcCentroid(&pc_eval);

	glm::mat4 centInv = glm::translate(glm::vec3(-centroid.x(), -centroid.y(), -centroid.z() ));
	glm::mat4 cent = glm::translate(glm::vec3(centroid.x(), centroid.y(), centroid.z() ));
	glm::mat4 t = glm::translate(glm::vec3(tall.x(),tall.y(),tall.z()));
	glm::mat4 all = t *  cent * Rglm *  centInv;
	*/

	Eigen::Matrix4f Rt = icp->Rt();
	gl_reference_eval->setModelmatrix( MatrixUtils::ICPRt3Mat4(Rt));
	gl_knn_lines->updatePoints(pc_ref.points, pc_camera.points , icp->getNN());

	if(current_set >= initial_pos.size()) current_set = 0;
}



void runTest(void)
{
	
	//thread_1.lock();

	int error_count = 0;

	while(current_set < initial_pos.size()){

		pc_ref = pc_ref_as_loaded;
		// Move the point cloud to a different position. 
		PointCloudTransform::Transform(&pc_ref, initial_pos[current_set], initial_rot[current_set] );
		Pose pose;
		pose.t =  Eigen::Matrix4f::Identity();
		icp->compute(pc_ref, pose, pose_result, rms);

		if(std::abs(ground_truth_rms[current_set] - rms) > 0.00001 ) {
			error_count++;
			std::cout << "[ERROR] - test: " << N+1 << ", Ground truth  " << ground_truth_rms[current_set] <<  " deviates from result " <<  rms << std::endl;
		}

		current_set++;
		N++;

		overall_rms = overall_rms * (N - 1)/N + rms / N;


		Eigen::Affine3f a_mat;
		a_mat = pose_result;
		gl_reference_eval->setModelmatrix(MatrixUtils::Affine3f2Mat4(a_mat));

		std::cout << "[INFO] - test: " << N << " case " << current_set-1 << ", mean rms: " << overall_rms << std::endl;

		

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


	//-----------------------------------------------------------
	// Rendering

	gl_reference_point_cloud->draw(pm, vm);

	gl_camera_point_cloud->draw(pm, vm);

	gl_reference_eval->draw(pm, vm);

	gl_knn_lines->draw(pm, vm);

	
	Sleep(25);
}




int main(int argc, char** argv)
{

	std::string data_path = "../data/test/stanford_bunny_desk";
	std::vector<std::string> files;
	bool ret = ReadFiles::GetFileList(data_path, "ply", files);

	if (!ret) {
		std::cout << "[ERROR] - Did not find any files in " << data_path << endl;
		return 1;
	}

	bool err = false;

	date = TimeUtils::GetCurrentDateTime();


	initial_pos.push_back(Eigen::Vector3f(0.0, -0.0, 0.65)); initial_rot.push_back(Eigen::Vector3f(45.0, 0.0, 180.0)); ground_truth_rms.push_back(0.0015492);
//	initial_pos.push_back(Eigen::Vector3f(-0.02, -0.04, 0.65)); initial_rot.push_back(Eigen::Vector3f(45.0, 0.0, 180.0)); ground_truth_rms.push_back(0.0015501);
	/*initial_pos.push_back(Eigen::Vector3f(0.0, 0.0, 0.2)); initial_rot.push_back(Eigen::Vector3f(0.0, 0.0, 0.0)); ground_truth_rms.push_back(0.0015498);
	initial_pos.push_back(Eigen::Vector3f(0.2, 0.15, 0.0)); initial_rot.push_back(Eigen::Vector3f(0.0, 0.0, 0.0));ground_truth_rms.push_back(0.0015492);
	initial_pos.push_back(Eigen::Vector3f(0.0, 0.2, 0.15)); initial_rot.push_back(Eigen::Vector3f(0.0, 0.0, 0.0));ground_truth_rms.push_back(0.0015492);
	initial_pos.push_back(Eigen::Vector3f(0.05, 0.2, 0.05)); initial_rot.push_back(Eigen::Vector3f(0.0, 0.0, 0.0));ground_truth_rms.push_back(0.0015501);
	initial_pos.push_back(Eigen::Vector3f(0.0, 0.2, 0.0)); initial_rot.push_back(Eigen::Vector3f(10.0, 0.0, 0.0));ground_truth_rms.push_back(0.0015504);
	initial_pos.push_back(Eigen::Vector3f(0.2, 0.0, 0.0)); initial_rot.push_back(Eigen::Vector3f(25.0, 0.0, 0.0));ground_truth_rms.push_back(0.0015504);
	initial_pos.push_back(Eigen::Vector3f(0.0, 0.0, 0.2)); initial_rot.push_back(Eigen::Vector3f(0.0, 25.0, 20.0));ground_truth_rms.push_back(0.0015504);
	initial_pos.push_back(Eigen::Vector3f(0.2, 0.2, 0.0)); initial_rot.push_back(Eigen::Vector3f(20.0, 10.0, -10.0));ground_truth_rms.push_back(0.0015483);
	initial_pos.push_back(Eigen::Vector3f(0.0, 0.2, 0.2)); initial_rot.push_back(Eigen::Vector3f(10.0, 10.0, 0.0));ground_truth_rms.push_back(0.0015492);
	initial_pos.push_back(Eigen::Vector3f(0.0, 0.2, 0.2)); initial_rot.push_back(Eigen::Vector3f(2.0, 45.0, 10.0));ground_truth_rms.push_back(0.00247738);*/


	// ground truth pose data
	gt_pose.push_back( std::make_pair( Eigen::Vector3f(0.1, -0.12, 0.65), Eigen::Vector3f(45.0, 0.0, 180.0)));
	gt_pose.push_back( std::make_pair( Eigen::Vector3f(0.1, -0.07, 0.6), Eigen::Vector3f(45.0, 0.0, 180.0)));
	gt_pose.push_back( std::make_pair( Eigen::Vector3f(0.0, -0.07, 0.6), Eigen::Vector3f(45.0, 0.0, 180.0)));
	gt_pose.push_back( std::make_pair( Eigen::Vector3f(0.0, 0.0, 0.7), Eigen::Vector3f(45.0, 0.0, 180.0)));
	gt_pose.push_back( std::make_pair( Eigen::Vector3f(-0.0, -0.05, 0.67), Eigen::Vector3f(45.0, 0.0, 180.0)));

	int run_test = 4;

	/*------------------------------------------------------------------------
	Load the first object for the test
	*/
	sampling_method = SamplingMethod::UNIFORM;
	sampling_param.grid_x = 0.005;
	sampling_param.grid_y = 0.005;
	sampling_param.grid_z = 0.005;
	Sampling::SetMethod(sampling_method, sampling_param);

	ReaderWriterOBJ::Read(ref_file, pc_ref_as_loaded.points, pc_ref_as_loaded.normals, false, false);
	Sampling::Run(pc_ref_as_loaded, pc_ref_as_loaded);
	pc_ref = pc_ref_as_loaded;


	// Move the point cloud to a different position. 
	PointCloudTransform::Transform(&pc_ref,  gt_pose[run_test].first, gt_pose[run_test].second, false);


	
	ReaderWriterOBJ::Read(ref_file, pc_eval_as_loaded.points, pc_eval_as_loaded.normals, false, false);
	Sampling::Run(pc_eval_as_loaded, pc_eval_as_loaded);
	pc_eval = pc_eval_as_loaded;

	PointCloudTransform::Transform(&pc_eval, gt_pose[run_test].first, gt_pose[run_test].second, false);
	
	/*------------------------------------------------------------------------
	Load the second object for the test. 
	*/
	camera_file = files[run_test];
	ReaderWriterUtil::Read(camera_file, pc_camera_as_loaded.points, pc_camera_as_loaded.normals, true, false);
	//LoaderObj::Read(camera_file, &pc_camera_as_loaded.points, &pc_camera_as_loaded.normals, false, true);
	//Sampling::Run(pc_camera_as_loaded, pc_camera_as_loaded);
	pc_camera = pc_camera_as_loaded;


		
	/*------------------------------------------------------------------------
	Crete an ICP instance
	*/
	Pose pose; 
//	pose.t = MatrixTransform::CreateAffineMatrix(Eigen::Vector3f(0.15, -0.05, 0.6), Eigen::Vector3f(45.0, 0.0, 180.0));
	pose.t =Eigen::Affine3f::Identity();
	

	icp = new texpert::ICP();
	icp->setMinError(0.00000001);
	icp->setMaxIterations(1);
	icp->setVerbose(true, 1);
	icp->setRejectMaxAngle(45.0);
	icp->setRejectMaxDistance(0.1);
	icp->setRejectionMethod(ICPReject::DIST_ANG);
	icp->setCameraData(pc_camera_as_loaded);
	icp->compute(pc_ref, pose, pose_result, rms);
	
	//pose_result = pose.t.matrix();


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
	gl_reference_point_cloud = new	isu_ar::GLPointCloudRenderer(pc_ref.points, pc_ref.normals);
	gl_reference_point_cloud->setPointColor(glm::vec3(0.0,1.0,0.0));
	gl_reference_point_cloud->setNormalColor(glm::vec3(0.0,0.8,0.8));
	gl_reference_point_cloud->setNormalGfxLength(0.02f);

	gl_reference_eval = new	isu_ar::GLPointCloudRenderer(pc_eval.points, pc_eval.normals);
	gl_reference_eval->setPointColor(glm::vec3(1.0,1.0,0.0));
	gl_reference_eval->setNormalColor(glm::vec3(0.5,0.8,0.8));
	gl_reference_eval->setNormalGfxLength(0.02f);
	gl_reference_eval->enablePointRendering(true);
	

	/*glm::mat4 test_matrix(0.993924, 0.0264946, -0.106836, -0.00770066,
					-0.085335, 0.798538, -0.595866, 0.036563,
					0.0695257, 0.601362, 0.795946, -0.00742521,
					0, 0, 0, 1);*/

	glm::mat4 test_matrix(0.993924, -0.085335, 0.0695257, 0,
					0.0264946, 0.798538, 0.601362,0,
					-0.106836, -0.595866, 0.795946, 0,
					0.00770066,  0.036563, -0.00742521, 1);


	Eigen::Affine3f a_mat;
	a_mat = pose_result;
	gl_reference_eval->setModelmatrix(MatrixUtils::Affine3f2Mat4(a_mat));

	
	

	gl_camera_point_cloud = new	isu_ar::GLPointCloudRenderer(pc_camera.points, pc_camera.normals);
	gl_camera_point_cloud->setPointColor(glm::vec3(1.0,0.0,0.0));
	gl_camera_point_cloud->setNormalColor(glm::vec3(0.8,0.5,0.0));
	gl_camera_point_cloud->setNormalGfxLength(0.02f);
	gl_camera_point_cloud->enablePointRendering(true);


	// line renderer for debugging nearest neighbors
	gl_knn_lines = new isu_ar::GLLineRenderer(pc_ref.points, pc_camera.points, icp->getNN());
	gl_knn_lines->updatePoints();


	//thread_1.lock(); // block the start until the render window is up
	//std::thread test_run_static(runTest);  

	//Sleep(100);
	// start the window along with  point cloud processing
	window->start();



	// delete all instances. 
	delete window;
	//test_run_static.detach();

}




