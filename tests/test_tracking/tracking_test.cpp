/*This file implements tracking test. This includes object detection using feature matching  & registration using ICP. 

Version 1.0


Sindhura Challa
Iowa State University 
lchalla@iastate.edu
August 2020


Last Edited: Aug 3rd, 2020. 
*/ 


// GLM include files
#define GLM_FORCE_INLINE
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtx/transform.hpp>  // transformation
#include <glm/gtx/quaternion.hpp> // quaternions

// STL
#include <iostream>
#include <string>
#include <Windows.h>
#include <fstream>

// TrackingExpert
#include "trackingx.h"
#include "graphicsx.h"
#include "TrackingMain.h" 
#include "ReaderWriterOBJ.h"
#include "ReaderWriterPLY.h"

using namespace texpert;

// The OpenGL window
isu_ar::GLViewer* window;

// OpenGL point cloud objects showing the point cloud
isu_ar::GLPointCloudRenderer* gl_camera_point_cloud;
isu_ar::GLPointCloudRenderer* gl_reference_point_cloud;
isu_ar::GLPointCloudRenderer* gl_reference_eval; // evaluation point cloud for visual evaluation

// Instance of a PointCloudProducer
// It creates a 3D point cloud using the given depth image. 
texpert::PointCloudProducer*			producer;

// helper variable to enable/disable the gl renderer. 
int							gl_normal_rendering = 0;
int							gl_ref_rendering = 1;
int							gl_nn_rendering = 0;

// Helper variables to set the point cloud sampling. 
SamplingParam		sampling_param;
SamplingMethod		sampling_method;


// The reference point cloud.
// The first one is the point cloud for all detection & ICP purposes.
// The second one is the raw point cloud as loaded. 
PointCloud			pc_ref;
PointCloud			pc_ref_as_loaded;

// The test point cloud
// The first one is the point cloud for all detection & ICP purposes.
// The second one is the raw point cloud as loaded. 
PointCloud			pc_camera;
PointCloud			pc_camera_as_loaded;


// matching
FDMatching* _feature_matching;
double				_distance_step;
double				_angle_step;


// variable to store the files. 
std::string			ref_file = "../data/stanford_bunny_pc.obj";
std::string			camera_file = "../data/test/test-6.obj";

string date = "";

/*
The main render and processing loop. 
@param pm - projection matrix
@param vm - camera view matrix. 
*/
void render_loop(glm::mat4 pm, glm::mat4 vm) {

	//-----------------------------------------------------------
	// Rendering
	//gl_reference_point_cloud->updatePoints();
	gl_reference_point_cloud->draw(pm, vm);

	//gl_camera_point_cloud->updatePoints();
	gl_camera_point_cloud->draw(pm, vm);

	gl_reference_eval->draw(pm, vm);

	
	Sleep(25);
}




int main(int argc, char** argv)
{
	bool err = false;
	date = TimeUtils::GetCurrentDateTime();

	_feature_matching = new FDMatching();


	/*------------------------------------------------------------------------
	Load a reference point cloud and downsample it
	*/
	sampling_method = SamplingMethod::UNIFORM;
	sampling_param.grid_x = 0.005;
	sampling_param.grid_y = 0.005;
	sampling_param.grid_z = 0.005;
	Sampling::SetMethod(sampling_method, sampling_param);

	LoaderObj::Read(ref_file, &pc_ref_as_loaded.points, &pc_ref_as_loaded.normals, false, true);
	Sampling::Run(pc_ref_as_loaded, pc_ref_as_loaded);
	pc_ref = pc_ref_as_loaded;

	// init feature matching
	_feature_matching->setAngleStep(15.0);
	_feature_matching->setDistanceStep(0.05);
	_feature_matching->invertPose(false);
	_feature_matching->setVerbose(true);
	_feature_matching->extract_feature_map(&pc_ref.points, &pc_ref.normals);
	_feature_matching->setClusteringThreshold(15.0, 5.0);
	
	LoaderObj::Read(camera_file, &pc_camera_as_loaded.points, &pc_camera_as_loaded.normals, false, true);
	Sampling::Run(pc_camera_as_loaded, pc_camera_as_loaded);
	pc_camera = pc_camera_as_loaded;

	
	// run the matching algorithm

	vector<Pose> poses;
	_feature_matching->detect(&pc_camera.points, &pc_camera.normals, poses);
	

	// final result
	glm::mat4 m = MatrixUtils::Affine3f2Mat4(poses[0].t);
	
/*-------------------------------------------------------------------------------------
	create the renderer.
	The renderer executes the main loop in this demo.
	*/
	glm::mat4 vm = glm::lookAt(glm::vec3(0.0f, 0.0, -0.5f), glm::vec3(0.0f, 0.0f, 0.5), glm::vec3(0.0f, 1.0f, 0.0f));
	window = new isu_ar::GLViewer();
	window->create(1280, 1280, "Tracking test");
	window->addRenderFcn(render_loop);
	window->setViewMatrix(vm);
	window->setClearColor(glm::vec4(1, 1, 1, 1));
	window->enableCameraControl(true);


	/*
	Create the content
	*/
	// Point cloud showing the reference model.
	gl_reference_point_cloud = new	isu_ar::GLPointCloudRenderer(pc_ref.points, pc_ref.normals);
	gl_reference_point_cloud->setPointColor(glm::vec3(0.0, 1.0, 0.0));
	gl_reference_point_cloud->setNormalColor(glm::vec3(0.0, 0.8, 0.8));
	gl_reference_point_cloud->setNormalGfxLength(0.02f);
	gl_reference_point_cloud->enablePointRendering(false);

	gl_reference_eval = new	isu_ar::GLPointCloudRenderer(pc_ref.points, pc_ref.normals);
	gl_reference_eval->setPointColor(glm::vec3(1.0, 1.0, 0.0));
	gl_reference_eval->setNormalColor(glm::vec3(0.5, 0.8, 0.8));
	gl_reference_eval->setNormalGfxLength(0.02f);
	gl_reference_eval->enablePointRendering(true);
	gl_reference_eval->setModelmatrix(m);
	

	gl_camera_point_cloud = new	isu_ar::GLPointCloudRenderer(pc_camera.points, pc_camera.normals);
	gl_camera_point_cloud->setPointColor(glm::vec3(1.0, 0.0, 0.0));
	gl_camera_point_cloud->setNormalColor(glm::vec3(0.8, 0.5, 0.0));
	gl_camera_point_cloud->setNormalGfxLength(0.02f);
	gl_camera_point_cloud->enablePointRendering(true);

	
	// start the window along with  point cloud processing
	window->start();

	// delete all instances. 
	delete window;


}





