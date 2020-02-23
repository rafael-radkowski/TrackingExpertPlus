


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
#include "./camera/StructureCoreCaptureDevice.h"  // the camera
#include "trackingx.h"
#include "graphicsx.h"
#include "TrackingMain.h" 
#include "ReaderWriterOBJ.h"
#include "ReaderWriterPLY.h"

using namespace texpert;

#define WITH_STATIC_TEST

// instance of a structure core camera 
texpert::StructureCoreCaptureDevice*	camera;

// Instance of a PointCloudProducer
// It creates a 3D point cloud using the given depth image. 
texpert::PointCloudProducer*			producer;

// Point cloud sampling parameters
SamplingParam							sParam;



// The OpenGL window
isu_ar::GLViewer* window;
isu_ar::GLPointCloudRenderer* gl_camera_point_cloud;
isu_ar::GLPointCloudRenderer* gl_reference_point_cloud;
isu_ar::GLPointCloudRenderer* gl_reference_eval; // evaluation point cloud for visual evaluation

int							gl_normal_rendering = 0;

TrackingMain*		tm;

SamplingParam		sampling_param;
SamplingMethod		sampling_method;


// The reference point cloud
PointCloud			pc_ref;
PointCloud			pc_ref_as_loaded;

// The test point cloud
PointCloud			pc_camera;
PointCloud			pc_camera_as_loaded;



// Registrations
PCRegistratation*	pc_reg;

//#define _DEBUG
//#ifdef _DEBUG
//std::string			ref_file = "../data/stanford_bunny_pc.obj";
//std::string			camera_file = "../data/stanford_bunny_pc.obj";
//#else
//std::string			ref_file = "../data/tracking/N0-000_pc.obj";
std::string			ref_file = "../data/stanford_bunny_pc.obj ";
std::string			camera_file = "../data/test/bunny_2_2020-02-21_06-56-55_pc.obj";
//#endif

int file_counter = 1;
string date = "";

void keyboard_callback( int key, int action) {


	cout << key << " : " << action << endl;
	switch (action) {
	case 0:  // key up
	
		switch (key) {
		case 87: // w
		{
			string file = "bunny_";
			file.append(to_string(file_counter));
			file.append("_");
			file.append(date);
			file.append("_pc.obj");
			
			file_counter++;

			ReaderWriterOBJ::Write(file, pc_camera.points, pc_camera.normals);
			ReaderWriterPLY::Write(file, pc_camera.points, pc_camera.normals);
			break;
			} 
		case 78: // n
			{
			gl_normal_rendering = (++gl_normal_rendering)%2;
			gl_camera_point_cloud->enableNormalRendering((gl_normal_rendering==1)? true :false);
			gl_reference_point_cloud->enableNormalRendering((gl_normal_rendering==1)? true :false);

			break;
			}
		}
		break;
		
		case 1: // key down

			break;
	}
}


/*
The main render and processing loop. 
@param pm - projection matrix
@param vm - camera view matrix. 
*/
void render_loop(glm::mat4 pm, glm::mat4 vm) {

#ifdef WITH_STATIC_TEST

	//bool err = pc_reg->process(pc_camera);

#else
	
	// get the latest camera images
	// Note that the camera works asynchronously. 
	// The images are the two last one the camera obtained. 
	cv::Mat img_color;
	cv::Mat img_depth;
	camera->getRGBFrame(img_color);
	camera->getDepthFrame(img_depth);

	// render the image. 
	cv::imshow("RGB", img_color);
	cv::imshow("Depth", img_depth);

	// Acquire new camera image
	producer->process();

	// track
	bool err = pc_reg->process(pc_camera);

#endif

	// get the objects back
	std::vector<glm::mat4> pose = pc_reg->getGlPoses();
	

	//-----------------------------------------------------------
	// Rendering
	//gl_reference_point_cloud->updatePoints();
	gl_reference_point_cloud->draw(pm, vm);

	//gl_camera_point_cloud->updatePoints();
	gl_camera_point_cloud->draw(pm, vm);

	if(pose.size() > 0){
		gl_reference_eval->setModelmatrix(pose[0]);
		//gl_reference_eval->updatePoints();
	}
	gl_reference_eval->draw(pm, vm);

	
	Sleep(25);
}




int main(int argc, char** argv)
{
	bool err = false;
	date = TimeUtils::GetCurrentDateTime();

	pc_reg = new PCRegistratation();


	/*------------------------------------------------------------------------
	Load a reference point cloud and downsample it
	*/
	sampling_method = SamplingMethod::UNIFORM;
	sampling_param.grid_x = 0.01;
	sampling_param.grid_y = 0.01;
	sampling_param.grid_z = 0.01;
	Sampling::SetMethod(sampling_method, sampling_param);

	LoaderObj::Read(ref_file, &pc_ref_as_loaded.points, &pc_ref_as_loaded.normals, false, true);
	Sampling::Run(pc_ref_as_loaded, pc_ref_as_loaded);
	pc_ref = pc_ref_as_loaded;


	// Move the point cloud to a different position. 
	PointCloudTransform::Transform(&pc_ref,  Eigen::Vector3f(0.0, 0.2, 0.0), Eigen::Vector3f(0.0, 0.0, 90.0));

	
	/*------------------------------------------------------------------------
	Add the point cloud to the registration unit
	*/
	err = pc_reg->addReferenceObject(pc_ref);
	if(!err) cout << "[ERROR] - Did not add reference point set." << endl; 


#ifdef WITH_STATIC_TEST
	// the "static test" loads a point cloud from a file

	LoaderObj::Read(camera_file, &pc_camera_as_loaded.points, &pc_camera_as_loaded.normals, false, true);
	Sampling::Run(pc_camera_as_loaded, pc_camera_as_loaded);
	pc_camera = pc_camera_as_loaded;

#else 
	/*
	Open a camera device. 
	*/
	camera =  new texpert::StructureCoreCaptureDevice();
	camera->readCameraParameters("../data/camera_param/StructureCore_default_params.json", true);


	// start the point cloud producer
	producer = new texpert::PointCloudProducer(*camera, pc_camera);

	/* Set the sampling parameters. 
	The producer instance can provide the full frame (RAW), a uniformly smapled 
	point cloud (UNIFORM), and a randomly sampled (RANDOM) point cloud. */
	sParam.uniform_step = 6;
	sParam.random_max_points = 5000;
	producer->setSampingMode(UNIFORM, sParam);
#endif

	/*
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
	Create the content
	*/
	gl_reference_point_cloud = new	isu_ar::GLPointCloudRenderer(pc_ref.points, pc_ref.normals);
	gl_reference_point_cloud->setPointColor(glm::vec3(0.0,1.0,0.0));
	gl_reference_point_cloud->setNormalColor(glm::vec3(0.0,0.8,0.8));

	gl_reference_eval = new	isu_ar::GLPointCloudRenderer(pc_ref.points, pc_ref.normals);
	gl_reference_eval->setPointColor(glm::vec3(1.0,1.0,0.0));
	gl_reference_eval->setNormalColor(glm::vec3(0.5,0.8,0.8));
	
#ifdef WITH_STATIC_TEST
	gl_camera_point_cloud = new	isu_ar::GLPointCloudRenderer(pc_camera.points, pc_camera.normals, texpertgfx::DYNAMIC);
	gl_camera_point_cloud->setPointColor(glm::vec3(1.0,0.0,0.0));
	gl_camera_point_cloud->setNormalColor(glm::vec3(0.8,0.5,0.0));
#else
	
	// the point cloud renderer
	gl_camera_point_cloud = new isu_ar::GLPointCloudRenderer(pc_camera.points, pc_camera.normals);
	gl_camera_point_cloud->setPointColor(glm::vec3(1.0,0.0,0.0));

	/*
	Test if the camera is ready to run. 
	*/
	if (!camera->isOpen()) {
		std::cout << "\n[ERROR] - Cannot access camera." << std::endl;
		return -1;
	}

#endif


	// start the window along with  point cloud processing
	window->start();

	// delete all instances. 
	delete window;
	delete camera;


}





