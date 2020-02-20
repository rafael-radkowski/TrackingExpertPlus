/*
 Camera and point cloud test.

 This code implements a test to verify the camera and point cloud producer 
 functionality. Currently, a Structure Core camera is used to acquire 
 depth data. The PointCloudProducer instance processes the depth data
 and generates a 3D point cloud. 

 The entire PointCloudProducer instance works with Nvidia Cuda. Thus,
 the code tests whether or not sufficient Cuda capabilities are given. 

 Rafael Radkowski
 Iowa State University
 Oct 2019
 rafael@iastate.edu
 +1 (515) 294 7044

-----------------------------------------------------------
Last edits:

*/



// stl
#include <iostream>
#include <string>
#include <Windows.h>
#include <chrono>

// GLM include files
#define GLM_FORCE_INLINE
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtx/transform.hpp>  // transformation
#include <glm/gtx/quaternion.hpp> // quaternions



// local
#include "StructureCoreCaptureDevice.h"  // the camera
#include "Types.h"  // PointCloud data type
#include "cuda/cuPCU3f.h"  // point cloud samping
#include "PointCloudProducer.h"

// graphics
#include "GLViewer.h"
#include "GLPointCloud.h"
#include "GLPointCloudRenderer.h"

#define WITH_PERFORMANCE_MEASUREMENT

using namespace texpert;

// instance of a structure core camera 
texpert::StructureCoreCaptureDevice* camera;

// Instance of a PointCloudProducer
// It creates a 3D point cloud using the given depth image. 
texpert::PointCloudProducer* producer;

// Global storage for the point cloud data. 
PointCloud					camera_point_cloud;

// The OpenGL window
isu_ar::GLViewer* window;

// the point cloud renderers
isu_ar::GLPointCloudRenderer* gl_point_cloud;


#ifdef WITH_PERFORMANCE_MEASUREMENT
double	avg_all = 0.0;
double	avg_process = 0.0;
double	avg_rendering = 0.0;
int		frame_count = 0;
#endif 

/*
The main render and processing loop. 
@param pm - projection matrix
@param vm - camera view matrix. 
*/
void render_loop(glm::mat4 pm, glm::mat4 vm) {

#ifdef WITH_PERFORMANCE_MEASUREMENT
	 auto time0 = std::chrono::high_resolution_clock::now();
#endif 

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

#ifdef WITH_PERFORMANCE_MEASUREMENT
	auto time1 = std::chrono::high_resolution_clock::now();
#endif

	// process the point cloud. 
	// Note that the point cloud producer only creates a new point
	// cloud if the images are new. It skips the step otherwise. 
	producer->process();

#ifdef WITH_PERFORMANCE_MEASUREMENT
	auto time2 = std::chrono::high_resolution_clock::now();
#endif

	// Update the opengl points and draw the points. 
	gl_point_cloud->updatePoints();
	gl_point_cloud->draw(pm, vm);


#ifdef WITH_PERFORMANCE_MEASUREMENT
	auto time3 = std::chrono::high_resolution_clock::now();

	std::chrono::duration<double> all = time3-time0;
	std::chrono::duration<double> processing = time2-time1;
	std::chrono::duration<double> rendering = time3-time2;

	frame_count++;
	avg_all = avg_all * (frame_count-1)/frame_count + all.count() /frame_count;
	avg_process = avg_process * (frame_count-1)/frame_count + processing.count() /frame_count;
	avg_rendering = avg_rendering * (frame_count-1)/frame_count + rendering.count() /frame_count;

	if (frame_count % 60 == 0) {
		cout << "Avg. time, all: " << avg_all << " s, point cloud processing: " << avg_process << " s, rendering: " << avg_rendering << " s, " << 1.0/avg_all << " fps" << endl;
	}

#endif
	
	Sleep(10);
}





int main(int argc, char* argv)
{

	/*
	Open a camera device. 
	*/
	camera =  new texpert::StructureCoreCaptureDevice();

	/*
	Create a point cloud producer. It creates a point cloud 
	from a depth map. Assign a camera and the location to store the point cloud data. 
	*/
	producer = new texpert::PointCloudProducer(*camera, camera_point_cloud);

	/*
	Set the sampling parameters. 
	The producer instance can provide the full frame (RAW), a uniformly smapled 
	point cloud (UNIFORM), and a randomly sampled (RANDOM) point cloud.
	*/
	SamplingParam sParam;
	sParam.uniform_step = 2;
	sParam.random_max_points = 5000;
	producer->setSampingMode(UNIFORM, sParam);


	/*
	create the renderer.
	The renderer executes the main loop in this demo. 
	*/
	window = new isu_ar::GLViewer();
	window->create(1280, 1280, "Camera test and point cloud test");
	window->addRenderFcn(render_loop);
	//window->addKeyboardCallback(std::bind(&FMEvalApp::keyboard_callback, this, _1, _2));
	window->setViewMatrix(glm::lookAt(glm::vec3(1.0f, 0.0, -5.5f), glm::vec3(0.0f, 0.0f, 3.f), glm::vec3(0.0f, 1.0f, 0.0f)));
	window->setClearColor(glm::vec4(1, 1, 1, 1));

	/*
	Create an OpenGL 3D point cloud.
	This instance draws a point cloud. 
	*/
	gl_point_cloud = new isu_ar::GLPointCloudRenderer(camera_point_cloud.points, camera_point_cloud.normals);


	/*
	Test if the camera is ready to run. 
	*/
	if (!camera->isOpen()) {
		std::cout << "\n[ERROR] - Cannot access camera." << std::endl;
		return -1;
	}

	// start the window along with  point cloud processing
	window->start();

	// delete all instances. 
	delete window;
	delete camera;

	return 1;
}