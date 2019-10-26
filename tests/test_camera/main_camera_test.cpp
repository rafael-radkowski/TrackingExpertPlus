/*
 Camera and point cloud test.



*/

// stl
#include <iostream>
#include <string>

// GLM include files
#define GLM_FORCE_INLINE
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtx/transform.hpp>  // transformation
#include <glm/gtx/quaternion.hpp> // quaternions



// local
#include "StructureCoreCaptureDevice.h"  // the camera
#include "Types.h"
#include "cuda/cuPCU3f.h"  // point cloud samping

// graphics
#include "GLViewer.h"
#include "GLPointCloud.h"
#include "GLPointCloudRenderer.h"

using namespace texpert;

texpert::StructureCoreCaptureDevice* camera;


isu_ar::GLViewer* window;

// the point cloud renderers
isu_ar::GLPointCloudRenderer* gl_point_cloud;

PointCloud					camera_point_cloud;


void camera_callback(void) {
	
	cv::Mat img_color;
	cv::Mat img_depth;
	camera->getRGBFrame(img_color);
	camera->getDepthFrame(img_depth);

	cv::imshow("RGB", img_color);
	cv::imshow("Depth", img_depth);
	
	cv::waitKey(1);
}



/*
Process the uniform sampling point cloud generation
*/
void  process_uniform_sampling(float* imgBuf)
{

	//vector<float3> points_host(640 * 480);
	//vector<float3> normals_host(640 * 480);

	camera_point_cloud.resize(640*480);
	/*cuPCU3f::CreatePointCloud(	(float*)imgBuf, 640, 480, 1, 320.0, 320.0, 0.0, 0.0, 4, 
								(vector<float3>&)camera_point_cloud.points, 
								(vector<float3>&)camera_point_cloud.normals);*/

	
	cuSample3f::UniformSampling((float*)imgBuf, 640, 480, 320.0, 320.0, 0.0, 0.0, 4, false,
								(vector<float3>&)camera_point_cloud.points, 
								(vector<float3>&)camera_point_cloud.normals);
	

}


void render_loop(glm::mat4 pm, glm::mat4 vm) {

	cv::Mat img_color;
	cv::Mat img_depth;
	camera->getRGBFrame(img_color);
	camera->getDepthFrame(img_depth);

	cv::imshow("RGB", img_color);
	cv::imshow("Depth", img_depth);


	// process points
	process_uniform_sampling( (float*)img_depth.data);

	gl_point_cloud->updatePoints();
	gl_point_cloud->draw(pm, vm);
}





int main(int argc, char* argv)
{
	camera =  new texpert::StructureCoreCaptureDevice();


	// create the renderer
	window = new isu_ar::GLViewer();
	window->create(1768, 2024, "Camera test and point cloud test");
	window->addRenderFcn(render_loop);
	//window->addKeyboardCallback(std::bind(&FMEvalApp::keyboard_callback, this, _1, _2));
	window->setViewMatrix(glm::lookAt(glm::vec3(0.0f, 0.0, -3.5f), glm::vec3(0.0f, 0.0f, 00.f), glm::vec3(0.0f, 1.0f, 0.0f)));
	window->setClearColor(glm::vec4(1, 1, 1, 1));


	gl_point_cloud = new isu_ar::GLPointCloudRenderer(camera_point_cloud.points, camera_point_cloud.normals);




	int w = 640;
	int h = 480;

	// Allocate device memory
	cuPCU3f::AllocateDeviceMemory(w, h, 1);   

	// set sampling parameters
	cuSample3f::CreateUniformSamplePattern(w, h, 4);
	//cuSample3f::CreateRandomSamplePattern(w, h, 20);


	if (!camera->isOpen()) {
		std::cout << "\n[ERROR] - Cannot access camera." << std::endl;
		return -1;
	}


	window->start();


	delete window;
	delete camera;

	return 1;
}