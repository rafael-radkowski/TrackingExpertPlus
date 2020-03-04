/*
 Gear transmission box demo.

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

-----------------------------------------------------------------




*/
// STL
#include <iostream>
#include <string>
#include <fstream>

// TrackingExpert
#include "trackingx.h"
#include "graphicsx.h"

// local
#include "DemoScene.h"

// instance of a structure core camera 
texpert::StructureCoreCaptureDevice* camera;

// The OpenGL window
isu_ar::GLViewer* window;


// demo content 
DemoScene* gear;


// The Video Background
isu_gfx::GLVideoCanvas*	video_bg;
cv::Mat img_color;

/*
The main render and processing loop. 
@param pm - projection matrix
@param vm - camera view matrix. 
*/
void render_loop(glm::mat4 pm, glm::mat4 vm) {

	// fetch a new frame	
	camera->getRGBFrame(img_color);



	video_bg->draw(pm, vm, glm::mat4());
	gear->draw(pm, vm);
}


int main(int argc, char* argv)
{
	std::cout << "TrackingExpert+ Gear Box Demo" << endl;
	std::cout << "Version 0.9" << endl;


	/*
	Open a camera device. 
	*/
	camera =  new texpert::StructureCoreCaptureDevice();

	/*
	Test if the camera is ready to run. 
	*/
	if (!camera->isOpen()) {
		std::cout << "\n[ERROR] - Cannot access camera." << std::endl;
		return -1;
	}


	
	/*
	create the renderer.
	The renderer executes the main loop in this demo. 
	*/
	window = new isu_ar::GLViewer();
	window->create(1280, 960, "Gear Box Demo");
	window->addRenderFcn(render_loop);
	//window->addKeyboardCallback(std::bind(&FMEvalApp::keyboard_callback, this, _1, _2));
	window->setViewMatrix(glm::lookAt(glm::vec3(1.0f, 0.0, -0.5f), glm::vec3(0.0f, 0.0f, 0.f), glm::vec3(0.0f, 1.0f, 0.0f)));
	window->setClearColor(glm::vec4(1, 1, 1, 1));
	window->enableCameraControl(true);
	
	/*
	Create the video background
	*/
	camera->getRGBFrame(img_color);
	video_bg = new isu_gfx::GLVideoCanvas();
	video_bg->create(img_color.rows,  img_color.cols, (unsigned char*)img_color.data, true);

	
	gear = new DemoScene();
	gear->create();
	
	
	
	window->start();

	// cleanup
	delete camera;
	delete video_bg;
	delete window;

	return 1;
}