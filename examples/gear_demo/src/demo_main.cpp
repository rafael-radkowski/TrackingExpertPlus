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
#include "ICaptureDevice.h"

#include "KinectAzureCaptureDevice.h"  // the camera
#include "PointCloudProducer.h"
#define _WITH_PRODUCER

// local
#include "DemoScene.h"
#include "GearBoxRenderer.h"
#include "PartDatabase.h"

// instance of a structure core camera 
KinectAzureCaptureDevice* camera;

// The OpenGL window
isu_ar::GLViewer* window;


// demo content 
DemoScene* gear;

PartDatabase* database;
GearBoxRenderer* renderer;

// The Video Background
isu_gfx::GLVideoCanvas*	video_bg;
cv::Mat img_color;
cv::Mat img_resized;

/*
The main render and processing loop. 
@param pm - projection matrix
@param vm - camera view matrix. 
*/
void render_loop(glm::mat4 pm, glm::mat4 vm) {

	// fetch a new frame	

	//imshow("Hej", img_color);

	camera->getRGBFrame(img_color);

	//TODO: Get the video to display on the background
	/*
	The issue may involve either my graphics driver or the card itself (NVIDIA GeForce 970M).
	The function glTextureSubImage2D is included in a OGL extension package called ARB_direct_state_access
	that is only used in GL version 4.5 and above.  For either of the aforementioned reasons (due to the 
	age of either) or for reasons relating to my software fluency, GLEW insists that this extension 
	package, along with GL versions 4.5 and 4.6, does not exist.  I am still working on a solution for
	my end, but otherwise, it seems like this function should work on more modern graphics cards.
	*/
	video_bg->draw(pm, vm, glm::mat4(1.0f));

	//gear->draw(pm, vm);
	renderer->draw(pm, vm);
}

void getKey(int key, int action)
{
	switch (action)
	{
	case 0: //Key up
		break;

	case 1: //Key down
		switch (key)
		{
		case 68: //d
			renderer->progress(true);
			break;
		case 65: //a
			renderer->progress(false);
			break;
		}
		break;
	}
}


int main(int argc, char* argv)
{
	std::cout << "TrackingExpert+ Gear Box Demo" << endl;
	std::cout << "Version 0.9" << endl;

	/*
	Open a camera device. 
	*/
	camera =  new KinectAzureCaptureDevice();
	//img_color = cv::Mat(camera->getRows(texpert::CaptureDeviceComponent::COLOR), camera->getCols(texpert::CaptureDeviceComponent::COLOR), CV_8UC4);

	/*
	Test if the camera is ready to run. 
	*/
	if (!camera->isOpen()) {
		std::cout << "\n[ERROR] - Cannot access camera." << std::endl;
		return -1;
	}

	//camera->changeResolution(1536);

	/*
	create the renderer.
	The renderer executes the main loop in this demo. 
	*/
	window = new isu_ar::GLViewer();
	window->create(1280, 960, "Gear Box Demo");
	window->addRenderFcn(render_loop);
	window->addKeyboardCallback(getKey);
	window->setViewMatrix(glm::lookAt(glm::vec3(1.0f, 0.0, -0.5f), glm::vec3(0.0f, 0.0f, 0.f), glm::vec3(0.0f, 1.0f, 0.0f)));
	window->setClearColor(glm::vec4(1, 1, 1, 1));
	window->enableCameraControl(true);
	
	/*
	Create the video background
	*/
	camera->getRGBFrame(img_color);
	video_bg = new isu_gfx::GLVideoCanvas();
	video_bg->create(img_color.rows,  img_color.cols, (unsigned char*)img_color.data, true);

	/*
	Load part models
	*/
	database = new PartDatabase();
	database->loadObjsFromFile("D:/WorkRepos/TrackingExpertPlus/examples/gear_demo/models/load_models.txt");
	//database->loadObjsFromFile("D:/noPath.txt");  //This is meant to break 

	/*
	Load models into the GearBoxRenderer sequence
	*/
	renderer = new GearBoxRenderer();
	int idx = 0;
	for (int i = 0; i < database->getNumModels(); i++)
	{
		Model* curModel = database->getObj(i);
		if (!curModel->name.compare("null") == 0)
		{
			renderer->addModel(curModel, curModel->name);
			idx++;
		}
	}

	//gear = new DemoScene();
	//gear->create();
	
	renderer->updateInPlace();
	
	window->start();

	// cleanup
	delete camera;
	delete video_bg;
	delete window;

	return 1;
}