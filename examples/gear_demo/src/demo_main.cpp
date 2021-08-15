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
#include "ProcedureRenderer.h"

#define PI 3.1415926535

// instance of a structure core camera 
KinectAzureCaptureDevice* camera;

// The OpenGL window
isu_ar::GLViewer* window;


// demo content 
DemoScene* gear;

PartDatabase* database;
GearBoxRenderer* renderer;

ProcedureRenderer* proc_renderer;

// The Video Background
isu_gfx::GLVideoCanvas*	video_bg;
cv::Mat img_color;
cv::Mat img_ref;

/*
The main render and processing loop. 
@param pm - projection matrix
@param vm - camera view matrix. 
*/
void render_loop(glm::mat4 pm, glm::mat4 vm) {

	// fetch a new frame	
	camera->getRGBFrame(img_ref);
	memcpy(img_color.ptr(), img_ref.ptr(), img_ref.rows * img_ref.cols * sizeof(CV_8UC4));

	video_bg->draw(pm, vm, glm::mat4());

	//renderer->draw(pm, vm);
	proc_renderer->draw(pm, vm);
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
			//renderer->progress(true);
			proc_renderer->progress(true);
			break;
		case 65: //a
			//renderer->progress(false);
			proc_renderer->progress(false);
			break;
		}
		break;
	}
}


int main(int argc, char* argv)
{
	std::cout << "TrackingExpert+ Gear Box Demo" << endl;
	std::cout << "Version 0.9" << endl;
	std::cout << "-----------------------------" << endl;

	/*
	Open a camera device. 
	*/
	camera = new KinectAzureCaptureDevice(0, KinectAzureCaptureDevice::Mode::RGBIRD, false);

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
	window->create(camera->getCols(texpert::CaptureDeviceComponent::COLOR), camera->getRows(texpert::CaptureDeviceComponent::COLOR), "Gear Box Demo");
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
	video_bg->create(img_color.rows, img_color.cols, img_color.ptr(), true);

	/*
	Init procedure renderer
	*/
	std::vector<std::string> steps = std::vector<std::string>(18);

	steps.at(0) = "Base";
	steps.at(1) = "P1Base";
	steps.at(2) = "P1Bearing1";
	steps.at(3) = "P1Bearing2";
	steps.at(4) = "P1Cap";
	steps.at(5) = "Part1";
	steps.at(6) = "P2Base";
	steps.at(7) = "P2Bearing1";
	steps.at(8) = "P2Cap";
	steps.at(9) = "P2Gear";
	steps.at(10) = "P2Bearing2";
	steps.at(11) = "Part2";
	steps.at(12) = "P3Base";
	steps.at(13) = "P3Bearing1";
	steps.at(14) = "P3Cap";
	steps.at(15) = "P3BigGear";
	steps.at(16) = "P3Bearing2";
	steps.at(17) = "Part3";

	proc_renderer = new ProcedureRenderer();
	proc_renderer->init("ExGearProc.json", steps);

	///*
	//Load part models
	//*/
	//database = new PartDatabase();
	//database->loadObjsFromFile("D:/WorkRepos/TrackingExpertPlus/examples/gear_demo/models/load_models.txt");

	//database->setNumDuplicates("N1-002_pc_gfx.obj", 1);
	//database->setNumDuplicates("N3-002_pc_gfx.obj", 1);

	///*
	//Set part model positions
	//*/
	//database->setModelPos("N1-001_pc_gfx.obj", glm::rotate(glm::translate(glm::vec3(0.5f, -0.3f, -0.12f)), (float)PI / 2, glm::vec3(0, 0, 1)));
	//database->setModelPos("N1-002_pc_gfx.obj", glm::rotate(glm::translate(glm::vec3(0.5f, -0.3f, -0.24f)), (float)PI / 2, glm::vec3(0, 0, 1)));
	//database->setModelPos("N1-002_pc_gfx.obj-1", glm::rotate(glm::translate(glm::vec3(0.2f, -0.3f, -0.24f)), (float)PI / 2, glm::vec3(0, 0, 1)));
	//database->setModelPos("N1-003_pc_gfx.obj", glm::translate(glm::vec3(0.6f, -0.3f, -0.24f)));

	//database->setModelPos("N4-001_pc_gfx.obj", glm::rotate(glm::translate(glm::vec3(-0.4f, -0.3f, -0.26f)), (float)-PI / 2, glm::vec3(0, 0, 1)));
	//database->setModelPos("N4-002_pc_gfx.obj", glm::rotate(glm::translate(glm::vec3(-0.15f, -0.3f, -0.26f)), (float)-PI / 2, glm::vec3(0, 0, 1)));
	//database->setModelPos("N3-002_pc_gfx.obj", glm::rotate(glm::translate(glm::vec3(-0.65f, -0.3f, -0.26f)), (float)-PI, glm::vec3(0, 0, 1)));
	//database->setModelPos("N4-003_pc_gfx.obj", glm::rotate(glm::translate(glm::vec3(-0.45f, -0.3f, -0.26f)), (float)-PI / 2, glm::vec3(0, 0, 1)));
	//database->setModelPos("N4-004_pc_gfx.obj", glm::rotate(glm::translate(glm::vec3(-0.3f, -0.3f, -0.26f)), (float)-PI / 2, glm::vec3(0, 0, 1)));

	//database->setModelPos("N2-001_pc_gfx.obj", glm::rotate(glm::translate(glm::vec3(-0.25f, -0.3f, 0.43f)), (float)PI / 2, glm::vec3(0, 0, 1)));
	//database->setModelPos("N3-001_pc_gfx.obj", glm::rotate(glm::translate(glm::vec3(-0.55f, -0.3f, 0.25f)), (float)PI / 2, glm::vec3(0, 0, 1)));
	//database->setModelPos("N3-002_pc_gfx.obj-1", glm::rotate(glm::translate(glm::vec3(-0.65f, -0.3f, 0.25f)), (float)PI, glm::vec3(0, 0, 1)));
	//database->setModelPos("N2-002_pc_gfx.obj", glm::rotate(glm::translate(glm::vec3(0.33f, -0.3f, 0.25f)), (float)PI / 2, glm::vec3(0, 0, 1)));
	//database->setModelPos("N2-003_pc_gfx.obj", glm::rotate(glm::translate(glm::vec3(0.48f, -0.3f, 0.25f)), (float)PI / 2, glm::vec3(0, 0, 1)));

	///*
	//Load models into the GearBoxRenderer sequence
	//*/
	//renderer = new GearBoxRenderer();
	//int idx = 0;
	//for (int i = 0; i < database->getNumModels(); i++)
	//{
	//	Model* curModel = database->getObj(i);
	//	if (!curModel->name.compare("null") == 0)
	//	{
	//		renderer->addModel(curModel, curModel->name);
	//		idx++;
	//	}
	//}

	//renderer->setSteps();
	//
	//renderer->updateInPlace();

	std::cout << "-----------------------------" << endl;
	std::cout << "Use the W and D keys to cycle through the stages of the assembly process." << endl;
	
	window->start();

	// cleanup
	delete camera;
	delete video_bg;
	delete database;
	delete window;

	return 1;
}