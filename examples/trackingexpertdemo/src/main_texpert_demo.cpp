// stl
#include <iostream>
#include <string>
#include <vector>

// local
#include "ArgParser.h"
#include "TrackingExpertDemo.h"

using namespace texpert;

int main(int argc, char** argv) {

	std::cout << "TrackingExpertPlusDemo" << std::endl;
	std::cout << "V0.5-20200803, Aug. 2020\n" << std::endl;
	std::cout << "--------------------------------------------------------------------------" << std::endl;
	std::cout << "The demo demonstrates the use of TrackingExpert+" << std::endl;
	std::cout << "It was prepared to track one object given a 3D point cloud model as input.\n" << std::endl;

	std::cout << "Rafael Radkowski" << std::endl;
	std::cout << "Iowa State University" << std::endl;
	std::cout << "rafael@iastate.edu" << std::endl;
	std::cout << "MIT License" << std::endl;
	std::cout << "--------------------------------------------------------------------------\n" << std::endl;

	// parse the command line arguments
	Arguments params = ArgParser::Parse(argc, argv);

	CaptureDeviceType type = CaptureDeviceType::None;
	if (params.camera_type.compare("AzureKinect") == 0) type = KinectAzure;

	
	// Start the demo
	TrackingExpertDemo* demo = new TrackingExpertDemo();
	demo->setVerbose(params.verbose);
	demo->setCamera(type); 
	demo->loadScene(params.scene_file); // ignored when a camera is set.
	demo->loadModel(params.model_file, "model");

	demo->run();

	delete demo;

	return 1;
}