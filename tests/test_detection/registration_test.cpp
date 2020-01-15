#include <iostream>
#include <string>



// GLM include files
#define GLM_FORCE_INLINE
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtx/transform.hpp>  // transformation
#include <glm/gtx/quaternion.hpp> // quaternions


// local
#include "ArgParser.h"
#include "registration_analysis.h"
#include "LogFileWriter.h"


using namespace texpert;


texpert::FMEvalApp*			app;

using namespace std;
using namespace cs557;
using namespace isu_ar;

#define _WITH_LOG
// -r  ../data/stanford_bunny_02.obj -t ../data/stanford_bunny_02.obj -mode manual -log ./logs/test_manual.csv -c_ang 12.0 -c_dist 0.05 -n 1000 -dist 0.002 -ang 6.0 -grid 0.01 -wnd_w 1920  -wnd_h 1536 -verbose

int main(int argc, char** argv)
{

	Arguments app_params = ArgParser::Parse(argc, argv);

	LogAdmin::startNewLog("registration_test_log");
	LogFileWriter lfwRegTest = LogFileWriter("registration_test_log");
	lfwRegTest.enableLog(true);

	if (!app_params.valid) {
		cout << "[ERROR] - Not enough arguments provided to run the application." << endl;
		lfwRegTest.addLog("[ERROR] - Not enough arguments provided to run the application.");
		ArgParser::Help();
		return 1;
	}

	SamplingParam param;
	param.grid_x = app_params.uniform_grid_size;
	param.grid_y = app_params.uniform_grid_size;
	param.grid_z = app_params.uniform_grid_size;

	int window_height = app_params.window_height;
	int window_width = app_params.windows_width;

	FMEvalApp::Mode mode = FMEvalApp::Mode::Auto;
	if(app_params.mode.compare("manual") == 0)
		mode = FMEvalApp::Mode::Manual;


	app = new FMEvalApp(window_height, window_width);
	app->setSamplingMethod(SamplingMethod::UNIFORM, param);
	lfwRegTest.addLog("Sampling method set");
	app->setMatchingParameters(app_params.distance_step, app_params.angle_step);
	lfwRegTest.addLog("Matching parameters set");
	app->setClusteringThreshold(app_params.cluster_dist_th, app_params.cluster_ang_th);
	lfwRegTest.addLog("Clustering threshold set");
	app->setLogFile(app_params.log_output_path);
	lfwRegTest.addLog("Log file set");
	app->init(app_params.test_model_path_and_file);
	lfwRegTest.addLog("Test point cloud initialized");
	app->setNumAutoTestRuns(app_params.test_runs);
	lfwRegTest.addLog("Number of automatic test runs set");
	app->setVerbose(app_params.verbose);
	lfwRegTest.addLog("Verbose mode activated");
	app->start(mode);
	lfwRegTest.addLog("Application started");

	delete app;
	lfwRegTest.addLog("Application deleted");

	return 1;



	return 1;
}