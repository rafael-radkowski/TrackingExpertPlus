#include "registration_analysis.h"


using namespace texpert;

/*
Constructor
@param window_width, window_height - size of the opengl window in pixel.
@param window_name - string containing the label for the opengl window.
*/
FMEvalApp::FMEvalApp(int window_height, int window_width, string window_name):
	_window_height(window_height), _window_width(window_width), _window_name(window_name)
{
	_renderer = NULL;
	_gl_ref = NULL;
	_gl_test = NULL;
	_gl_check = NULL;
	_use_same = false;
	_test_file = "";
	_ref_file = "";


	_angle_step = 12.0;
	_distance_step = 0.03;
	_noise = 0.0;

	_sampling_method = SamplingMethod::UNIFORM;
	_sampling_param.grid_x = 0.1;
	_sampling_param.grid_y = 0.1;
	_sampling_param.grid_z = 0.1;

	_N_testruns = 1;
	_N_current = 0;
	_with_auto = false;

	_error_threshold = 0.01;
	_N_Good = 0;
	_verbose = false;
	_logfolder_name = "";
	
	_feature_matching = new FDMatching();
}


FMEvalApp::~FMEvalApp()
{
	if (_renderer) delete _renderer;
	if (_gl_ref) delete _gl_ref;
	if (_gl_test) delete _gl_test;
	if (_gl_check) delete _gl_check;
	if (_feature_matching) delete _feature_matching;
}

/*
Init the feature matching instance by loading a reference and a test point cloud.
The reference point set feeds the feature map and remains static. The test point cloud is transformed 
so that it matches the pose of the reference point set. 
@param reference_pointcloud - string containing the path to the reference point cloud.
@param test_pointcloud - string containing the path to the test point cloud.
BOTH MUST BE AN OBF MODEL FILE
*/
bool FMEvalApp::init(string reference_pointcloud, string test_pointcloud)
{
	_use_same = false;

	_test_file = test_pointcloud;
	_ref_file = reference_pointcloud;
	bool ret = init_detection();
	if (!ret) {
		cout << "[ERROR] - Error while initializing detector." << endl;
	}
	ret = init_graphics();
	if (!ret) {
		cout << "[ERROR] - Error while initializing graphics." << endl;
	}
	return ret;
}


/*
Init the feature matching instance by loading a reference and a test point cloud.
Same function as init(string reference_pointcloud, string test_pointcloud);
But the same model will be used for reference and testing. 
RUN INIT AFTER ALL PARAMETERS ARE SET
@param pointcloud_file - string containing the path to the reference point cloud.
BOTH MUST BE AN OBF MODEL FILE
*/
bool FMEvalApp::init(string pointcloud_file)
{
	_use_same = true;
	_test_file = pointcloud_file;
	_ref_file = pointcloud_file;
	bool ret = init_detection();
	if (!ret) {
		cout << "[ERROR] - Error while initializing detector." << endl;
	}
	ret = init_graphics();
	if (!ret) {
		cout << "[ERROR] - Error while initializing graphics." << endl;
	}
	return ret;
}


/*
Set a sampling method. 
Default is set to UNIFORM with a sampling grid of 0.1;
@param method - the sampling method of type SamplingMethod.
@param param - the sampling parameters of type SamplingParam.
Note that the parameters must belong to the method. 
@return true - if successfully set. 
*/
bool FMEvalApp::setSamplingMethod(SamplingMethod method, SamplingParam param)
{
	_sampling_method = method;
	_sampling_param = param;
	return true;
}

/*
Set the feature matching parameters
*/
bool FMEvalApp::setMatchingParameters(double distance_step, double angle_step)
{
	if(distance_step > 0.0)
		_distance_step = distance_step;

	if(angle_step > 0.0)
		_angle_step = angle_step;

	return true;
}



/*
Set the cluster threshold for pose clusterin algorithm. 
Pose clustering considers all poses as equal (similar) if their 
center-distance and their angle delta are under a threshold. 
@param distance_th - the distance threshold. 
@param angle_th - the angle threshold in degress. 
BOTH VALUES NEED TO BE LARGER THAN 1
*/
void FMEvalApp::setClusteringThreshold(float distance_th, float angle_th)
{
	if (!_feature_matching) return;

	_feature_matching->setClusteringThreshold(distance_th, angle_th);
}


/*
Init the opengl graphics content
*/
bool FMEvalApp::init_graphics(void)
{
	// create the renderer
	_renderer = new GLViewer();
	_renderer->create(_window_width, _window_height, "Feature Matching test");
	_renderer->addRenderFcn(std::bind( &FMEvalApp::render_function, this, _1, _2 ));
	_renderer->addKeyboardCallback(std::bind(&FMEvalApp::keyboard_callback, this, _1, _2));
	_renderer->setViewMatrix(glm::lookAt(glm::vec3(0.0f, 0.0, 0.5f), glm::vec3(0.0f, 0.0f, 00.f), glm::vec3(0.0f, 1.0f, 0.0f)));
	_renderer->setClearColor(glm::vec4(0, 0, 0, 1));
   
	// load two programs since the renderer do not change the material
	int program0 = cs557::LoadAndCreateShaderProgram("../src/shaders/simple_point_renderer.vs", "../src/shaders/simple_point_renderer.fs");
	int program1 = cs557::LoadAndCreateShaderProgram("../src/shaders/simple_point_renderer.vs", "../src/shaders/simple_point_renderer.fs");
	int program2 = cs557::LoadAndCreateShaderProgram("../src/shaders/simple_point_renderer.vs", "../src/shaders/simple_point_renderer.fs");


	_light0.pos = glm::vec3(0.0f, 5.0f, 3.0f);
	_light0.dir = glm::vec3(0.0f, 0.0f, 0.0f);
	_light0.k1 = 0.1;
	_light0.intensity = 1.7;
	_light0.index = 0;
	

	_light1.pos = glm::vec3(0.0f, 5.0f, -3.0f);
	_light1.dir = glm::vec3(0.0f, 0.0f, 0.0f);
	_light1.k1 = 0.1;
	_light1.intensity = 1.0;
	_light1.index = 1;
	


	// reference point set is red
	_gl_ref = new GLPointCloud();
	_gl_ref->create(_pc_ref, program0);
	_light0.apply(_gl_ref->getProgram());
	_light1.apply(_gl_ref->getProgram());

	_mat_ref.ambient_mat = glm::vec3(1.0,0.0,0.0) ;
	_mat_ref.ambient_int = 0.5;
	_mat_ref.diffuse_mat = glm::vec3(1.0,0.0,0.0);
	_mat_ref.diffuse_int = 2.5;
	_mat_ref.specular_mat = glm::vec3(1.0,1.0,1.0);
	_mat_ref.specular_int = 0.2;
	_mat_ref.specular_s = 12.0;
	_mat_ref.apply(_gl_ref->getProgram());
	

	// test point set is green
	_gl_test = new GLPointCloud();
	_gl_test->create(_pc_test, program1);
	_light0.apply(_gl_test->getProgram());
	_light1.apply(_gl_test->getProgram());

	_mat_test.ambient_mat = glm::vec3(0.0,1.0,0.0) ;
	_mat_test.ambient_int = 0.5;
	_mat_test.diffuse_mat = glm::vec3(0.0,1.0,0.0);
	_mat_test.diffuse_int = 2.5;
	_mat_test.specular_mat = glm::vec3(1.0,1.0,1.0);
	_mat_test.specular_int = 0.2;
	_mat_test.specular_s = 12.0;
	_mat_test.apply(_gl_test->getProgram());


	_gl_check = new GLPointCloud();
	_gl_check->create(_pc_test, program2);
	_light0.apply(_gl_check->getProgram());
	_light1.apply(_gl_check->getProgram());


	_mat_check.ambient_mat = glm::vec3(1.0,1.0,0.0) ;
	_mat_check.ambient_int = 0.5;
	_mat_check.diffuse_mat = glm::vec3(0.0,1.0,0.0);
	_mat_check.diffuse_int = 2.5;
	_mat_check.specular_mat = glm::vec3(1.0,1.0,1.0);
	_mat_check.specular_int = 0.2;
	_mat_check.specular_s = 12.0;
	_mat_check.apply(_gl_check->getProgram());

	return true;
}

/*
Init the feature matching part. 
*/
bool FMEvalApp::init_detection(void)
{
	Sampling::SetMethod(_sampling_method, _sampling_param);

	if (!_use_same) {
		LoaderObj::Read(_test_file, &_pc_test_as_loaded.points, &_pc_test_as_loaded.normals, false, true);
		LoaderObj::Read(_ref_file, &_pc_ref_as_loaded.points, &_pc_ref_as_loaded.normals, false, true);

		Sampling::Run(_pc_ref_as_loaded, _pc_ref_as_loaded);
		Sampling::Run(_pc_test_as_loaded, _pc_test_as_loaded);

		_pc_test = _pc_test_as_loaded;
		_pc_ref = _pc_ref_as_loaded;
	}
	else {
		LoaderObj::Read(_ref_file, &_pc_ref_as_loaded.points, &_pc_ref_as_loaded.normals, false, true);
		Sampling::Run(_pc_ref_as_loaded, _pc_ref_as_loaded);

		_pc_test = _pc_ref_as_loaded;
		_pc_ref = _pc_ref_as_loaded;
		_pc_test_as_loaded = _pc_ref_as_loaded;
	}

	// Move the point cloud to a different position. 
	PointCloudTransform::Transform(&_pc_test,  Eigen::Vector3f(0.0, 0.2, 0.0), Eigen::Vector3f(0.0, 0.0, 90.0));

	// init feature matching
	_feature_matching->setAngleStep(_angle_step);
	_feature_matching->setDistanceStep(_distance_step);
	_feature_matching->invertPose(true);
	_feature_matching->extract_feature_map(&_pc_ref.points, &_pc_ref.normals);

	return true;
}

/*
Start a single test run with a random position
*/
void FMEvalApp::startSingleTest(void)
{
	// generate a random position and orientation

	std::vector<float> p = RandomGenerator::FloatPosition(-0.25, 0.25);
	std::vector<float> a = RandomGenerator::FloatPosition(-90.0, 90.0);
	Eigen::Vector3f ep(&p[0]);
	Eigen::Vector3f ea(&a[0]);

	//ea.x() = 0.0;
	//ea.y() = 0.0;
	//ea.z() = 0.0;

	//ep.x() = 0.0;
	//ep.y() = 0.0;
	//ep.z() = 0.0;

	// Re-create the point cloud and transform it to a random position/orientation

	_pc_test = _pc_test_as_loaded;
	PointCloudTransform::Transform(&_pc_test,  ep, ea);


	// update the gl models

	_gl_test->update(_pc_test);
	_gl_check->update(_pc_test);

	// 

	std::vector<int> matches;
	if (_use_same) { // the objects are index aligned if the same models are used
		for (int i = 0; i < _pc_test_as_loaded.size(); i++) {
			matches.push_back(i);
		}
	}else
	{ // knn with kd-tree

	}

	if (_verbose) {
		cout << "-----------------\nNew try\nRandom pos: " << ep.x() << ", " << ep.y() << ", " << ep.z() << endl;
		cout << "Random ang: " << ea.x() << ", " << ea.y() << ", " << ea.z() << endl;
	}

	// run the matching algorithm

	vector<Pose> poses;
	_feature_matching->detect(&_pc_test.points, &_pc_test.normals, poses);


	// final result
	glm::mat4 m = MatrixUtils::Affine3f2Mat4(poses[0].t);

	
	double e = PointCloudEval::RMS(_pc_ref, _pc_test, poses[0].t, matches);

	if (_verbose) {
		//MatrixUtils::PrintGlm4(m);
		cout << "RMS: " << e << endl;
		cout << "Votes: " << poses[0].votes << endl;
	}
	

	// update the 3D model to show the final result
	
	_gl_check->setModelmatrix(m);


	// log the results

	logResult(_N_current, e, poses[0].votes, ep, ea);


	// overall evaluation

	if (e < _error_threshold) {
		_N_Good++;
	}
	
}

void FMEvalApp::keyboard_callback( int key, int action) {


	//cout << key << " : " << action << endl;


	switch (action) {
	case 0:  // key up
	
		switch (key) {
		case 81: // q
			startSingleTest();
			break;
		}
		break;
	case 1: // key down

		break;
	}
}

void FMEvalApp::render_function(glm::mat4 pm, glm::mat4 vm)
{
	if (_with_auto)
	{
		if (_N_current < _N_testruns) {
			startSingleTest();
			_N_current++;

			if (_N_current % 100 == 0 && _N_current > 1) {
				cout << ". ";
			}
			if (_N_current % 1000 == 0 && _N_current > 1) {
				cout << "\n";
			}

		}
		else {
			double pr = double(_N_Good) / double(_N_current);
			cout << "\n[INFO] - Done testing " << _N_current  << " poses with a precision of " << pr*100.0 << "% (" << _error_threshold  <<" acceptance threshold)." << endl;
			_with_auto = false;
			writeResults();

			string msg = to_string(pr);
			msg.append(",");
			msg.append(to_string(_N_Good));
			msg.append(",");
			msg.append(to_string(_angle_step));
			msg.append(",");
			msg.append(to_string(_distance_step));
			msg.append(",");
			msg.append(to_string(_noise));


			string p = "./logs/";
			p.append(_logfolder_name);
			p.append("/result_summary.csv");
			LogReaderWriter::FlashWrite(p, msg);

			_N_current = 0;
			_N_Good = 0;
			_renderer->stop();
		}
		
	}


	if(_gl_ref)
		_gl_ref->draw(pm, vm);
	if(_gl_test)
		_gl_test->draw(pm, vm);
	if(_gl_check)
		_gl_check->draw(pm, vm);

}


/*
Start the renderer and the test run
@param mode - the process mode: Manual, use starts each run manually. 
				Auto - the program runs N tries automatically. 
*/
void FMEvalApp::start(Mode mode)
{

	createLogHeader();

	switch (mode) {

	case Manual:
		_with_auto = false;
		if (_renderer) {
			_renderer->start();
		}
		else
		{
			cout << "[ERROR] - Cannot start renderer. Init the application first." << endl;
		}
		break;
	case Auto:
		cout << "\n";
		_with_auto = true;
		
		if (_renderer) {
			_renderer->start();
		}
		else
		{
			cout << "[ERROR] - Cannot start renderer. Init the application first." << endl;
		}
		break;
	}
}


/*
Set a log file to store the results. 
Setting this log file will automatically start the log writer. 
Set path = "" to disable the writer. 
@param path - relative or absolute path to a log file. 
Data is stored in csv format. So store a csv;
*/
void FMEvalApp::setLogFile(string path)
{
	_with_log = false;
	if (path.length() > 0) _with_log = true;
	_logfile = path;

	string log_name = "";
	int idx = path.find_last_of('/');
	if (idx > 0) {
		log_name = path.substr(0, idx);

		int idx2 = log_name.find_last_of('/');
		if (idx2 > 0) {
			log_name = log_name.substr(idx2+1, idx - idx2 - 1);

			_logfolder_name = log_name;
		}
	}
}


void FMEvalApp::createLogHeader(void)
{
	if (!_with_log) return;
	LogReaderWriter::Create(_logfile);


	LogMetaData d;
	d.file_ref = _ref_file;
	d.file_test = _test_file;
	d.num_points_ref = _pc_ref.size();
	d.num_points_test = _pc_test.size();
	d.matching_type = "PPF";
	d.distance_step = _distance_step;
	d.angle_step = _angle_step;
	d.noise = _noise;
	d.N_good = 0;
	d.N_tests = _N_testruns;
	d.rms_th = _error_threshold;
	d.sampling_grid = _sampling_param.grid_x;
	d.sampling_type = "UNIFORM";

	LogReaderWriter::WriteMetaData(d);

	string msg = "pr,tp,ang,dist,noise";

	string p = "./logs/";
	p.append(_logfolder_name);
	p.append("/result_summary.csv");
	LogReaderWriter::FlashWriteHeader(p, msg);

}

void FMEvalApp::logResult(int idx, double rms, int votes, Eigen::Vector3f pos, Eigen::Vector3f rot)
{
	if (!_with_log) return;
	
	LogData d;
	d.iteration = idx;
	d.rms = rms;
	d.votes = votes;
	d.x = pos.x();
	d.y = pos.y();
	d.z = pos.z();
	d.rx = rot.x();
	d.ry = rot.y();
	d.rz = rot.z();

	LogReaderWriter::Write(d);
}


void FMEvalApp::writeResults(void)
{
	if (!_with_log) return;

	LogMetaData d;
	d.file_ref = _ref_file;
	d.file_test = _test_file;
	d.num_points_ref = _pc_ref.size();
	d.num_points_test = _pc_test.size();
	d.matching_type = "PPF";
	d.distance_step = _distance_step;
	d.angle_step = _angle_step;
	d.noise = _noise;
	d.N_good = _N_Good;
	d.N_tests = _N_testruns;
	d.rms_th = _error_threshold;

	LogReaderWriter::WriteResults(d);
}


/*
For an automatic evaluation, set the number of test runs
@param N - integer with the number of test runs;
*/
void FMEvalApp::setNumAutoTestRuns(int N)
{
	if (N < 1) return;
	_N_testruns = N;
}


/*
Get more console output
@param enable - true activates verbose mode. 
Default is false. 
*/
void FMEvalApp::setVerbose(bool enable)
{
	_verbose = enable;
}