#pragma once


#include <iostream>
#include <string>
#include <iomanip> 


// GLM include files
#define GLM_FORCE_INLINE
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtx/transform.hpp>  // transformation
#include <glm/gtx/quaternion.hpp> // quaternions


// local
#include "GLViewer.h"
#include "GLPointCloud.h"
#include "LoaderOBJ.h"
#include "Types.h"
#include "PointCloudTransform.h"
#include "RandomGenerator.h"
#include "FDMatching.h"
#include "Sampling.h"
#include "Utils.h"
#include "LogReaderWriter.h"


using namespace cs557;
using namespace std;
using namespace std::placeholders;
using namespace isu_ar;

namespace texpert {

	class FMEvalApp
	{
	public:

		typedef enum Mode {
			Manual,
			Auto
		}Mode;



		/*
		Constructor
		@param window_width, window_height - size of the opengl window in pixel.
		@param window_name - string containing the label for the opengl window.
		*/
		FMEvalApp(int window_height, int window_width, string window_name = "Feature Matching Evaluation" );
		~FMEvalApp();

		/*
		Init the feature matching instance by loading a reference and a test point cloud.
		The reference point set feeds the feature map and remains static. The test point cloud is transformed 
		so that it matches the pose of the reference point set. 
		@param reference_pointcloud - string containing the path to the reference point cloud.
		@param test_pointcloud - string containing the path to the test point cloud.
		BOTH MUST BE AN OBF MODEL FILE
		*/
		bool init(string reference_pointcloud, string test_pointcloud);

		/*
		Init the feature matching instance by loading a reference and a test point cloud.
		Same function as init(string reference_pointcloud, string test_pointcloud);
		But the same model will be used for reference and testing. 
		RUN INIT AFTER ALL PARAMETERS ARE SET.
		@param pointcloud_file - string containing the path to the reference point cloud.
		BOTH MUST BE AN OBF MODEL FILE
		*/
		bool init(string pointcloud_file);


		/*
		Set a sampling method. 
		Default is set to UNIFORM with a sampling grid of 0.1;
		@param method - the sampling method of type SamplingMethod.
		@param param - the sampling parameters of type SamplingParam.
		Note that the parameters must belong to the method. 
		@return true - if successfully set. 
		*/
		bool setSamplingMethod(SamplingMethod method, SamplingParam param);


		/*
		Set the feature matching parameters
		*/
		bool setMatchingParameters(double distance_step, double angle_step);


		/*
		Set the cluster threshold for pose clusterin algorithm. 
		Pose clustering considers all poses as equal (similar) if their 
		center-distance and their angle delta are under a threshold. 
		@param distance_th - the distance threshold. 
		@param angle_th - the angle threshold in degress. 
		BOTH VALUES NEED TO BE LARGER THAN 1
		*/
		void setClusteringThreshold(float distance_th, float angle_th);


		/*
		Start a single test run with a random position
		*/
		void startSingleTest(void);


		/*
		Start the renderer and the test run
		@param mode - the process mode: Manual, use starts each run manually. 
					  Auto - the program runs N tries automatically. 
		*/
		void start(Mode mode = Mode::Manual);


		/*
		Set a log file to store the results. 
		Setting this log file will automatically start the log writer. 
		Set path = "" to disable the writer. 
		@param path - relative or absolute path to a log file. 
		Data is stored in csv format. So store a csv;
		*/
		void setLogFile(string path);


		/*
		For an automatic evaluation, set the number of test runs
		@param N - integer with the number of test runs;
		*/
		void setNumAutoTestRuns(int N);

		/*
		Get more console output
		@param enable - true activates verbose mode. 
		Default is false. 
		*/
		void setVerbose(bool enable);

	private:

		/*
		Init the opengl graphics content
		*/
		bool init_graphics(void);


		/*
		Init the feature matching part. 
		*/
		bool init_detection(void);


		/*
		The keyboard callback the renderer calls
		*/
		void keyboard_callback(int key, int action);


		/*
		The render function the renderer callse
		*/
		void render_function(glm::mat4 pm, glm::mat4 vm);


		void createLogHeader(void);
		void logResult(int idx, double rms, int votes, Eigen::Vector3f pos, Eigen::Vector3f rot);
		void writeResults(void);

		//---------------------------------------------------------
		// Members


		// opengl main renderer
		GLViewer*				_renderer;
		int						_window_height;
		int						_window_width;
		string					_window_name;

		// the point cloud renderers
		GLPointCloud*			_gl_ref;
		GLPointCloud*			_gl_test;
		GLPointCloud*			_gl_check;

		//  light for the model
		cs557::LightSource		_light0;
		cs557::LightSource		_light1;
		cs557::Material			_mat_ref;
		cs557::Material			_mat_test;
		cs557::Material			_mat_check;



		// The reference point cloud
		PointCloud			_pc_ref;
		// The test point cloud
		PointCloud			_pc_test;

		// test and reference point cloud as loaded from files
		PointCloud			_pc_test_as_loaded;
		PointCloud			_pc_ref_as_loaded;

		// use the same 3D point cloud for reference and test
		bool				_use_same;

		string				_test_file;
		string				_ref_file;


		SamplingParam		_sampling_param;
		SamplingMethod		_sampling_method;

		double				_noise;

		//------------------------------------------------------------------------
		// matching
		FDMatching*		    _feature_matching;
		double				_distance_step;
		double				_angle_step;

		// for auto testing

		int					_N_testruns;
		int					_N_current;
		bool				_with_auto;

		//------------------------------------------------------------------------
		// Log file
		bool				_with_log;
		string				_logfile;
		string				_logfolder_name;

		double				_error_threshold;
		int					_N_Good;


		bool				_verbose;
	};

};