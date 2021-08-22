#include "ArgParser.h"


//using namespace texpert;


namespace texpert_demo::ArgParserTypes{

        texpert_demo::Arguments opt;
	int error_count = 0;
}

using namespace texpert_demo::ArgParserTypes;


/*
Parse the arguments
@param argc - number of arguments
@param argv - the argument line
@return struct with arguments
*/
texpert_demo::Arguments texpert_demo::ArgParser::Parse(int& argc, char** argv)
{
	//cout << argc << endl;
	//cout << argv[0] << endl;

	if (argc < 1) {
		Help();
		return opt;
	}

	opt.valid = true;

	// extract the path
	Path(argv[0]);


	int pos = 1;
	while(pos < argc)
	{
		string c_arg(argv[pos]); 

		if (c_arg.compare("-cam") == 0) { // camera type
			if (argc >= pos+1){ 
				opt.camera_type = string(argv[pos+1]);
				opt.scene_file = "";
			}
			else ParamError(c_arg);
		}
		else if(c_arg.compare("-scene") == 0){ // image width 
			if (argc >= pos){
				opt.scene_file =  string(argv[pos+1]);
				opt.camera_type = "None";
			}
			else ParamError(c_arg);
		}
		else if(c_arg.compare("-model") == 0){ // image width 
			if (argc >= pos){
				opt.model_file =  string(argv[pos+1]);
			}
			else ParamError(c_arg);
		}
		else if(c_arg.compare("-verbose") == 0 ){ // help
			opt.verbose = true;
		}
		else if(c_arg.compare("-wnd_w") == 0){ // window width 
			if (argc >= pos) opt.windows_width = atoi( string(argv[pos+1]).c_str() );
			else ParamError(c_arg);
		}
		else if(c_arg.compare("-wnd_h") == 0){ // window height  
			if (argc >= pos) opt.window_height =atoi(  string(argv[pos+1]).c_str() );
			else ParamError(c_arg);
		}
		else if(c_arg.compare("-fdang") == 0){ // window width 
			if (argc >= pos) opt.fd_angle_step = atof( string(argv[pos+1]).c_str() );
			else ParamError(c_arg);
		}
		else if(c_arg.compare("-fdrad") == 0){ // window width 
			if (argc >= pos) opt.fd_curvature_radius = atof( string(argv[pos+1]).c_str() );
			else ParamError(c_arg);
		}
		else if(c_arg.compare("-cluster_t") == 0){ // window height  
			if (argc >= pos) opt.fd_cluster_trans_th =atof(  string(argv[pos+1]).c_str() );
			else ParamError(c_arg);
		}
		else if(c_arg.compare("-cluster_r") == 0){ // window height  
			if (argc >= pos) opt.fd_cluster_rot_th =atof(  string(argv[pos+1]).c_str() );
			else ParamError(c_arg);
		}
		else if(c_arg.compare("-icp_rms") == 0){ // window height  
			if (argc >= pos) opt.icp_min_rms =atof(  string(argv[pos+1]).c_str() );
			else ParamError(c_arg);
		}
		else if(c_arg.compare("-icp_dist") == 0){ // window height  
			if (argc >= pos) opt.icp_outlier_dist_th =atof(  string(argv[pos+1]).c_str() );
			else ParamError(c_arg);
		}
		else if(c_arg.compare("-icp_ang") == 0){ // window height  
			if (argc >= pos) opt.icp_outlier_ang_th =atof(  string(argv[pos+1]).c_str() );
			else ParamError(c_arg);
		}
		else if(c_arg.compare("-icp_max") == 0){ // window height  
			if (argc >= pos) opt.icp_max_iterations =atoi(  string(argv[pos+1]).c_str() );
			else ParamError(c_arg);
		}
		else if(c_arg.compare("-grid") == 0){ // window height  
			if (argc >= pos) opt.sampling_grid_size =atof(  string(argv[pos+1]).c_str() );
			else ParamError(c_arg);
		}
		else if(c_arg.compare("-int") == 0){ // intrinsic camera params
			if (argc >= pos) opt.intrincic_params_file =atof(  string(argv[pos+1]).c_str() );
			else ParamError(c_arg);
		}
		else if(c_arg.compare("-help") == 0 || c_arg.compare("-h") == 0){ // help
			Help();
		}
		else if(c_arg.compare("-bf_si") == 0){ // bilateral filter sigma I
			if (argc >= pos) opt.filter_sigmaI =atof(  string(argv[pos+1]).c_str() );
			else ParamError(c_arg);
		}
		else if(c_arg.compare("-bf_ss") == 0){ // bilateral filter sigma S
			if (argc >= pos) opt.filter_sigmaS =atof(  string(argv[pos+1]).c_str() );
			else ParamError(c_arg);
		}
		else if(c_arg.compare("-bf_ks") == 0){ // bilateral filter kernel size
			if (argc >= pos) opt.filter_kernel =atoi(  string(argv[pos+1]).c_str() );
			else ParamError(c_arg);
		}
		else if(c_arg.compare("-with_filter") == 0){ // bilateral filter kernel size
			opt.filter_enabled = true;
		}
		else if(c_arg.compare("-cam_offset") == 0){ // cuda camera procuder grid offset
			if (argc >= pos) opt.camera_sampling_offset = atoi(  string(argv[pos+1]).c_str() );
			else ParamError(c_arg);
		}
		pos++;
	}


	if (opt.verbose)
		Display();

	return opt;
}


/*
Display help
*/
//static
void texpert_demo::ArgParser::Help(void)
{
	cout << "[ERROR] - Missing parameters\nUsage:" << endl;
	cout << "setforge_r [3d model path and filename] " << endl;
	cout << "Optional parameters:" << endl;
	cout << "\t-intr [param] - path and filename for the intrinsic camera parameters." << endl;
	cout << "\t-o [param] - set the output path" << endl;
	cout << "\t-img_w [param] \t- set the widht of the output image in pixels (integer)." << endl;
	cout << "\t-img_h [param] \t- set the height of the output image in pixels (integer)." << endl;
	cout << "\t-wnd_w [param] \t- set the widht of the application window in pixels (integer)." << endl;
	cout << "\t-wnd_h [param] \t- set the height of the application window in pixels (integer)." << endl;
	cout << "\t-m [param] \t- set the camera path models. Can be USER, POSE, SPHERE, POLY, TREE." << endl;
	cout << "\t-seg [param] \t- for the camera path SPHERE model, set the number of segments (integer)." << endl;
	cout << "\t-rows [param] \t-for the camera path SPHERE model, set the number of rows (integer)" << endl;
	cout << "\t-dist [param] \t-for the camera path SPHERE model, set the sphere radius (float)" << endl;
	cout << "\t-sub [param] \t-for the camera path model POLY and POSE, set the number of subdivisions for the polyheder (int)" << endl;
	cout << "\t-num [param] \t- for camera path control POSE, set the number of images to generate (int)" << endl;
	cout << "\t-limx [param] \t- for camera path control POSE, set the x-axis limit for the random position (float)" << endl;
	cout << "\t-limy [param] \t- for camera path control POSE, set the y-axis limit for the random position (float)" << endl;
	cout << "\t-lim_near [param] \t- for camera path control POSE, set the near z-axis limit (positive along the camera axis) for the random position (float)" << endl;
	cout << "\t-lim_far [param] \t- for camera path control POSE, set the far z-axis limit (postive along the camera axis) for the random position (float)" << endl;
	cout << "\t-level [param] \t-for the camera path TREE, the number of tree levels for the Balanced Pose Tree (int)" << endl;
	cout << "\t-up \t- Renders objects only in the upright position if set, where up is the positive y-direction." << endl;
	cout << "\t-rand_col [param] - enable color randomization. Param: path and filename of a json file with color parameters." << endl;
	cout << "\t-verbose \t- displays additional information." << endl;
	cout << "\t-help \t- displays this help menu" << endl;

	cout << "\nExample: DatasetRenderer ../data/stanford_bunny_02_rot.obj -m POLY -img_w 1280 -img_h 1024 -wnd_w 1280 - wnd_h 1024 -o output -sub=1 -rad 1.3\n" << endl;
}


/*
Display all arguments
*/
//static 
void texpert_demo::ArgParser::Display(void)
{
	std::cout << "\nParameters:\n--------------------------------------------------------------------------" << endl;
	std::cout << "Camera type: \t\t" << opt.camera_type << std::endl;
	std::cout << "Model file: \t\t" << opt.model_file << std::endl;
	std::cout << "Scene file: \t\t" << opt.scene_file << std::endl;

	std::cout << "Descriptor histogram bin angle:\t" << opt.fd_angle_step << std::endl;
	std::cout << "Descriptor cluster translation:\t" << opt.fd_cluster_trans_th << std::endl;
	std::cout << "Descriptor cluster rotation: \t" << opt.fd_cluster_rot_th << std::endl;
	std::cout << "ICP min RMS: \t\t\t" << opt.icp_min_rms << std::endl;
	std::cout << "ICP outlier reject angle: \t" << opt.icp_outlier_ang_th << std::endl;
	std::cout << "ICP outlier reject distance: \t" << opt.icp_outlier_dist_th << std::endl;
	std::cout << "ICP max iterations: \t\t" << opt.icp_max_iterations << std::endl;
	std::cout << "Sampling voxel edge size:\t" << opt.sampling_grid_size << std::endl;
	std::cout << "Intrinsic and dist. camera file: \t" << opt.intrincic_params_file << std::endl;
	std::cout << "Filter sigma S: \t\t" << opt.filter_sigmaS << std::endl;
	std::cout << "Filter sigma I: \t\t" << opt.filter_sigmaI << std::endl;
	std::cout << "Filter kernel size:\t\t" << opt.filter_kernel << std::endl;

	if (opt.filter_enabled) {
		std::cout << "Filter:\t\t\tEnabled " << std::endl;
	}else{
		std::cout << "Filter:\t\t\tDisabled " << std::endl;
	}
	std::cout << "\nCurrent working path: \t" << opt.current_path << std::endl;
	
	
	std::cout << "Wnd width:\t\t\t" << opt.windows_width << endl;
	std::cout << "Wnd height:\t\t\t" << opt.window_height << endl;

	if (opt.verbose) {
		std::cout << "Verbose: True " << std::endl;
	}else{
		std::cout << "Verbose: False " << std::endl;
	}
	std::cout << "\n--------------------------------------------------------------------------\n" << std::endl;



}


/*
Extract the current path
*/
//static 
void texpert_demo::ArgParser::Path(char* path)
{
	string str(path);

	int idx = str.find_last_of("\\");
	if(idx == -1){ 
		idx = str.find_last_of("/");
	}
	opt.current_path = str.substr(0, idx+1);
}

/*
Extract the current path
*/
//static 
void texpert_demo::ArgParser::ParamError(string option)
{
	cout << "[ERROR] - Parameter for option " << option << " is missing or invalid." << endl;
	error_count++;
}
