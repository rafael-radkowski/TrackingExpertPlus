#include "ArgParser.h"


using namespace texpert;


namespace ArgParserTypes{

	Arguments opt;
	int error_count = 0;
}

using namespace ArgParserTypes;


/*
Parse the arguments
@param argc - number of arguments
@param argv - the argument line
@return struct with arguments
*/
Arguments ArgParser::Parse(int& argc, char** argv)
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

				if (opt.camera_type.compare("AzureKinectMKV") == 0) {
					opt.scene_file = string(argv[pos + 2]);
				}
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
void ArgParser::Help(void)
{
	cout << "[ERROR] - Missing parameters\nUsage:" << endl;

	cout << "\nExample: TrackingExpertExample -cam AzureKinect -model ../data/stanford_bunny_pc.obj\n" << endl;
}


/*
Display all arguments
*/
//static 
void ArgParser::Display(void)
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
void ArgParser::Path(char* path)
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
void ArgParser::ParamError(string option)
{
	cout << "[ERROR] - Parameter for option " << option << " is missing or invalid." << endl;
	error_count++;
}