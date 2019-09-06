#include "ArgParser.h"


using namespace isu_ar;


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

	if (argc < 2) {
		Help();
		return opt;
	}

	// counts if an r and a t are present to mark this argument set as valid
	// if rt == 2; last line of this function. 
	int rt = 0;

	// extract the path
	Path(argv[0]);

	

	int pos = 1;
	while(pos < argc)
	{
		string c_arg(argv[pos]); 

		if (c_arg.compare("-r") == 0) { // reference model
			if (argc >= pos + 1) {
				opt.ref_model_path_and_file = string(argv[pos + 1]);
				rt++;
			}
			else ParamError(c_arg);
		}
		else if (c_arg.compare("-t") == 0) { // test model
			if (argc >= pos + 1) {
				opt.test_model_path_and_file = string(argv[pos + 1]);
				rt++;
			}
			else ParamError(c_arg);
		}
		else if (c_arg.compare("-log") == 0) { // output path
			if (argc >= pos+1) opt.log_output_path = string(argv[pos+1]);
			else ParamError(c_arg);
		}
		else if (c_arg.compare("-mode") == 0) { // output path
			if (argc >= pos+1) opt.mode = string(argv[pos+1]);
			else ParamError(c_arg);
		}
		else if (c_arg.compare("-n") == 0) { // test runs
			if (argc >= pos+1) opt.test_runs = atoi(string(argv[pos+1]).c_str());
			else ParamError(c_arg);
		}
		else if (c_arg.compare("-dist") == 0) { //
			if (argc >= pos+1) opt.distance_step = atof(string(argv[pos+1]).c_str());
			else ParamError(c_arg);
		}
		else if (c_arg.compare("-ang") == 0) { // 
			if (argc >= pos+1) opt.angle_step = atof(string(argv[pos+1]).c_str());
			else ParamError(c_arg);
		}
		else if (c_arg.compare("-c_dist") == 0) { // 
			if (argc >= pos+1) opt.cluster_dist_th = atof(string(argv[pos+1]).c_str());
			else ParamError(c_arg);
		}
		else if (c_arg.compare("-c_ang") == 0) { // 
			if (argc >= pos+1) opt.cluster_ang_th = atof(string(argv[pos+1]).c_str());
			else ParamError(c_arg);
		}
		else if (c_arg.compare("-grid") == 0) { // 
			if (argc >= pos+1) opt.uniform_grid_size = atof(string(argv[pos+1]).c_str());
			else ParamError(c_arg);
		}
		else if(c_arg.compare("-wnd_w") == 0){ // image width 
			if (argc >= pos) opt.windows_width = atoi( string(argv[pos+1]).c_str() );
			else ParamError(c_arg);
		}
		else if(c_arg.compare("-wnd_h") == 0){ // image width 
			if (argc >= pos) opt.windows_width =atoi(  string(argv[pos+1]).c_str() );
			else ParamError(c_arg);
		}
		else if(c_arg.compare("-help") == 0 || c_arg.compare("-h") == 0){ // help
			Help();
		}
		else if(c_arg.compare("-verbose") == 0 ){ // help
			opt.verbose = true;
		}
	

		pos++;
	}

	if(rt == 2)
		opt.valid = true;

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
	cout << "Object_Registration -r [reference model path and filename] -t [test model path and filename] " << endl;
	cout << "Optional parameters:" << endl;
	cout << "\t-log [param] - set the log output path and file" << endl;
	cout << "\t-mode [param] - set to 'auto' or 'manual' for auto-run or manual testing (manual is default)." << endl;
	cout << "\t-dist [param] - set the distance step for ppf  (double)." << endl;
	cout << "\t-step[param] - set the angle step for ppf in degree  (double)." << endl;
	cout << "\t-c_dist [param] - set the distance threshold for pose clustering (double)." << endl;
	cout << "\t-c_ang [param] -  set the angle threshold for pose clustering in degree (double)." << endl;
	cout << "\t-n [param] - set the number of tests to run (integer)" << endl;
	cout << "\t-grid [param] - set the uniform voxel sampling (double)." << endl;
	cout << "\t-wnd_w [param] \t- set the width of the GL window in pixels (integer)." << endl;
	cout << "\t-wnd_h [param] \t- set the height of the GL window in pixels (integer)." << endl;

	cout << "\t-verbose \t- displays additional information." << endl;
	cout << "\t-help \t- displays this help menu" << endl;

	cout << "\nExample: DatasetRenderer ../data/stanford_bunny_02_rot.obj -m POLY -img_w 1280 -img_h 1024 -wnd_w 1280 - wnd_h 1024 -o output -sub=1 -rad 1.3\n" << endl;
}


/*
Display all arguments
*/
//static 
void ArgParser::Display(void)
{
	std::cout << "---------------------------------------------------------------"  << endl;
	std::cout << "\nParameters:" << endl;
	std::cout << "-r:\t" << opt.ref_model_path_and_file << endl;
	std::cout << "-t:\t" << opt.test_model_path_and_file << endl;
	std::cout << "-r:\t" << opt.ref_model_path_and_file << endl;
	std::cout << "-log:\t" << opt.log_output_path << endl;
	std::cout << "-dist:\t" << opt.distance_step << endl;
	std::cout << "-step:\t" << opt.angle_step << endl;
	std::cout << "-c_dist:\t" << opt.cluster_dist_th << endl;
	std::cout << "-c_step:\t" << opt.cluster_ang_th << endl;
	std::cout << "-grid:\t" << opt.uniform_grid_size << endl;
	std::cout << "---------------------------------------------------------------"  << endl;
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