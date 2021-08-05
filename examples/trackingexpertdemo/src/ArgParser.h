#ifndef __ARGPARSER__
#define __ARGPARSER__

// stl
#include <iostream>
#include <string>
#include <vector>


using namespace std;

namespace texpert_demo   // Define different namespace to avoid conflict with ArgParser build into trackingexpert
{

typedef struct _Arguments
{
	string	camera_type;
	string	scene_file;
	string	model_file;

	float	fd_angle_step;
	float	fd_cluster_trans_th;
	float	fd_cluster_rot_th;
	float	fd_curvature_radius;

	float	icp_min_rms;
	float	icp_outlier_ang_th;
	float	icp_outlier_dist_th;
	int		icp_max_iterations;

	float	filter_sigmaI;
	float	filter_sigmaS;
	int		filter_kernel;
	bool	filter_enabled;

	float	sampling_grid_size;

	string	intrincic_params_file;
	string	current_path;

	int		camera_sampling_offset;

	// helpers
	bool		verbose;
	bool		valid;



	int		windows_width;
	int		window_height;

	



	_Arguments()
	{

		camera_type = "AzureKinect";
		scene_file = "";
		model_file = "";

		current_path = "";
		fd_angle_step = 12.0f;
		fd_cluster_trans_th = 0.03f;
		fd_cluster_rot_th = 45.0f;
		icp_min_rms = 0.00000001f;
		icp_outlier_ang_th = 45.0f;
		icp_outlier_dist_th = 0.1f;
		icp_max_iterations = 200;
		sampling_grid_size = 0.015f;
		camera_sampling_offset = 12;
		fd_curvature_radius = 0.1f;

		intrincic_params_file = "";

		windows_width = 1280;
		window_height = 1280;
		verbose = false;

		filter_sigmaI = 16.0;
		filter_sigmaS = 12.0;
		filter_kernel = 9;
		filter_enabled = false;

	}


}Arguments;


class ArgParser
{
public:

	/*
	Parse the arguments
	@param argc - number of arguments
	@param argv - the argument line
	@return struct with arguments
	*/
	static Arguments Parse(int& argc, char** argv);

	/*
	Display help
	*/
	static void Help(void);

	/*
	Display all arguments
	*/
	static void Display(void);

private:

	/*
	Extract the current path
	*/
	static void Path(char* path);

	/*
	Extract the current path
	*/
	static void ParamError(string option);
};



}// namespace arlab


#endif
