/*
class ArgParser

The argument parser is a part of the object tracking example application.
It parses arguments necessary for PPF and CPF feature descriptors. 

All functionality is static. Use
ArgParser::Help();
to get a list of the supported arguments. 

Features:
- Parses arguments from a command line string

Rafael Radkowski
Iowa State University
rafael@iastate.edu
+1 (515) 294 7044
MIT License
------------------------------------------------------
Last Changes:

*/

#ifndef __ARGPARSER__
#define __ARGPARSER__

// stl
#include <iostream>
#include <string>
#include <vector>

// local
#include "types.h"

using namespace std;

namespace isu_ar
{

typedef struct _Arguments
{
	
	string  current_path;
	string	ref_model_path_and_file;
	string	test_model_path_and_file;


	string	log_output_path;
	string  mode; // auto or manual.

	int		windows_width;
	int		window_height;
	
	int		test_runs;

	double	distance_step;
	double	angle_step;
	double	cluster_dist_th;
	double	cluster_ang_th;

	double	uniform_grid_size;



	// helpers
	bool		verbose;
	bool		valid;



	_Arguments()
	{

		ref_model_path_and_file = "";
		test_model_path_and_file = "";
		log_output_path = "";

		mode = "manual";
		test_runs = 1000;

		distance_step = 0.1;
		angle_step = 12.0;
		uniform_grid_size = 0.01;

		cluster_dist_th = 0.001;
		cluster_ang_th = 12.0;

		windows_width = 1280;
		window_height = 1024;


		verbose = false;
		valid = false;
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