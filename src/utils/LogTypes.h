/*
LogMetaData

This class is a part of the performance analysis example. 
It is a helper object to pass meta information pertaining to a test to
the LogReaderWriter.

Note that this is an tool which is not necessary to enable the main functionality.  

Features:
- Stores meta information. 

Rafael Radkowski
Iowa State University
rafael@iastate.edu
+1 (515) 294 7044
July 19, 2019
MIT License
----------------------------------------------------
last edited:

*/
#pragma once

// stl
#include <iostream>
#include <string>
#include <vector>
#include <list>
#include <numeric>
#include <bitset> 

#include <Eigen/Dense>

// GLM include files
#define GLM_FORCE_INLINE
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtx/transform.hpp>  // transformation
#include <glm/gtx/quaternion.hpp> // quaternions



using namespace std;

typedef struct _LogData
{
	int		iteration;
	double	rms;
	int		votes;

	float   x;
	float	y;
	float	z;
	float	rx;
	float	ry;
	float	rz;

	_LogData( )
	{
		iteration = 0;
		rms = 0;

		x = 0;
		y = 0;
		z = 0;
		rx = 0;
		ry = 0;
		rz = 0;
	}

}LogData;



typedef struct _LogMetaData
{
	string			file_ref;
	string			file_test;
	int				num_points_test;
	int				num_points_ref;

	int				N_tests;
	int				N_good;
	double			rms_th;

	string			matching_type;

	double			angle_step;
	double			distance_step;
	double			noise;

	string			sampling_type;
	double			sampling_grid;

	_LogMetaData( )
	{
		angle_step = 0;
		distance_step = 0;
		noise = 0;
		N_tests = 0;
		N_good = 0;
		rms_th = 0;
	}

}LogMetaData;


