/*
class SamplingTypes

Rafael Radkowski
Iowa State University
rafael@iastate.edu
MIT License
---------------------------------------------------------------

Last edits:
Aug 9, 2020, RR:
- Added a method to validate the correctness of the value and to correct them if required. 
*/


#ifndef __SAMPLING_TIMES__
#define __SAMPLING_TIMES__

#include <algorithm>

/*
The different datasets sampling types for the camera data and the loaded model. 
RAW: all points are used
UNIDORM: the points are uniformly distributed, using a voxel raster or a grid on the image.
RANDOM: random points are selected, without any distribution. 
*/
typedef enum _SamplingMethod
{
	RAW = 0,
	UNIFORM = 1,
	RANDOM = 2,

}SamplingMethod;



typedef struct _SamplingParam
{
    // voxel grid size 
    // for uniform sampling
    float grid_x;
    float grid_y;
    float grid_z;

	// step size for uniform sampling
	int		uniform_step; 

	int		random_max_points;
	float	ramdom_percentage;

    _SamplingParam()
    {
        // Unit of the grid is the model unit. 
        grid_x = 0.01f;
        grid_y = 0.01f;
        grid_z = 0.01f;

		uniform_step = 1;
		random_max_points = 5000;
		ramdom_percentage = 25; // currently not in use. Use the max random points number. 
    }

	void validate(void) {
		uniform_step = std::max(1, uniform_step);
		random_max_points = std::max(1, random_max_points);
		ramdom_percentage = std::max(1.0f, std::min(100.0f, ramdom_percentage));

		grid_x =  std::max(0.0001f, grid_x);
		grid_y =  std::max(0.0001f, grid_y);
		grid_z =  std::max(0.0001f, grid_z);
	}

}SamplingParam;



#endif