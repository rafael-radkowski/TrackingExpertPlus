/*
class SamplingTypes

Rafael Radkowski
Iowa State University
rafael@iastate.edu
MIT License
---------------------------------------------------------------
*/

#ifndef __SAMPLING_TIMES__
#define __SAMPLING_TIMES__

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

    _SamplingParam()
    {
        // Unit of the grid is the model unit. 
        grid_x = 0.01;
        grid_y = 0.01;
        grid_z = 0.01;
    }

}SamplingParam;



#endif