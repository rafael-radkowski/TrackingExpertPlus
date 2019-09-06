#pragma once
/*
File contains types for the class Sample.

Rafael Radkowski
Iowa State University
rafael@iastate.edu
Sep 4th, 2017
All rights reserved
*/




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

	// Note, the last three are experimental and should not be used for an application.
	// They can also not be set via the API
	POI_RANDOM = 3,
	POI_UNIFORM = 4,
	UNIFORM_FALLOFF = 5
}SamplingMethod;