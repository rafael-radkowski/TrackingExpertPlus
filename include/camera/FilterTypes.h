#pragma once

/*
@class FilterTypes.h

@brief - Defines filter types for the PointCloudProducer class

Rafael Radkowski
Iowa State University
rafael@iastate.edu
September 27, 2017
MIT License
---------------------------------------------------------------

#include <iostream>
#include <string>

*/


/*!
Filter methods the PointCloudProducer provides
*/
typedef enum {
	NONE,
	BILATERAL
}FilterMethod;

/*!
Filter parameters for the point cloud PointCloudProducer filters
*/
typedef struct FilterParams {

	// kernel size in pixel
	int		kernel_size; 
	
	// bilateral filter
	float	sigmaI; // intensity sigma
	float	sigmaS; // spatial sigma



	FilterParams() {
		kernel_size = 9;
		sigmaI = 12.0f;
		sigmaS = 6.0f;
	}

}FilterParams;