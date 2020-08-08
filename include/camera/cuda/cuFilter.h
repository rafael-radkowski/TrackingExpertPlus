#pragma once
/*
@class cuFilter.h/.cu

@brief - the class implements a bilateral filter.

The function implements a bilateral filter, which filters the position of a pixel 
depending on the spatial and intensity distribution of the points in a kernel around the point.

A bilateral filter is a non-linear, edge-preserving and noise-reducing smoothing
filter for images. The intensity value at each pixel in an image is replaced by
a weighted average of intensity values from nearby pixels. This weight can be
based on a Gaussian distribution. Crucially, the weights depend not only on
Euclidean distance of pixels, but also on the radiometric differences (e.g. range
differences, such as color intensity, depth distance, etc.). This preserves sharp
edges by systematically looping through each pixel and adjusting weights to the
adjacent pixels accordingly.

See https://en.wikipedia.org/wiki/Bilateral_filter for further information.

Rafael Radkowski
Iowa State University
rafael@iastate.edu
September 27, 2017
MIT License
---------------------------------------------------------------
Last edited:

Aug 7, 2020, RR
- Added a function to return the allocated memory. 

*/


// stl
#include <stdio.h>
#include <iostream>
#include <vector>
#include <algorithm>
#include <fstream>

// cuda
#include "cuda_runtime.h"

// OpenCV
#include <opencv2\highgui.hpp>
#include <opencv2\core.hpp>


namespace texpert
{

class cuFilter
{
public:


	typedef struct Params{
		int kernel_size;  // size of the kernel in pixel. MUST BE ODD
		float sigmaI; // filter intensity sigma, [1, inf]
		float sigmaS; // filter spatial sigma, [1, inf]

		Params() {
			kernel_size = 9;
			sigmaI = 12.0;
			sigmaS = 6.0;

		}

	}Params;


	/*!
	Apply a bilateral filter on the image. 
	@param src_image_ptr -	A image of size [width x height x channels] organized as an array [v0, v1, v2, ......., V_N].
							The function expects that src_image_ptr is the image in host memory since the function copies the image to device memory. 
							The values are expected to be floats in mm. 
	@param width -			The width of the image in pixels. 
	@param height -			The height of the image in pixels. 
	@param dst_image_ptr -	A float pointer to store (return) the location of the results. Depending on the setting of to_host, the 
							pointer points to either device memory (to_host == false) or copies the entire dataset back to the host (to_host == true). 
	@param to_host			Indicate whether or not the function shoudl copy the image data back to the host. 
					
	*/
	static void ApplyBilateralFilter(float* src_image_ptr, int width, int height, int chanels, float* dst_image_ptr, bool to_host = false);


	/*!
	Set the filter params for the bilateral filter. 
	Note that the function checks the filter parameters internally. They cannot be beyond the specified range (see header). 
	@param params - struct of type params. 
	*/
	static void SetBilateralFilterParams(cuFilter::Params params);

	/*!
	Enable or disable the filter. Not that all filter functions (ApplyBilateralFilter) still do the initial from host to device copy operation. 
	dst_image_ptr returns the location of the data on the device or host depending of the to_host setting. 
	@param enable - true enables the fileter. 
	*/
	static void Enable(bool enable);

	/*!
	Init the device memory. 
	The dvice memory can come from a global device manager. 
	An internal parameter 'use_global_memory' woudl look for cuda device resources to obtain a pointer. 
	If the variable is set to false, the class creates its own memory for everything. 
	@param width - the width of the image in pixels
	@param height - the height of the image in pixels
	@param channels - the number of channels. A depth image should have only 1 channel.
	*/
	static void AllocateDeviceMemory(int width, int height, int chanels);


	/*!
	Free all device memory
	*/
	static void FreeDeviceMemory(void);


	/*!
	Return the device memory this class allocated. 
	@return int - allocated memory in bytes. 
	*/
	static int GetMemoryCount(void);


private:



};

}//namespace texpert