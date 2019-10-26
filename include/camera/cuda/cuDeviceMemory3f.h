#pragma once
/*
class cuDeviceMemory

This class manages all cuda memory that is required for image processing within the 
TrackingExpert application. It is a global storage, which is integrated as a singleton to
maintain global access to the cuda device memory from different locations of the application.

It supports the following memory:

	- Input depth image device memory, DevInImagePtr()

	- point array that stores all points extracted from the depth image, DevPointPtr()
	- normal array that stores all normal vectors generated from the depth image per point, DevNormalsPtr()
	- point image, organizes all points as an image grid, DevPointImagePtr()
	- normal image, organizes all normal vectors as an image grid, DevNormalsImagePtr()

Rafael Radkowski
Iowa State University
rafael@iastate.edu
September 27, 2017
MIT License
---------------------------------------------------------------
*/

// stl
#include <stdio.h>
#include <iostream>
#include <vector>
#include <algorithm>
#include <conio.h>

// cuda
#include "cuda_runtime.h"

// OpenCV
#include <opencv2\highgui.hpp>
#include <opencv2\core.hpp>


namespace texpert
{





class cuDevMem3f
{

	friend class cuPCU3f;
	friend class cuSample3f;

public:

	/*
	Allocate device memory for an image of width x height x channels.
	@param width - the width of the image in pixels
	@param height - the height of the image in pixels
	@param channels - the number of channels. A depth image should have only 1 channel.
	*/
	static void AllocateDeviceMemory(int width, int height, int channels);


	/*
	Frees all the memory
	*/
	static void FreeAll(void);


protected:

	/*
	Return the pointer to the input image memory;
	@return - pointer of type unsigned short to the device memory
	*/
	//static unsigned short* DevInImagePtr(void);
	static float* DevInImagePtr(void);

	/*
	Return the pointer to the output memory for points stored as float3 array;
	@return - pointer of type float3 to the device memory
	*/
	static float3* DevPointPtr(void);

	/*
	Return the pointer to the output memory for normal vectors stored as float3 array;
	@return - pointer of type float3 to the device memory
	*/
	static float3* DevNormalsPtr(void);


	/*
	Return a pointer to the memory that stores the points' x, y, z, values as array, organized as image grid
	[x0 | y0 | z0 | x1 | y1 | z1 | .... | zN | yN | zN ]
	*/
	static float* DevPointImagePtr(void);

	/*
	Return a pointer to the memory that stores the points normal vector components nx, ny, nz values as array, organized as image grid
	[nx0 | ny0 | nz0 | nx1 | ny1 | nz1 | .... | nzN | nyN | nzN ]
	*/
	static float* DevNormalsImagePtr(void);


	/*
	An array for different parameters which are required on the device
	*/
	static float* DevParamsPtr(void);

};

}//namespace texpert