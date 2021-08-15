#pragma once
/*!
@class   PointCloudGPU

@brief  This class is a GPU version of the pointcloud. STores the same points and normals, but as float3s instead of eigen vectors. Also stores voxel indices.

Features:
- GPU version of pointcloud
- stores voxel indices to transfer back easily
- easy process to convert Point cloud to intermediate storage, and intermediate storage to PointCloudGPU



Tyler Ingebrand
Iowa State University
tyleri@iastate.edu
Undergraduate

-------------------------------------------------------------------------------
Last edited:

June 5, 2020 - Tyler Ingebrand
- Added documentation

*/
#include "vector_types.h"  
#include "Types.h"

class PointCloudGPU
{
public:
	bool allocated;

	float3	*points;
	float3	*normals;

	int		*voxelIndex;
	
	size_t		numberOfPoints;
	size_t		maxNumPoints;

	float   *pose;
	float	*ITpose; //inverse transposed pose, used for finding absol normal values. Need to find it only once, so we are going to use the CPU to find it.

	//this should not be used!!! We need to know amount of space to allocate, and this does not provide it!
	PointCloudGPU();
	/*
	Destructor, deallocates space
	*/
	~PointCloudGPU();
	
	/*
	 constructor. Needs to know max possible points so it can allocate max space needed
	 @param maxPoints - The max possible points in a pointcloud we want to downsample. Example - Depth columns * Depth Height = 350,000ish points
	*/
	PointCloudGPU(int maxPoints);


	/*
	Updates our object with the pointcloud given. Transfers them to GPU
	 @param pc - the pointcloud we are interested in, and want on GPU
	 @return true if successful, false if you did not allocate enough space for the pointcloud pc when you used the constructor, or allocation otherwise failed
	*/
	bool updatePointCloudGPU(PointCloud &pc);
	
};