#pragma once

#include "cuda_runtime.h"
#include <vector>
#include "PointCloudGPU.h"
/*!
@class   GPUvoxelDownsample

@brief  This class implements methods in CUDA and c++ to downsample a pointcloud using voxel algorithms. 

Allows you to calculate absolute values for points, normals, and voxel indices quickly on the GPU.

Features:
- Calculate abolsute values
- Calculate voxel indices
- Copy data back to CPU



Tyler Ingebrand
Iowa State University
tyleri@iastate.edu
Undergraduate

-------------------------------------------------------------------------------
Last edited:

May 26, 2020, Tyler Ingebrand
- Updated algoritm, documentation
*/

class GPUvoxelDownsample{

	public:
		/*
		constructor - should not be used;
		*/
		GPUvoxelDownsample();

		
		/*
		constructor. Allocates space for rows*columns number of points for each PointCloud.
		@param numCams - The number of cameras or videos we need space for
		@param rows - The number of rows of pixels in depth image. THis is needed to calculate max possible number of points, since it cannot exceed 1 point per pixel
		@param columns - The number of columns of pixels in depth image. THis is needed to calculate max possible number of points, since it cannot exceed 1 point per pixel
		*/
		GPUvoxelDownsample(size_t numCams, int rows, int columns);
		
		/*!
		set boundaries and voxel size before downsampling	
		@param min - The minimum boundaries in XTZ directions as float3
		@param max - The maximum boundaries in XTZ directions as float3
		@param voxelSize - The width of each voxel
		*/
		void setBoundaries(float3 min, float3 max, float voxelSize);

		
		
		/*!
		Moves pointclouds in vector src to GPU, moves the points into global space, get voxel indices,
		@param src - The vector of pointclouds that we want to voxel downsample
		@param numberPC - The number of pointclouds in Src
		*/
		void voxelDownSample(vector<PointCloud*>& src);
		/*!
		Moves pointclouds back from GPU to CPU. They pointclouds have been moved to global space
		@param dest - The vector of pointclouds that we want to copy back to
		@param numberPC - The number of pointclouds in dest
		*/
		void copyBackToHost(vector<PointCloud*>& dest);
		/*!
		Done on CPU. Iterates through voxel indices, stored internally, and moves a single point per voxel to the DownSampled Pointcloud..
		Must have called setBoundaries, voxelDownsample, and copyBackToHost first
		@param src - The vector of pointclouds we are downsampling from
		@param numPC - The number of pointclouds in src
		@param DownsampledPointCloud - The pointcloud containing the new points, in no order, but 1 point per voxel only. Points are alloacted on first come,
			first serve basis, so which point is chosen may vary.

		*/
		void removeDuplicates(vector<PointCloud*>& src, PointCloud* DownsampledPointCloud);
		/*!
		Returns whether or not the boundaries have changed. this can be used to create an autosizeing voxel array, that resets variables when the voxel array changes
		This effectively expands the voxel array as needed.
		@return -bool if we need to recreate the voxel grid or not

		*/
		bool needToReset();

		/*!
		Returns the current min boundaries
		@return float3 with minX, minY, minZ

		*/
		float3 getMin();
		/*!
		Returns the current min boundaries
		@return float3 with maxX, maxY, maxZ

		*/
		float3 getMax();

		/*
		Returns if object was properly init using the correct constructor
		@return true if correct constructor used
		*/
		bool properConstructorUsed();

	private:
		//variables
		float3 min;
		float3 max;
		float voxelSize;

		bool needsReset; //call this if the dimensions change
		bool properlyInit; //true if right constructor used
		//for voxel downsampling
		std::vector<PointCloudGPU*> pcGPUs;
		//destination for voxels on host
		std::vector<int*> voxelDestination;
		vector<bool> voxelArray;
		vector<int> voxelsToReset;
		/*
		Inserts the value into the sorted list. Works in LogN. Does not insert element if it is already in list.
		Returns true if succesfful, false if element was already in list
		*/
		static bool insertIntoSortedList(vector<int>& list, int value);



		

};