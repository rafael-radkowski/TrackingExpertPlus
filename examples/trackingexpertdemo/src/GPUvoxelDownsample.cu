#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "GPUvoxelDownsample.h"


#include <chrono>
#include <vector>
#include <unordered_map>

#include "PointCloudGPU.h"
#include "Types.h"
#include <vector>

#include <stdio.h>
#include "EasyTimer.h"


/*!
	Cuda kernal for calculating absolute values for points and normals. Multiplies them by the pose and inverse transposed pose (ITpose), respectively.
	@param pc - the point cloud, stored in cuda allocated memory, to be processed. Values are put back into pointcloud
*/
__global__ void calculateAbsolutePointsNormals(float3* points, float3* normals, float* pose, float* ITpose, size_t pointCount)
{
	int index = blockIdx.x*blockDim.x + threadIdx.x; // current block * its width + thread index
	//int stride = blockDim.x * gridDim.x; //width of block times number of blocks

	if (index >= pointCount)
	{
		return;
	}
	
	float x = points[index].x;
	float y = points[index].y;
	float z = points[index].z;

	points[index].x = (x * pose[0]) + (y * pose[1]) + (z * pose[2]) + pose[3]; //X absol, 
	points[index].y = (x * pose[4]) + (y * pose[5]) + (z * pose[6]) + pose[7]; //Y absol, 
	points[index].z = (x * pose[8]) + (y * pose[9]) + (z * pose[10]) + pose[11]; //Z absol, 

	x = normals[index].x;
	y = normals[index].y;
	z = normals[index].z;
	if (isnan(x)) x = 0.0;
	if (isnan(y)) y = 0.0;
	if (isnan(z)) z = 0.0;

	normals[index].x = (x * ITpose[0]) + (y * ITpose[1]) + (z * ITpose[2]); //X absol, 
	normals[index].y = (x * ITpose[4]) + (y * ITpose[5]) + (z * ITpose[6]); //Y absol, 
	normals[index].z = (x * ITpose[8]) + (y * ITpose[9]) + (z * ITpose[10]); //Z absol, 
		
	


}


/*!
	Cuda kernal for calculating voxel index of each point. Must already be translated into absolute coordinates. sets to -1 if it is out of bounds, or a 0 vector
	@param pc - the point cloud, stored in cuda allocated memory, to be processed. Values are put back into pointcloud(voxelIndices variable). 
	@param xMin - x min boundary for point cloud
	@param yMin - y min boundary for point cloud
	@param zMin - z min boundary for point cloud
	@param xMax - x max boundary for point cloud
	@param yMax - y max boundary for point cloud
	@param zMax - z max boundary for point cloud
	@param xSize - width of voxel in x direction
	@param ySize - width of voxel in y direction
	@param zSize - width of voxel in z direction
*/
__global__ void calculateVoxelIndex(float3* points, int* voxelIndex, float3 min, float3 max, float voxelSize, size_t pointCount)
{
	int index = blockIdx.x*blockDim.x + threadIdx.x; // current block * its width + thread index


	if (index >= pointCount) return;//thread check

	if (points[index].x <= min.x || points[index].x >= max.x || //out of bounds in x or y direction, and in bounds in Z direction
		points[index].y <= min.y || points[index].y >= max.y ||
		points[index].z <= min.z || points[index].z >= max.z)
	{
		voxelIndex[index] = -1; // out of bounds
		return;
	}

		//if point is 0ish, set voxel to -1. we have a lot of extra 0 vectors
	 if (	fabs(points[index].x) < .001f &&
			fabs(points[index].y) < .001f &&
			fabs(points[index].z) < .001f)
	{
		voxelIndex[index] = -1;
		return;
	}

	//otherwise, find xyz index, then convert to 1d index.
	int indexX = (points[index].x - min.x) / voxelSize;
	int indexY = (points[index].y - min.y) / voxelSize;
	int indexZ = (points[index].z - min.z) / voxelSize;

	int countX = std::ceil((max.x - min.x) / voxelSize);
	int countY = std::ceil((max.y - min.y) / voxelSize);

	voxelIndex[index] = indexX + (indexY * countX) + indexZ * (countX * countY);
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/*!
	Downsamples and stores values in intermediate storage inter.
	@param inter - The matrix of intermediate storage structs that contains the initial and final values
	@param numberPC - The number of point clouds in inter

	@param xMin - x min boundary for point cloud
	@param yMin - y min boundary for point cloud
	@param zMin - z min boundary for point cloud
	@param xMax - x max boundary for point cloud
	@param yMax - y max boundary for point cloud
	@param zMax - z max boundary for point cloud
	@param xSize - width of voxel in x direction
	@param ySize - width of voxel in y direction
	@param zSize - width of voxel in z direction

	@param cudaMem - preallocated cuda memory class. Stores pointers to pre allocated cuda memory, so we do not have to reallocate constantly
	
*/
void GPUvoxelDownsample::voxelDownSample(vector<PointCloud*>& src)
{
	//detirmine how many blocks to use, and thread counts. Needs to be optimized by trial and error
	// THIS MUST BE DYMANIC, OR IT WILL NOT PROCESS ALL POINTS
	int threadsPerBlock = 64; //max number of threads per block according to cuda 10.2 = 1024
	int numBlocks = (int)std::ceil(src[0]->size() / ((double)threadsPerBlock)); //rounds up
	
	size_t numberPC = src.size();
	//create list of index
	for (size_t i = 0; i < numberPC; i++)
		if(pcGPUs[i]->updatePointCloudGPU(*src[i]) == false) return; //if we cannot update pc, we have allocation issue, no data, return;

	//convert to absolute coordinates
	//skip i = 0 because pc 0 is considered global already. (pose is idenity matrix, so multiplying does nothing.)
	for (size_t i = 0; i < numberPC; i++)
	{
		calculateAbsolutePointsNormals <<<numBlocks, threadsPerBlock >>> (pcGPUs[i]->points, pcGPUs[i]->normals, pcGPUs[i]->pose, pcGPUs[i]->ITpose, pcGPUs[i]->numberOfPoints);
	}
	//find voxel index
	for (size_t i = 0; i < numberPC; i++)
	{
		calculateVoxelIndex <<<numBlocks, threadsPerBlock >>> (pcGPUs[i]->points, pcGPUs[i]->voxelIndex, min, max, voxelSize, pcGPUs[i]->numberOfPoints);
	}
}



void GPUvoxelDownsample::copyBackToHost(vector<PointCloud*>& dest)
{
	size_t numberPC = dest.size();

	for (size_t i = 0; i < numberPC; i++)
	{
		if (pcGPUs[i]->allocated == false)
		{
			dest[i]->points.resize(0);
			dest[i]->normals.resize(0);
			continue;
		} // if the pcGPU is not allocated properly, do nothing, make PC empty

		dest[i]->points.resize(pcGPUs[i]->numberOfPoints);
		dest[i]->normals.resize(pcGPUs[i]->numberOfPoints);
		
		cudaMemcpy(dest[i]->points.data(), pcGPUs[i]->points, pcGPUs[i]->numberOfPoints * sizeof(float3), cudaMemcpyDeviceToHost);
		cudaMemcpy(dest[i]->normals.data(), pcGPUs[i]->normals, pcGPUs[i]->numberOfPoints * sizeof(float3), cudaMemcpyDeviceToHost);
		cudaMemcpy(voxelDestination[i], pcGPUs[i]->voxelIndex, pcGPUs[i]->numberOfPoints * sizeof(int), cudaMemcpyDeviceToHost);

	}

}

GPUvoxelDownsample::GPUvoxelDownsample()
{
	needsReset = false;
	properlyInit = false;
}
GPUvoxelDownsample::GPUvoxelDownsample(size_t numCams, int rows, int columns)
{
	min = make_float3(-1.0f, -1.0f, -1.0f);
	max = make_float3(1.0f, 1.0f, 1.0f);
	voxelSize = .1f;
	needsReset = false;
	properlyInit = true;

	//init cuda
	for (size_t i = 0; i < numCams; i++)
	{
		pcGPUs.push_back(new PointCloudGPU(rows*columns));
		voxelDestination.push_back((int*)malloc(rows*columns * sizeof(int)));
		for (int j = 0; j < rows*columns; j++)
			voxelDestination[i][j] = -1;

	}

}

void GPUvoxelDownsample::setBoundaries(float3 min, float3 max, float voxelSize)
{
	this->min = min;
	this->max = max;
	this->voxelSize = voxelSize;
	int countX = (int)std::ceil((max.x - min.x) / voxelSize);
	int countY = (int)std::ceil((max.y - min.y) / voxelSize);
	int countZ = (int)std::ceil((max.z - min.z) / voxelSize);


	voxelArray = vector<bool>(countX*countY*countZ, false);
	needsReset = false;

}

void GPUvoxelDownsample::removeDuplicates(vector<PointCloud*>& src,  PointCloud* DownsampledPointCloud)
{
	//get number PC
	size_t numPC = src.size();
	
	//clear old points
	DownsampledPointCloud->points.clear();
	DownsampledPointCloud->normals.clear();

	//this hashtable is a sparse table, meaning it stores no information on empty voxels. This saves time and space. 
	unordered_map<int, pair< Eigen::Vector3f, Eigen::Vector3f>> hashTable = unordered_map<int, pair< Eigen::Vector3f, Eigen::Vector3f>>();

	//for each pc
	for (size_t pc = 0; pc < numPC; pc++)
	{
		//for each point
		for (size_t i = 0; i < src[pc]->size(); i++) 
		{
			int thisIndex = voxelDestination[pc][i]; //get the precalculated voxel index (1D)
			if (thisIndex >= 0) //negative points are out of bounds (invalid), skip them
			{
				hashTable[thisIndex] =  pair< Eigen::Vector3f, Eigen::Vector3f> (src[pc]->points[i], src[pc]->normals[i]);

			}
		}
	}

	for (auto iter = hashTable.begin(); iter != hashTable.end(); ++iter)
	{
		DownsampledPointCloud->points.emplace_back(iter->second.first); //the first second refers to the key value pair, so we want the value, which is also a pair of point, normal
		DownsampledPointCloud->normals.emplace_back(iter->second.second);

	}

}

bool GPUvoxelDownsample::properConstructorUsed()
{
	return properlyInit;
}
bool GPUvoxelDownsample::needToReset()
{
	return needsReset;
}

float3 GPUvoxelDownsample::getMin() {
	return min;
}

float3 GPUvoxelDownsample::getMax()
{
	return max;
}

bool GPUvoxelDownsample::insertIntoSortedList(vector<int>& list, int value)
{
	if (list.size() == 0) //empty list
	{
		list.emplace_back(value);
		return true;
	}
	int upperIndex = list.size();
	int lowerIndex = 0;
	while (upperIndex != lowerIndex)
	{
		int middleIndex = (upperIndex + lowerIndex) / 2;
		int currentValue = list[middleIndex];
		if (value < currentValue)upperIndex = middleIndex;
		else if (value > currentValue) lowerIndex = middleIndex + 1;
		else return false;
	}

	list.insert(list.begin() + lowerIndex, value); //upperIndex = lowerIndex = where it needs to go
	return true;
}
