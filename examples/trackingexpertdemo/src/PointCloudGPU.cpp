#include "vector_types.h"  
#include <thrust/device_vector.h>
#include "PointCloudGPU.h"
#include "Types.h"

PointCloudGPU::PointCloudGPU()
{
	allocated = false;
}
PointCloudGPU::~PointCloudGPU()
{

	cudaFree(points);
	cudaFree(normals);
	cudaFree(voxelIndex);
	cudaFree(pose);
	cudaFree(ITpose);
}
PointCloudGPU::PointCloudGPU(int maxPoints)
{
		//assume we can allocate memory
		allocated = true;
		cudaGetLastError(); //clearing error message in case there was errors from other stuff that is not a problem for us

		//copy over pre allocated memory
		cudaMallocManaged((void**)&points, maxPoints * sizeof(float3));
		if(cudaGetLastError() != cudaSuccess)	allocated = false; //if there is every an error of any kind, we did not allocate memory so set it to false;
		cudaMallocManaged((void**)&normals, maxPoints * sizeof(float3));
		if (cudaGetLastError() != cudaSuccess)	allocated = false;
		cudaMallocManaged((void**)&voxelIndex, maxPoints * sizeof(int));
		if (cudaGetLastError() != cudaSuccess)	allocated = false;
		cudaMallocManaged((void**)&pose, 16 * sizeof(float));
		if (cudaGetLastError() != cudaSuccess)	allocated = false;
		cudaMallocManaged((void**)&ITpose, 16 * sizeof(float));
		if (cudaGetLastError() != cudaSuccess)	allocated = false;
		//we cant exceed this
		maxNumPoints = maxPoints;


}

bool PointCloudGPU::updatePointCloudGPU(PointCloud &pc)
{
	//if we dont have any space, or we dont have enough space for all points
		if (!allocated || pc.points.size() > maxNumPoints) return false;

		numberOfPoints = pc.points.size();
		//copy data into memory
		cudaMemcpy(points, (float3*)pc.points.data(), pc.points.size() * sizeof(float3), cudaMemcpyHostToDevice);
		cudaMemcpy(normals,(float3*)pc.normals.data(), pc.normals.size() * sizeof(float3), cudaMemcpyHostToDevice);
		Eigen::Matrix4f tempITpose = pc.pose.inverse().transpose();
		for (int row = 0; row < 4; row++)
		{
			for (int col = 0; col < 4; col++)
			{
				pose[row*4+col] = pc.pose(row,col);
				ITpose[row * 4 + col] = tempITpose(row,col);
			}
		}

		return true;
}
	


//int PointCloudGPU::getCurrentNumPoints() { return numberOfPoints; }
//float3* PointCloudGPU::getPoints() { return points; }
//float3* PointCloudGPU::getNormals() { return normals; }
//int* PointCloudGPU::getVoxelIndices() { return voxelIndex; }
//float* PointCloudGPU::getPose() { return pose; }
//float* PointCloudGPU::getITpose() { return ITpose; }