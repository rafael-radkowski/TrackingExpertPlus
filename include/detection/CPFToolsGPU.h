#pragma once

/*
@class CPFToolsGPU

@brief This is the GPU implementation of the already-CPU-implemented CPFTools static class.

WARNING: When building, add "-fmad=false" to the CUDA command line in the trackingx project properties.
fma is short for "fused multiply add."  CUDA does this operation in a non-IEEE standard way, and so
there are some calculations, such as in DiscretizeCurvature, that are correct in debug mode (where fmad
is disabled by default) but incorrect in release mode (where fmad is enabled by default).  

William Blanchard
Iowa State University
wsb@iastate.edu
(847) 707-1421
2 Oct 2020

MIT License
-------------------------------------------------------------------------------------------------------
Last edits:
15 October 2020
- BUGFIX: Fixed case where ang variable in DiscretizeCPF would be calculated as nan by replacing with 0
- FEATURE: Added CUDA error statements
- Added some function documentation

*/

#include <cuda_runtime.h>
#include <vector_types.h>

#include <iostream>
#include <vector>
#include <algorithm>
#include <math.h>

#include <Eigen\Dense>
#include <Eigen\Geometry>

#include "Types.h"
#include "CPFTypes.h"
#include "KNN.h"

using namespace texpert;

class CPFToolsGPU
{
public:
	typedef struct CPFParamGPU
	{
		int angle_bins;

		CPFParamGPU() {
			angle_bins = 12;
		}

	}CPFParamGPU;

	/*!
	Allocates memory for *size* points in a point cloud
	@param size - the number of points in a point cloud this class should use
	*/
	static void AllocateMemory(uint32_t size);

	/*!
	Deallocates all memory used for this class
	*/
	static void DeallocateMemory();

	/*!
	Finds the angle between two vectors
	@param a - vector to calculate angle from
	@param b - vector to calculate angle to
	*/
	static float AngleBetween(const Eigen::Vector3f& a, const Eigen::Vector3f& b);

	/*!
	Finds the reference frames for all given points, given their respective normals.
	The indices of the points should match the indices of their normals.
	@param dst - the destination of the resulting Eigen Affine3f matrices, holding the ref frames
	@param p - the list of points to find reference frames from
	@param n - the list of normals, whose indices correspond with their respective points
	*/
	static void GetRefFrames(vector<Eigen::Affine3f>& dst, vector<Eigen::Vector3f>& p, vector<Eigen::Vector3f>& n);

	/*!
	*/
	static void DiscretizeCurvature(vector<uint32_t>& dst, const vector<Eigen::Vector3f>& n1, PointCloud& pc, vector<Matches> matches, const float range = 10.0);

	/*!
	*/
	static void DiscretizeCPF(vector<CPFDiscreet>& dst, vector<uint32_t>& curvatures, vector<Matches> matches, vector<Eigen::Vector3f> pts, vector<Eigen::Affine3f> ref_frames);

	/*!
	*/
	static void SetParam(CPFParamGPU& param);

	/*!
	*/
	static void GetMaxMinAng(float& max, float& min);

	/*!

	*/
	static void Reset(void);
};