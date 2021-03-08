#pragma once

/*
@class CPFToolsGPU

@brief This is the GPU implementation of the already-CPU-implemented CPFTools static class. 

William Blanchard
Iowa State University
wsb@iastate.edu
(847) 707-1421
2 Oct 2020

MIT License
-------------------------------------------------------------------------------------------------------
Last edits:
6 November 2020
- Optimized copying of memory from the GPU to the CPU and vice versa
- Code cleanup

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

	static void AssignPointCloud(PointCloud& pc);

	static void AssignMatches(vector<Matches>& matches);

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
	Calculates the point curvatures from the given point cloud and places them in their corresponding
	locations in dst.
	@param dst - the vector in which the resulting point curvatures will be placed
	@param n1 - the normals of the points in the point cloud
	@param pc - the point cloud from which the curvatures will be found
	@param matches - the matches for each point in the point cloud found by KNN.
	@param range - 
	*/
	static void DiscretizeCurvature(vector<uint32_t>& dst, const vector<Eigen::Vector3f>& n1, PointCloud& pc, vector<Matches> matches, const double range = 10.0);

	/*!
	Calculates all discretized CPFs from a set of points and their corresponding curvatures, matches, and 
	reference frames, placing the results in dst.
	@param dst - the vector in which the resulting CPFDiscreets will be placed
	@param curvatures - the curvatures of all corresponding points in pts
	@param matches - the matches found by KNN from the points in pts
	@param pts - the points in the point cloud used to find the CPFDiscreets
	@param ref_frames - the reference frames corresponding to the points in pts
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