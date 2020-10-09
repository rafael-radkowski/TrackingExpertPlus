#pragma once

/*
@class CPFToolsGPU

@brief This is the GPU implementation of the already-implemented CPFTools static class.

William Blanchard
Iowa State University
wsb@iastate.edu
(847) 707-1421
2 Oct 2020

MIT License
-------------------------------------------------------------------------------------------------------
Last edits:
9 October 2020
- BUGFIX: Corrected GetRefFrame alg
- BUGFIX: Corrected ponterToVecM4F function

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
	*/
	static void AllocateMemory(uint32_t size);

	/*!
	*/
	static void DeallocateMemory();

	/*!
	*/
	static float AngleBetween(const Eigen::Vector3f& a, const Eigen::Vector3f& b);

	/*!
	*/
	static void GetRefFrames(vector<Eigen::Affine3f>& dst, vector<Eigen::Vector3f>& p, vector<Eigen::Vector3f>& n);

	/*!
	*/
	static void DiscretizeCurvature(vector<uint32_t>& dst, const vector<Eigen::Vector3f>& n1, PointCloud& pc, vector<Matches> matches, const float range = 10.0);

	/*!
	*/
	static void DiscretizeCPF(vector<CPFDiscreet>& dst, vector<uint32_t>& curvatures, Matches* matches, int num_matches, vector<Eigen::Vector3f> pts, vector<Eigen::Affine3f> ref_frames);

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