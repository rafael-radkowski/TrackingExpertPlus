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
2 October 2020
- Added overall class documentation
- Added GetRefFrameGPU kernel for use in GetRefFrame

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
	static float AngleBetween(const Eigen::Vector3f& a, const Eigen::Vector3f& b);

	/*!
	*/
	static void GetRefFrames(vector<Eigen::Affine3f>& dst, vector<Eigen::Vector3f>& p, vector<Eigen::Vector3f>& n);

	/*!
	*/
	static void DiscretizeCurvature(vector<uint32_t>& dst, const vector<Eigen::Vector3f>& n1, PointCloud& pc, const Matches* matches, const float range = 10.0);

	/*!
	*/
	static CPFDiscreet DiscretizeCPF(const std::uint32_t& c0, const std::uint32_t& c1, const Eigen::Vector3f& p0, const Eigen::Vector3f& p1);

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