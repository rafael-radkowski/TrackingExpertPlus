#pragma once

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
	static Eigen::Affine3f GetRefFrame(vector<Eigen::Vector3f>& p, vector<Eigen::Vector3f>& n);

	/*!
	*/
	static uint32_t DiscretizeCurvature(const Eigen::Vector3f& p1, const Eigen::Vector3f& n1, const PointCloud& pc, const MyMatches& matches, const float range = 10.0);

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