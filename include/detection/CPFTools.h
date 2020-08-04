#pragma once;
/*
@class CPFTools

@brief The class implements functions to extract feature descriptors and other helpers which are required
to determine the CPF features. 


Rafael Radkowski
Iowa State University
rafael@iastate.edu
(515) 294-7044
14 Oct 2017

MIT License
-------------------------------------------------------------------------------------------------------
Last edits:


*/

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

class CPFTools
{
public:
	typedef struct CPFParam
	{
		int angle_bins;

		CPFParam() {
			angle_bins = 12;
		}

	}CPFParam;
	
	/*!
	*/
	static float AngleBetween(const Eigen::Vector3f& a, const Eigen::Vector3f& b);

	/*!
	*/
	static Eigen::Affine3f GetRefFrame(Eigen::Vector3f& p, Eigen::Vector3f& n);

	/*!
	*/
	static uint32_t DiscretizeCurvature(const Eigen::Vector3f& p1, const Eigen::Vector3f& n1, const PointCloud& pc, const MyMatches& matches, const float range = 10.0);

	/*!
	*/
	static CPFDiscreet DiscretizeCPF(const std::uint32_t& c0, const std::uint32_t& c1, const Eigen::Vector3f& p0, const Eigen::Vector3f& p1);
	
	/*!
	*/
	static void SetParam(CPFParam& param);

	/*!
	*/
	static void GetMaxMinAng(float& max, float& min);

	/*!
	
	*/
	static void Reset(void);
};