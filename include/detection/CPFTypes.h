#pragma once

// STL
#include <iostream>
#include <string>
#include <vector>


// Eigen
//#include <Eigen/Dense>


namespace texpert{


typedef struct _CPFDiscreet
{

	std::uint32_t data[4];	

	int point_idx;
	float alpha;

	bool operator==(const _CPFDiscreet& d) const {
			return (d[0] == data[0]) &&
				(d[1] == data[1]) &&
				(d[2] == data[2]) &&
				(d[3] == data[3]);
		}

	std::uint32_t operator[](const int i) const { return data[i]; }

	std::uint32_t& operator[](const int i) { return data[i]; }

	_CPFDiscreet(	std::uint32_t a, std::uint32_t b, 
			std::uint32_t c, std::uint32_t d) {
		data[0] = a;
		data[1] = b;
		data[2] = c;
		data[3] = d;
		point_idx = -1;
		alpha = 0.0;
	}

	
	_CPFDiscreet()
	{
		data[0] = 0;
		data[1] = 0;
		data[2] = 0;
		data[3] = 0;
		point_idx = -1;
		alpha = 0.0;
	}

}CPFDiscreet;

}

// Map for all ppfs 
//typedef std::unordered_multimap<CPFFeatureDiscreet, int> CPFMap;


typedef struct CPFParams
{
	// knn search radius
	float search_radius;

	// descriptor histogram bin range in angle degree,
	float	angle_step;

	float	multiplier;

	float	cluster_trans_threshold;
	float	cluster_rot_threshold;

	CPFParams() {
		multiplier = 10.0;
		search_radius = 0.1;
		angle_step = 12.0;
		cluster_trans_threshold = 0.03;
		cluster_rot_threshold = 0.8;
	}

}CPFParams;


// to be backward compatible with the old name
typedef CPFParams CPFToolsParams;