#ifndef __FDTYPES__
#define __FDTYPES__
/*
class FDTypes

The class implements feature descriptors types which are required for PPF and CPF

PPF is according to 
Bertram Drost et al., Model Globally, Match Locally: Efficient and Robust 3D Object Recognition
http://campar.in.tum.de/pub/drost2010CVPR/drost2010CVPR.pdf

Rafael Radkowski
Iowa State University
rafael@iastate.edu
MIT License
---------------------------------------------------------------
Last Changes:


*/

// stl;
#include  <unordered_map>

// local
#include "nurmur.h"

// Eigen 
#include "Eigen/Dense"


namespace texpert {

	/****************************************************************************


	*/
	typedef struct _PPFDiscreet {
		
		std::uint32_t data[4];

		bool operator==(const _PPFDiscreet& ppf) const {
			return (ppf[0] == data[0]) &&
				(ppf[1] == data[1]) &&
				(ppf[2] == data[2]) &&
				(ppf[3] == data[3]);
		}

		std::uint32_t operator[](const int i) const { return data[i]; }

		std::uint32_t& operator[](const int i) { return data[i]; }

		int point_index = -1;
		
	} PPFDiscreet; // struct PPFDiscreet


	/****************************************************************************


	*/
	typedef  struct _VotePair
	{
		
		int				model_i;  // model point id
		float			alpha_m;

		_VotePair(const int m_i, const float a_m)
			: model_i(m_i), alpha_m(a_m) {}
			
	}VotePair;



	// Map for all ppfs 
	typedef std::unordered_multimap<PPFDiscreet, VotePair> PPFMap;



	typedef struct Pose_ {
		int votes;
		Eigen::Affine3f t;

		// model associations
		int from_model_idx;
		int to_scene_idx;
	}Pose;

}


/*/****************************************************************************
hash for PPFDiscreet, this allows one to add it into a map.
*/


namespace std {
	template <>
	struct hash<texpert::PPFDiscreet> {
		std::size_t operator()(const texpert::PPFDiscreet& ppf) const {
			return murmurppf(ppf.data);
		}
	}; // struct hash
} // namespace std


#endif