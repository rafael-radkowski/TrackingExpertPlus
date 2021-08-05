#pragma once

//stl 
#include <iostream>
#include <string>
#include <algorithm>
#include <vector>
#include <iostream>
#include <unordered_map>

// Eigen
#include <Eigen/Dense>
#include <Eigen/Geometry>

// local
#include "Types.h"
#include "CPFTypes.h"
#include "CPFTools.h"
#include "KNN.h"


namespace texpert{

class CPFRenderHelpers
{
public:
	CPFRenderHelpers();
	~CPFRenderHelpers();


	/*
	Initialize memory
	*/
	void init(int point_size, int scene_size);

	/*
	Add a point pair. 
	*/
	void addMatchingPair(int object_id, int scene_id);


	bool getMatchingPairs(const int point_id, std::vector< std::pair<int, int> >& matching_pairs );


	void addVotePair(int object_id, int scene_id);


	bool getVotePairs(const int point_id, std::vector< std::pair<int, int> >& vote_pairs);

	std::pair<int, int>& getVotePair(const int point_id);



private:

	// contains the matching pair ids for all points <object id, scene id>
	// The first vector is the point id. 
	std::vector< std::vector< std::pair<int, int> >	>	 matching_pair_ids;

	// contains the pairs that won the voting match. 
	// <model, scene> point ids
	std::vector< std::pair<int, int> >					vote_pair_ids;

	std::pair<int,int> null_return;
	int _point_size;
	int _scene_size;
};

}//namespace texpert{
