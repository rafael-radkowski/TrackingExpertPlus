#include "CPFRenderHelpers.h"


CPFRenderHelpers::CPFRenderHelpers()
{
	_point_size = 0;
	_scene_size = 0;
}


CPFRenderHelpers::~CPFRenderHelpers()
{

}


/*
Initialize memory
*/
void CPFRenderHelpers::init(int point_size, int scene_size)
{
	_point_size = point_size;
	_scene_size = scene_size;

	matching_pair_ids.clear();
	matching_pair_ids.resize(_point_size);

	vote_pair_ids.clear();
	vote_pair_ids.reserve(_point_size);
}

/*
Add a point pair. 
*/
void CPFRenderHelpers::addMatchingPair(int point_id, int scene_id)
{
	matching_pair_ids[point_id].push_back(make_pair(point_id, scene_id));
}

void CPFRenderHelpers::addVotePair(int point_id, int scene_id)
{
	vote_pair_ids.push_back(make_pair(point_id, scene_id));
}

bool CPFRenderHelpers::getMatchingPairs(const int point_id, std::vector< std::pair<int, int> >& matching_pairs )
{
	if(point_id >= matching_pair_ids.size()  ) return false;

	matching_pairs = matching_pair_ids[point_id];
	return true;
}

bool CPFRenderHelpers::getVotePairs(const int point_id, std::vector< std::pair<int, int> >& vote_pairs)
{
	vote_pairs.clear();

	for_each(vote_pair_ids.begin(), vote_pair_ids.end(), [&](std::pair<int,int> p){
		if(p.first == point_id)
			vote_pairs.push_back(p);
	});

	if(vote_pairs.size() ==0) return false;

	return true;
}

 std::pair<int, int>& CPFRenderHelpers::getVotePair(const int point_id)
 {
	if(point_id >= vote_pair_ids.size() ) return std::make_pair(-1,-1);
	return vote_pair_ids[point_id];
 }