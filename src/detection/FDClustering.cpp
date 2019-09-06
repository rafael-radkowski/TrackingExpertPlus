#include "FDClustering.h"


using namespace isu_ar;


FDClustering::FDClustering()
{
	_translation_threshold = 0.05;
	_rotation_threshold = 0.2;
	_invert = false;
}


FDClustering::~FDClustering()
{
	
}


bool FDClustering::clusterPoses(const std::vector<Pose>& poses, vector<Pose>& pose_clusters, vector<int>& votes)
{
	votes.clear();
	pose_clusters.clear();

	int cluster_index = 0; // index to align the cluster votes and the clustered poses.

	_temp_cluster_votes.clear();
	_temp_clustered_poses.clear();

	std::vector<Pose>::const_iterator itr = poses.begin();
	while (itr != poses.end()) {

		bool cluster_found = false;

		int cluster_count = 0;
		// check whether the new pose fits into an existing cluster
		for_each(_temp_clustered_poses.begin(), _temp_clustered_poses.end(), [&](std::vector<Pose>& cluster) {
			if (similar((*itr).t, cluster.front().t)) {
				cluster_found = true;
				cluster.push_back(((*itr)));
				_temp_cluster_votes[cluster_count].first += (*itr).votes; // count the number of votes
				//_temp_cluster_votes[cluster_count].second++; // count the number of poses
			}
			cluster_count++;
		});

		// create a new cluster 
		if (!cluster_found) {
			std::vector<Pose> cluster;
			cluster.push_back((*itr));
			_temp_clustered_poses.push_back(cluster);
			_temp_cluster_votes.push_back(make_pair((*itr).votes, cluster_index));
			cluster_index++;
		}


		itr++;
	}

	// sort the cluster
	std::sort(_temp_cluster_votes.begin(), _temp_cluster_votes.end(), 
			  [](const std::pair<int, int>& a, const std::pair<int, int>& b) {
				return a.first > b.first;});

	int winner_idx = _temp_cluster_votes.front().second;
	int winner_votes = _temp_cluster_votes.front().first;


	Eigen::Vector4f rot_avg(0.0, 0.0, 0.0, 0.0);
	Eigen::Vector3f trans_avg(0.0, 0.0, 0.0);

	// get an avarage pose for the cluster with the hightest number of votes.
	for_each(_temp_clustered_poses[winner_idx].begin(), _temp_clustered_poses[winner_idx].end(),
		[&](const Pose& p) {
			trans_avg += p.t.translation();
			rot_avg += Eigen::Quaternionf(p.t.rotation()).coeffs();
			});
	
	size_t size = _temp_clustered_poses[winner_idx].size();

	trans_avg /= (float)size;
	rot_avg /= (float)size;

	Pose winner;
	winner.t = Eigen::Affine3f::Identity();
	winner.t.linear().matrix() = Eigen::Quaternionf(rot_avg).normalized().toRotationMatrix();
	winner.t.translation().matrix() = trans_avg;
	winner.votes = winner_votes;

	// Inverse to move the model into the scene point cloud. 
	if(_invert)
		winner.t = winner.t.inverse();

	votes.push_back(winner_votes);
	pose_clusters.push_back(winner);

	return true;
}


bool FDClustering::similar(Eigen::Affine3f a, Eigen::Affine3f b)
{
	// distance between the two positions. 
	float delta_t = (a.translation() - b.translation()).norm();

	// angle delta
	Eigen::AngleAxisf aa(a.rotation().inverse() * b.rotation());
	float delta_r = fabsf(aa.angle());

	return delta_t < _translation_threshold && delta_r < _rotation_threshold;
}


void FDClustering::setTranslationThreshold(float value)
{
	if (value > 0) {
		_translation_threshold = value;
	}
}

void FDClustering::setRotationThreshold(float value)
{
	if (value > 0) {
		_rotation_threshold = value / 180.0 * 3.14159265359;
	}
}

void FDClustering::invertPose(bool invert)
{
	_invert = invert;
}