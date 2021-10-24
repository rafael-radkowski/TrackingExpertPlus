#include "CPFClustering.h"

using namespace texpert_experimental;



//static 
int CPFClustering::Clustering(CPFMatchingData& src_data, CPFClusteringData& dst_clustering, CPFClusteringParam& params)
{
	dst_clustering.clear();

	if (src_data.getPoseCandidates().size() == 0) {
		if (params.verbose){
			std::cout << "[INFO] - CPFMatchingExp: Found " << src_data.getPoseCandidates().size() << " pose candidates." << std::endl;
			}
		return false;
	}

	src_data.cluster_clear();
	int cluster_count = 0;

	for (int i = 0; i < src_data.getPoseCandidates().size(); i++) {

		bool cluster_found = false;

		Eigen::Affine3f pose = src_data.getPoseCandidates()[i];

		// check whether the new pose fits into an existing cluster
		for (int j = 0; j < src_data.getPoseCandidates().size(); j++) {

			if (similarPose(pose, dst_clustering.getPoseCluster()[j].front(), params)) {
				cluster_found = true;
				dst_clustering.getPoseCluster()[j].push_back(pose); // remember the cluster
				dst_clustering.getPoseClusterVotes()[j].first += src_data.getPoseCandidatesVotes()[i]; // count the votes

				// RENDER HELPER
				//if (m_render_helpers) {
				//	data.m_debug_pose_candidates_id[j].push_back(i); // for debugging. Store the pose candiate id. 
				//}
			}

		}

		// create a new cluster 
		if (!cluster_found) {
			std::vector<Eigen::Affine3f> cluster;
			cluster.push_back(pose);
			dst_clustering.getPoseCluster().push_back(cluster); // remember the cluster
			dst_clustering.getPoseClusterVotes().push_back(make_pair(src_data.getPoseCandidatesVotes()[i], cluster_count)); // count the votes
			cluster_count++;

	
		}



	}

	// sort the cluster
	std::sort(dst_clustering.getPoseClusterVotes().begin(), dst_clustering.getPoseClusterVotes().end(),
		[](const std::pair<int, int>& a, const std::pair<int, int>& b) {
			return a.first > b.first; });


	if (params.verbose) {
		std::cout << "[INFO] - CPFMatchingExp: Found " << dst_clustering.getPoseCluster().size() << " pose clusters." << std::endl;
	}

	return true;

}




//static 
int CPFClustering::GetBest(CPFClusteringData& dst_clustering, CPFClusteringParam& params)
{
	// get the siz best hits
	int hits = 12;
	hits = (int)dst_clustering.getPoseCluster().size() - 1;


	for (int i = 0; i < hits; i++) {
		// <votes, cluster id> in pose_cluster
		std::pair< int, int>  votes_cluster_index = dst_clustering.getPoseClusterVotes()[i];

		combinePoseCluster(dst_clustering.getPoseCluster()[votes_cluster_index.second], votes_cluster_index.first, dst_clustering, params, false);

	}

	return 1;
}




bool CPFClustering::similarPose(Eigen::Affine3f a, Eigen::Affine3f b, CPFClusteringParam& param)
{
	float translation_threshold = 0.03f;
	float rotation_threshold = 0.8f;

	// distance between the two positions. 
	float delta_t = (a.translation() - b.translation()).norm();

	// angle delta
	Eigen::AngleAxisf aa(a.rotation().inverse() * b.rotation());
	float delta_r = fabsf(aa.angle());

	return delta_t < param.custering_threshold_t&& delta_r < param.custering_threshold_R;
}




bool CPFClustering::combinePoseCluster(std::vector<Eigen::Affine3f>& pose_clustser, int votes, CPFClusteringData& dst_data, CPFClusteringParam& params, bool invert )
{
	if (pose_clustser.size() == 0) {
		if (params.verbose) {
			std::cout << "[ERROR] - CPFClustering:combinePoseCluster: No pose clusters give ( " << pose_clustser.size() << " )." << std::endl;
		}
		return false;
	}

	size_t size = pose_clustser.size();
	Eigen::Matrix4Xf rot_accum(4, size);
	Eigen::Vector3f trans_avg(0.0, 0.0, 0.0);
	size_t cnt = 0;
	// get an average pose for the cluster with the hightest number of votes.
	// NOTE: If we wanted we could make this a weighted averaging by storing the number of votes for each pose in the cluster, but currently this is not store. 
	for_each(pose_clustser.begin(), pose_clustser.end(),
		[&](const Eigen::Affine3f& p) {
			trans_avg += p.translation();
			//rot_accum(Eigen::all,cnt) = Eigen::Quaternionf(p.rotation()).coeffs()/size; // potentially multiply by weights (which add to 1) rather than dividing by size (can't use this line with Eigen 3.3... 
			Eigen::Map<Eigen::Vector4f>(rot_accum.data() + cnt * 4, 4) = Eigen::Quaternionf(p.rotation()).coeffs() / size; // potentially multiply by weights (which add to 1) rather than dividing by size
			cnt++;
		});

	// (weighted) average is the e-vector corresponding to the largest eigenvalue of the product of (rot_accum * rot_accum')
	Eigen::EigenSolver<Eigen::Matrix4Xf> prod_eigen(rot_accum * rot_accum.transpose());
	Eigen::Vector4f::Index max_eval_index;
	Eigen::abs2(prod_eigen.eigenvalues().array()).maxCoeff(&max_eval_index);

	trans_avg /= (float)size;
	//Eigen::Vector4f rot_avg = prod_eigen.eigenvectors()(Eigen::all,max_eval_index); // Can't use this line with Eigen 3.3 so...
	//assert(!prod_eigen.eigenvectors().isRowMajor()); // assume to be column major (Eigen default)


	Eigen::Vector4cf rot_avg_complex = Eigen::Map<Eigen::Vector4cf>(prod_eigen.eigenvectors().data() + max_eval_index * 4);

	assert(rot_avg_complex.real().squaredNorm() > 1e6 * rot_avg_complex.imag().squaredNorm());

	Eigen::Vector4f rot_avg = rot_avg_complex.real();

	Eigen::Affine3f pose;
	pose = Eigen::Affine3f::Identity();
	pose.linear().matrix() = Eigen::Quaternionf(rot_avg).normalized().toRotationMatrix();
	pose.translation().matrix() = trans_avg;


	// Inverse to move the model into the scene point cloud. 
	if (invert)
		pose = pose.inverse();

	dst_data.getPoses().push_back(pose);
	dst_data.getPoseVotes().push_back(votes);

	return true;
}

