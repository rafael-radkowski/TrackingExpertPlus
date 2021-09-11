#include "CPFMatching.h"


#define M_PI 3.14159265359

namespace ns_CPFMatching
{


	KNN*	knn;



};

using namespace ns_CPFMatching;

/*
The class calculates descriptors for the reference model(s).
It uses the point cloud as stored in data, and populates the variables 'descriptors' and 'curvatures'
@param data - CPF model data
@param params - a parameter set.
*/
//static 
void CPFMatching::CalculateDescriptors(CPFModelData& data, CPFMatchingParams params)
{
	CPFTools::CPFParam param;
	param.angle_bins = params.cpf_angle_bins;

	CPFTools::SetParam(param);

	//----------------------------------------------------------------------------------------------------------
	// nearest neighbors

	std::vector<Matches> matches;
	size_t s = data.getPointCloud().size();


	if(knn == NULL) knn = new KNN();
	knn->populate(data.getPointCloud());

	matches.clear();

	knn->radius(data.getPointCloud(), params.knn_serach_radius, matches);

	//----------------------------------------------------------------------------------------------------------
	// Calculate point curvatures

	data.clear();

	PointCloud& pc = data.getPointCloud();
	CPFCurvatureVec& c = data.getCurvature();
	CPFDiscreetVec& d = data.getDescriptor();


	for (int i = 0; i < s; i++) {

		uint32_t curv = CPFTools::DiscretizeCurvature(pc.points[i], pc.normals[i], pc, matches[i], params.cpf_multiplier);
		c.push_back(curv);
	}

	//----------------------------------------------------------------------------------------------------------
	// Calculate the descriptor
	d.clear();
	d.reserve(s);
	for (int i = 0; i < s; i++) {
		uint32_t cur1 = c[i];

		// find the reference frame for this point
		Eigen::Affine3f T = CPFTools::GetRefFrame(pc.points[i], pc.normals[i]);

		// current knn matches length as defined in Cuda_Types.h
		const int num_matches = KNN_MATCHES_LENGTH;

		for (int j = 0; j < num_matches; j++) {
			if (matches[i].matches[j].distance > 0.0) {

				int id = matches[i].matches[j].second;

				// ToDo: Check this. Id cannot be larger than c.size(). Perhaps old data. 
				if(id>=c.size()) continue;
				
				uint32_t cur2 = c[id];

				// Move the point p1 into the coordinate frame of the point p0
				Eigen::Vector3f pt = T * pc.points[id];

				CPFDiscreet cpf = CPFTools::DiscretizeCPF(cur1, cur2, pc.points[i], pt); // get pc.points[id] into the current points coordinate frame.
				//CPFDiscreet cpf =  CPFTools::DiscretizeCPF(cur1, cur2, pc.points[i], pc.points[id]);
				cpf.point_idx = i;


				// get the angle
				// The point p0 is in the frame orign. n is aligned with the x-axis. 
				float alpha_m = atan2(-pt(2), pt(1));
				cpf.alpha = alpha_m;

				d.push_back(cpf);
			}

		}
	}
}





/*
Match a set of reference descriptors with a swet of scene descriptors.
The result is a list of associations between scenen and reference points with similar descriptors.
@param model_data - a data set containing curvatures and decriptors of model data of type CPFModelData
@param scene_data - a data set containing curvatures and decriptors of model data of type CPFSceneData
@param results - matching point pairs.
@param params - parameter for the process.
*/
//static 
void  CPFMatching::MatchDescriptors(CPFModelData& model_data, CPFSceneData& scene_data, CPFMatchingData& results, CPFMatchingParams params)
{
	if (params.verbose ) {
		std::cout << "[INFO] - CPFMatchingExp: Start matching descriptors." << std::endl;
	}

	// descriptor vector size. 
	int scr_s_size = scene_data.getDescriptor().size();
	int src_m_size = model_data.getDescriptor().size();

	// point cloud size. 
	int model_point_size = model_data.size();
	int scene_point_size = scene_data.size();

	CPFDiscreetVec& desc_model = model_data.getDescriptor();
	CPFDiscreetVec& desc_scene = scene_data.getDescriptor();

	results.voting_clear();

	int angle_bins = params.cpf_angle_bins;


	for (int i = 0; i < model_point_size; i++) {

		int point_id = i;

		std::vector<int> accumulator(scene_point_size * angle_bins, 0);

		int count = 0;

		// -------------------------------------------------------------------
		// For each point i and its descriptors, find matching descriptors.
		for (int k = 0; k < src_m_size; k++) {

			CPFDiscreet src = desc_model[k];

			if (src.point_idx != point_id) continue; // must be a descriptor for the current point i

			// search for the destination descriptor
			for (int j = 0; j < scr_s_size; j++) {

				CPFDiscreet dst = desc_scene[j];

				// compare the descriptor
				if (src.data[0] == dst.data[0] && src.data[1] == dst.data[1] && src.data[2] == dst.data[2] && dst.data[0] != 0) {

					// Voting, fill the accumulator
					float alpha = src.alpha - dst.alpha;

					int alpha_bin = static_cast<int>(static_cast<float>(angle_bins) * ((alpha + 2.0f * static_cast<float>(M_PI)) / (4.0f * static_cast<float>(M_PI))));
					alpha_bin = std::max(0, std::min(alpha_bin, angle_bins));


					accumulator[dst.point_idx * angle_bins + alpha_bin]++;

					// store the output vote pair
					results.getVotePairVec().push_back(make_pair(i, alpha));


				}
			}

		}

		// -------------------------------------------------------------------
		// Find the voting winner

		int max_vote = 0;
		vector<int> max_votes_idx;
		vector<int> max_votes_value;

		for (int k = 0; k < accumulator.size(); k++) {
			if (accumulator[k] >= max_vote && accumulator[k] != 0) {
				max_vote = accumulator[k];
				max_votes_idx.push_back(k);
				max_votes_value.push_back(accumulator[k]);
			}
			accumulator[k] = 0; // Set it to zero for next iteration
		}

		// -----------------------------------------------------------------------
		// Recover the pose

		PointCloud& pc_model  = model_data.getPointCloud();
		PointCloud& pc_scene = scene_data.getPointCloud();


		for (int k = 0; k < max_votes_idx.size(); k++) {
			if (max_vote == max_votes_value[k]) {

				int max_scene_id = max_votes_idx[k] / angle_bins; // model id
				int max_alpha = max_votes_idx[k] % angle_bins; // restores the angle


				Eigen::Vector3f model_point(pc_model.points[point_id][0], pc_model.points[point_id][1], pc_model.points[point_id][2]);
				Eigen::Vector3f model_normal(pc_model.normals[point_id].x(), pc_model.normals[point_id].y(), pc_model.normals[point_id].z());

				Eigen::Vector3f scene_point(pc_scene.points[max_scene_id][0], pc_scene.points[max_scene_id][1], pc_scene.points[max_scene_id][2]);
				Eigen::Vector3f scene_normal(pc_scene.normals[max_scene_id].x(), pc_scene.normals[max_scene_id].y(), pc_scene.normals[max_scene_id].z());

				Eigen::Affine3f T = CPFTools::GetRefFrame(model_point, model_normal);
				Eigen::Affine3f Tmg = CPFTools::GetRefFrame(scene_point, scene_normal);

				float angle = (static_cast<float>(max_alpha) / static_cast<float>(angle_bins)) * 4.0f * static_cast<float>(M_PI) - 2.0f * static_cast<float>(M_PI);

				Eigen::AngleAxisf rot(angle, Eigen::Vector3f::UnitX());

				// Compose the transformations for the final pose
				Eigen::Affine3f final_transformation(Tmg.inverse() * rot * T);


	

				results.getPoseCandidatesPose().push_back(final_transformation);
				results.getPoseCandidatesVotes().push_back(max_votes_value[k]);

				//std::cout << "\tangle: " << angle << std::endl;
			}
		}

	}

	if (params.verbose ) {
		std::cout << "[INFO] - CPFMatchingExp: Found " << results.getPoseCandidatesVotes().size() << " pose candidates." << std::endl;
	}


}



/*
Naive implementation of the descriptor calculator.
@param data - CPF model data
@param params - a parameter set.
*/
//static 
void CPFMatching::CalculateDescriptorsNaive(CPFModelData& data, CPFMatchingParams params)
{
	CPFTools::CPFParam param;
	param.angle_bins = params.cpf_angle_bins;

	CPFTools::SetParam(param);

	//----------------------------------------------------------------------------------------------------------
	// nearest neighbors

	std::vector<Matches> matches;
	size_t s = data.getPointCloud().size();


	if (knn == NULL) knn = new KNN();
	knn->populate(data.getPointCloud());

	matches.clear();

	knn->radius(data.getPointCloud(), params.knn_serach_radius, matches);


	//----------------------------------------------------------------------------------------------------------
	// Calculate point curvatures

	data.clear();

	PointCloud& pc = data.getPointCloud();
	CPFCurvatureVec& c = data.getCurvature();
	CPFDiscreetVec& d = data.getDescriptor();


	for (int i = 0; i < s; i++) {

		uint32_t curv = CPFTools::DiscretizeCurvatureNaive(pc.points[i], pc.normals[i], pc, matches[i], params.cpf_multiplier);
		c.push_back(curv);
	}


	//----------------------------------------------------------------------------------------------------------
	// Calculate the descriptor
	d.clear();
	d.reserve(s);
	for (int i = 0; i < s; i++) {
		uint32_t cur1 = c[i];

		// find the reference frame for this point
		Eigen::Affine3f T = CPFTools::GetRefFrame(pc.points[i], pc.normals[i]);

		// current knn matches length as defined in Cuda_Types.h
		const int num_matches = KNN_MATCHES_LENGTH;

		for (int j = 0; j < num_matches; j++) {
			if (matches[i].matches[j].distance > 0.0) {

				int id = matches[i].matches[j].second;

				// ToDo: Check this. Id cannot be larger than c.size(). Perhaps old data. 
				if (id >= c.size()) continue;

				uint32_t cur2 = c[id];

				// Move the point p1 into the coordinate frame of the point p0
				Eigen::Vector3f pt = T * pc.points[id];

				CPFDiscreet cpf = CPFTools::DiscretizeCPF(cur1, cur2, pc.points[i], pt); // get pc.points[id] into the current points coordinate frame.
				//CPFDiscreet cpf =  CPFTools::DiscretizeCPF(cur1, cur2, pc.points[i], pc.points[id]);
				cpf.point_idx = i;


				// get the angle
				// The point p0 is in the frame orign. n is aligned with the x-axis. 
				float alpha_m = atan2(-pt(2), pt(1));
				cpf.alpha = alpha_m;

				d.push_back(cpf);
			}

		}
	}

}


/*
Match a set of reference descriptors with a swet of scene descriptors.
The result is a list of associations between scenen and reference points with similar descriptors.
@param model_data - a data set containing curvatures and decriptors of model data of type CPFModelData
@param scene_data - a data set containing curvatures and decriptors of model data of type CPFSceneData
@param results - matching point pairs.
@param params - parameter for the process.
*/
//static 
void CPFMatching::MatchDescriptorsNaive(CPFModelData& model_data, CPFSceneData& scene_data, CPFMatchingData& results, CPFMatchingParams params)
{
	if (params.verbose) {
		std::cout << "[INFO] - CPFMatchingExp: Start matching descriptors." << std::endl;
	}

	// descriptor vector size. 
	int scr_s_size = scene_data.getDescriptor().size();
	int src_m_size = model_data.getDescriptor().size();

	// point cloud size. 
	int model_point_size = model_data.size();
	int scene_point_size = scene_data.size();

	CPFDiscreetVec& desc_model = model_data.getDescriptor();
	CPFDiscreetVec& desc_scene = scene_data.getDescriptor();

	results.voting_clear();

	int angle_bins = params.cpf_angle_bins;


	for (int i = 0; i < model_point_size; i++) {

		int point_id = i;

		std::vector<int> accumulator(scene_point_size * angle_bins, 0);

		int count = 0;

		// -------------------------------------------------------------------
		// For each point i and its descriptors, find matching descriptors.
		for (int k = 0; k < src_m_size; k++) {

			CPFDiscreet src = desc_model[k];

			if (src.point_idx != point_id) continue; // must be a descriptor for the current point i

			// search for the destination descriptor
			for (int j = 0; j < scr_s_size; j++) {

				CPFDiscreet dst = desc_scene[j];

				// compare the descriptor
				if (src.data[0] == dst.data[0] && src.data[1] == dst.data[1] && src.data[2] == dst.data[2] && dst.data[0] != 0) {

					// Voting, fill the accumulator
					float alpha = src.alpha - dst.alpha;

					int alpha_bin = static_cast<int>(static_cast<float>(angle_bins) * ((alpha + 2.0f * static_cast<float>(M_PI)) / (4.0f * static_cast<float>(M_PI))));
					alpha_bin = std::max(0, std::min(alpha_bin, angle_bins));


					accumulator[dst.point_idx * angle_bins + alpha_bin]++;

					// store the output vote pair
					results.getVotePairVec().push_back(make_pair(i, alpha));


				}
			}

		}

		// -------------------------------------------------------------------
		// Find the voting winner

		int max_vote = 0;
		vector<int> max_votes_idx;
		vector<int> max_votes_value;

		for (int k = 0; k < accumulator.size(); k++) {
			if (accumulator[k] >= max_vote && accumulator[k] != 0) {
				max_vote = accumulator[k];
				max_votes_idx.push_back(k);
				max_votes_value.push_back(accumulator[k]);
			}
			accumulator[k] = 0; // Set it to zero for next iteration
		}

		// -----------------------------------------------------------------------
		// Recover the pose

		PointCloud& pc_model = model_data.getPointCloud();
		PointCloud& pc_scene = scene_data.getPointCloud();


		for (int k = 0; k < max_votes_idx.size(); k++) {
			if (max_vote == max_votes_value[k]) {

				int max_scene_id = max_votes_idx[k] / angle_bins; // model id
				int max_alpha = max_votes_idx[k] % angle_bins; // restores the angle


				Eigen::Vector3f model_point(pc_model.points[point_id][0], pc_model.points[point_id][1], pc_model.points[point_id][2]);
				Eigen::Vector3f model_normal(pc_model.normals[point_id].x(), pc_model.normals[point_id].y(), pc_model.normals[point_id].z());

				Eigen::Vector3f scene_point(pc_scene.points[max_scene_id][0], pc_scene.points[max_scene_id][1], pc_scene.points[max_scene_id][2]);
				Eigen::Vector3f scene_normal(pc_scene.normals[max_scene_id].x(), pc_scene.normals[max_scene_id].y(), pc_scene.normals[max_scene_id].z());

				Eigen::Affine3f T = CPFTools::GetRefFrame(model_point, model_normal);
				Eigen::Affine3f Tmg = CPFTools::GetRefFrame(scene_point, scene_normal);

				float angle = (static_cast<float>(max_alpha) / static_cast<float>(angle_bins)) * 4.0f * static_cast<float>(M_PI) - 2.0f * static_cast<float>(M_PI);

				Eigen::AngleAxisf rot(angle, Eigen::Vector3f::UnitX());

				// Compose the transformations for the final pose
				Eigen::Affine3f final_transformation(Tmg.inverse() * rot * T);




				results.getPoseCandidatesPose().push_back(final_transformation);
				results.getPoseCandidatesVotes().push_back(max_votes_value[k]);

				//std::cout << "\tangle: " << angle << std::endl;
			}
		}

	}

	if (params.verbose) {
		std::cout << "[INFO] - CPFMatchingExp: Found " << results.getPoseCandidatesVotes().size() << " pose candidates." << std::endl;
	}

}