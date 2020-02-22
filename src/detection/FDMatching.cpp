#include "FDMatching.h"


using namespace texpert;

#define M_PI 3.14159265359

FDMatching::FDMatching()
{
	_kdtree = NULL;
	_points_test = NULL;
	_normals_test = NULL;
	_N = 0;
	_k = KNN_MATCHES_LENGTH-1;
	
	_verbose = false;

	_distance_step = 0.01;
	_angle_step = 12.0 / 180.0f * static_cast<float>(M_PI);
	_angle_bins = (int)(static_cast<float>(2 * M_PI) / _angle_step) + 1;
}


FDMatching::~FDMatching()
{


}



/*
Set the distance step
@param distance_step -  a value larger than 0.0;
*/
bool FDMatching::setDistanceStep(float distance_step)
{
	if (distance_step < 0.0) return false;

	_distance_step = distance_step;

	return true;
}


/*
Set the angle step
@param angle_step -  a value larger than 0.0;
*/
bool FDMatching::setAngleStep(float angele_step)
{
	if (angele_step < 3.0 && angele_step > 90.0){
		std::cout << "[Warning] - angele_step not set; parameter must be in range [3.0, 90.0]" << endl;
		return false;
	}

	_angle_step = angele_step / 180.0f * static_cast<float>(M_PI);;
	_angle_bins = (int)(static_cast<float>(2 * M_PI) / _angle_step) + 1;

	return true;
}




/*
Extract the ppf features from points and normal vectors
@param points - point as {x, y, z}
@param normals - normal vectors, points and normal vectors are index aligned.
@param map_results - the map in which all the results should be inserted to.
*/
bool FDMatching::extract_feature_map(vector<Eigen::Vector3f>* points, vector<Eigen::Vector3f>* normals)
{

	//_distance_step = g_ppf_dist_step;
	//_angle_step = g_ppf_ang_step / 180.0f * static_cast<float>(M_PI);

	_points_test = points;
	_normals_test = normals;
	_N = _points_test->size();

	_map_test_points.clear();


	//int k = g_dlg_ppf_nn;

	//------------------------------------------------------------------------------------------------------------
	
	// Pair all the points in the cloud
	vector<Eigen::Vector3f>::iterator pItr = _points_test->begin();
	vector<Eigen::Vector3f>::iterator nItr = _normals_test->begin();

	int counter = 0;
	int counter2 = 0;

	if(_verbose)
		_cprintf("\n[PPFExtTracking] - Start extracting descriptors.");

	int point_index = 0;
	while (pItr != points->end())
	{
		Eigen::Vector3f p0((*pItr)[0], (*pItr)[1], (*pItr)[2]);
		Eigen::Vector3f n0((*nItr).x(), (*nItr).y(), (*nItr).z());

		// find the reference frame for this point
		Eigen::Affine3f T = FDTools::getRefFrame(p0, n0);

		// link the first point with all other points in the scene
		vector<Eigen::Vector3f>::iterator pItr2 = points->begin();
		vector<Eigen::Vector3f>::iterator nItr2 = normals->begin();

		counter2 = 0;
		while (pItr2 != points->end())
		{
			if (pItr == pItr2) {
				counter2++;
				pItr2++;
				nItr2++;
				continue; // same point;
			}

			Eigen::Vector3f p1((*pItr2)[0], (*pItr2)[1], (*pItr2)[2]);
			Eigen::Vector3f n1((*nItr2).x(), (*nItr2).y(), (*nItr2).z());

			PPFDiscreet ppf = FDTools::DiscretizePPF(p0, n0, p1, n1, _distance_step, _angle_step);
			ppf.point_index = point_index;

			// Move the point p1 into the coordinate frame of the point p0
			Eigen::Vector3f pt = T * p1;

			// get the angle
			// The point p0 is in the frame orign. n is aligned with the x-axis. 
			float alpha_m = atan2(-pt(2), pt(1));

			VotePair vp(point_index, alpha_m);

			// Create the tuple for the map
			_map_test_points.insert(make_pair(ppf, vp));

			counter2++;
			pItr2++;
			nItr2++;
		}

		counter++;
		pItr++;
		nItr++;
		point_index++;
	}

	if(_verbose)
		_cprintf("\n[PPFExtTracking] - Descriptors for %d points extracted. \n", counter);
	return true;

}



bool  FDMatching::searchIn(vector<Eigen::Vector3f>* points, vector<Eigen::Vector3f>* normals, std::vector<Pose>& poses)
{
	return detect( points, normals,  poses);
}


/*
Extract the ppf features from points and normal vectors
@param points - point as {x, y, z}
@param normals - normal vectors, points and normal vectors are index aligned.
@param map_results - the map in which all the results should be inserted to.
*/
bool FDMatching::detect(vector<Eigen::Vector3f>* points, vector<Eigen::Vector3f>* normals,  std::vector<Pose>& poses)
{

	//_distance_step = g_ppf_dist_step;
	//_angle_step = g_ppf_ang_step / 180.0f * static_cast<float>(M_PI);
	int k = _k;

	poses.clear();

	std::vector<int> accumulator(_N * _angle_bins, 0);

	//------------------------------------------------------------------------------------------------------------
	// Principle curvatures



	//------------------------------------------------------------------------------------------------------------
	// Search knn
	if (_kdtree == NULL) {
		_kdtree = new Cuda_KdTree();
	}
	else {
		_kdtree->resetDevTree();
	}


	vector< vector<int> > index_vector;
	vector< vector<float> > distance_vector;
	vector<Cuda_Point> data;

	for (int i = 0; i < points->size(); i++)
	{
		Eigen::Vector3f p = (*points)[i];
		data.push_back(Cuda_Point(p.x(), p.y(), p.z() ));
		data[i]._id = i;
	}

	// search for nearest neighbors
	// The function searches for k+1 because the first result is the  querry point itself. 
	_kdtree->initialize(data);
	vector<MyMatches> matches;
	_kdtree->knn(data, matches, k+1);


	//------------------------------------------------------------------------------------------------------------
	// Extract features and match 

	// Pair all the points in the cloud
	vector<Eigen::Vector3f>::iterator pItr = points->begin();
	vector<Eigen::Vector3f>::iterator nItr = normals->begin();

	int counter = 0;
	int counter2 = 0;

	if(_verbose)
		_cprintf("\n[PPFExtTracking] - Start extracting descriptors.");


	vector<Pose> poses_candidates;

	int overall_max_votes = 0;
	int point_index = 0;
	while (pItr != points->end())
	{
		Eigen::Vector3f p0((*pItr)[0], (*pItr)[1], (*pItr)[2]);
		Eigen::Vector3f n0((*nItr).x(), (*nItr).y(), (*nItr).z());

		// find the reference frame for this point

		Affine3f T = FDTools::getRefFrame(p0, n0);

		// link the first point with all other points in the scene
	//	vector<dPoint>::iterator pItr2 = points->begin();
	//	vector<Vec3>::iterator nItr2 = normals->begin();


		MyMatches m = matches[point_index];
		

		// loop through all nearest neighbors
		for (int j = 0; j < KNN_MATCHES_LENGTH; j++)
		{
			MyMatch n = m.matches[j];
			
			if (n.first == n.second) continue;
			if (n.distance == 0) continue;
			
			int pidx = n.second;
			Eigen::Vector3f p1((*points)[pidx].x(), (*points)[pidx].y(), (*points)[pidx].z());
			Eigen::Vector3f n1((*normals)[pidx].x(), (*normals)[pidx].y(), (*normals)[pidx].z());

			
			PPFDiscreet ppf = FDTools::DiscretizePPF(p0, n0, p1, n1, _distance_step, _angle_step);
			ppf.point_index = point_index;

			// Compute the alpha_s angle

			// Rotate the point
			Eigen::Vector3f pt = T * p1;

			// get the angle
			float alpha_s = atan2(-pt(2), pt(1));

			// get similar fieatures
			auto similar_features = _map_test_points.equal_range(ppf);

			// Accumulate the votes of similar features
			std::for_each(
				similar_features.first,
				similar_features.second,
				[&](const std::pair<PPFDiscreet, VotePair>& match)
				{
					int model_i = match.second.model_i;
					float alpha_m = match.second.alpha_m;

					float alpha = alpha_m - alpha_s;
					int alpha_bin = static_cast<int>(static_cast<float>(_angle_bins) * ((alpha + 2.0f * static_cast<float>(M_PI)) / (4.0f * static_cast<float>(M_PI))));

					// Count votes
					accumulator[model_i * _angle_bins + alpha_bin]++;
				}
			);


		}

		//------------------------------------------------------------------------------------------------------------
		// Look for the best vote


		int max_votes = 0;
		int max_votes_idx = 0;

		for (int k = 0; k < accumulator.size(); k++) {
			if (accumulator[k] > max_votes) {
				max_votes = accumulator[k];
				max_votes_idx = k;
			}
			accumulator[k] = 0; // Set it to zero for next iteration
		}

		int max_model_i = max_votes_idx / _angle_bins;
		int max_alpha = max_votes_idx % _angle_bins;
		
		Eigen::Vector3f model_point((*_points_test)[max_model_i][0], (*_points_test)[max_model_i][1],(* _points_test)[max_model_i][2]);
		Eigen::Vector3f model_normal((*_normals_test)[max_model_i].x(), (*_normals_test)[max_model_i].y(), (*_normals_test)[max_model_i].z());

		Affine3f Tmg = FDTools::getRefFrame(model_point, model_normal);

		float angle = (static_cast<float>(max_alpha) / static_cast<float>(_angle_bins)) * 4.0f * static_cast<float>(M_PI) - 2.0f * static_cast<float>(M_PI);

		Eigen::AngleAxisf rot(angle, Eigen::Vector3f::UnitX());

		// Compose the transformations for the final pose
		Eigen::Affine3f final_transformation(T.inverse() * rot * Tmg);



		Pose pose;
		pose.t = final_transformation;
		pose.votes = max_votes;
		pose.to_scene_idx = point_index;
		pose.from_model_idx = max_model_i;

		
		//pose.t = pose.t.inverse();

		poses_candidates.push_back(pose);


	

		counter++;
		pItr++;
		nItr++;
		point_index++;
	}


	// sort all poses
	std::sort(poses_candidates.begin(), poses_candidates.end(),
			  [](const Pose& a, const Pose& b) {return a.votes > b.votes; });

	vector<int> votes;

	_cluster.clusterPoses(poses_candidates, poses, votes);


	if (_verbose) {
		_cprintf("\n[PPFExtTracking] - Descriptors for %d environment points extracted. ", counter);
		_cprintf("\n[PPFExtTracking] - Found  %d point to point relations.\n ", poses.size());
	}

	return true;
}



/*
Enable console outputs. 
*/
void FDMatching::setVerbose(bool verbose)
{
	_verbose = verbose;
}




/*
Set true to invert the pose. The standard pose transforms the 
reference object coord. to the test object coord, e.g. for a regular camera setting. The inverted pose
translates the test object to the reference object. 
@param invert - set true to invert the pose. Default is false. 
*/
bool FDMatching::invertPose(bool value)
{
	_cluster.invertPose(value);

	return true;
}


/*
Set the cluster threshold for pose clusterin algorithm. 
Pose clustering considers all poses as equal (similar) if their 
center-distance and their angle delta are under a threshold. 
@param distance_th - the distance threshold. 
@param angle_th - the angle threshold in degress. 
*/
void FDMatching::setClusteringThreshold(float distance_th, float angle_th)
{
	_cluster.setRotationThreshold(angle_th);
	_cluster.setTranslationThreshold(distance_th);
}
