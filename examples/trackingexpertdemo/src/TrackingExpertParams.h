#pragma once

#include <iostream>
#include <algorithm>

namespace texpert{


typedef struct TEParams {

	// app parameters
	bool		verbose; // enables additional debug console output.

	// registration parameters
	float		angle_step;  // descriptor histogram threshold [1, 180] degree
	float		cluster_trans_threshold; // clustering threshold [0, inf ]
	float		cluster_rot_threshold; //

	// curvature calculation parameters
	float		curvature_search_radius;  // radius in which normals are incorporated into the curvature [0, inf]

	// point cloud sampling threshold
	float		sampling_grid_size; // edge size of a single voxel cell for point sampling [0.005, inf]
	int			camera_sampling_offset; // Sampling pattern for the camera image. The distance is the number of pixels to ignore. 
										// Note that this parameter gets calculated. 

	// icp parameters
	float		icp_outlier_reject_angle; // normal vector rejection angle [0, 180] degree
	float		icp_outlier_reject_distance; // point outlier rejection distance [0.01f, 100]
	float		icp_termination_dist; // ICP RMS termination criteria [0, inf]
	float		icp_num_max_iterations; // ICP max. number of allowed iterations [1, 1000]


	TEParams() {
		verbose = false;
		angle_step = 12.0;
		cluster_trans_threshold = 0.03;
		cluster_rot_threshold = 0.8;
		curvature_search_radius = 0.1;
		sampling_grid_size = 0.015;
		camera_sampling_offset = 8;
		icp_outlier_reject_angle = 45.0;
		icp_outlier_reject_distance = 0.1;
		icp_termination_dist = 0.00000001;
		icp_num_max_iterations = 200;
	}

	bool valid(void) {

		bool error = false;

		if(angle_step > 180.0 || angle_step < 1.0){
			std::cout << "[INFO] - Updated angle step" << std::endl;
			error = true;
		}
		angle_step = std::min(180.0f, std::max(1.0f, angle_step));

		
		if( cluster_trans_threshold < 0.0){
			std::cout << "[INFO] - Updated cluster translation threshold" << std::endl;
			error = true;
		}
		cluster_trans_threshold = std::max(0.0f, cluster_trans_threshold);

		if(cluster_rot_threshold > 180.0 || cluster_rot_threshold < 1.0){
			std::cout << "[INFO] - Updated cluster rotation threshold" << std::endl;
			error = true;
		}
		cluster_rot_threshold = std::min(180.0f, std::max(1.0f, cluster_rot_threshold));

		if( curvature_search_radius < 0.0){
			std::cout << "[INFO] - Updated curvature search radius" << std::endl;
			error = true;
		}
		curvature_search_radius = std::max(0.0f, curvature_search_radius);

		
		if( sampling_grid_size < 0.005){
			std::cout << "[INFO] - Updated voxel sampling grid size" << std::endl;
			error = true;
		}
		sampling_grid_size =  std::max(0.005f, sampling_grid_size);

		if( camera_sampling_offset < 1 || camera_sampling_offset > 100){
			std::cout << "[INFO] - Updated camera sampling grid " << std::endl;
			error = true;
		}
		camera_sampling_offset = std::min( 1, std::max(100, camera_sampling_offset));



		if( icp_outlier_reject_angle < 1.0f || icp_outlier_reject_angle > 180.0f){
			std::cout << "[INFO] - Updated icp_outlier_reject_angle." << std::endl;
			error = true;
		}
		icp_outlier_reject_angle = std::min( 1.0f, std::max(180.0f, icp_outlier_reject_angle));


		if( icp_outlier_reject_distance < 0.001f || icp_outlier_reject_distance > 100.0f){
			std::cout << "[INFO] - Updated icp_outlier_reject_distance." << std::endl;
			error = true;
		}
		icp_outlier_reject_distance = std::min( 0.001f, std::max(100.0f, icp_outlier_reject_distance));


		return error;
	}


} TEParams;

}