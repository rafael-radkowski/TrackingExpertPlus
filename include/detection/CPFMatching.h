/*
@class CPFMatching

The class calculates and matches feature descriptors. It also calculates curvatures. 
Its main purpose is to match a reference point cloud with its counterpart in a scene point cloud. 

Responsibilities:
- Calculate curvatures
- Calculate descriptors. 
- Match descriptors.

and return descriptors pairs indicating relation between the object and the camera point cloud. 

Rafael Radkowski
Aug 2021

MIT License
--------------------------------------------------------------------------------
Last edited:


*/

//stl 
#include <iostream>
#include <string>
#include <algorithm>
#include <vector>
#include <iostream>
#include <unordered_map>
#include <algorithm>

// Eigen
#include <Eigen/Dense>
#include <Eigen/Geometry>

// local
#include "CPFRenderHelpers.h"
#include "Types.h"
#include "CPFTypes.h"
#include "CPFTools.h"
#include "KNN.h"


#include "CPFData.h"

/*
Search parameters.
*/
typedef struct CPFMatchingParams {
	float knn_serach_radius; // the radius curvature extractor search radius. 

	int cpf_angle_bins; // the number of histogram bins. 
	float cpf_multiplier; // value to magnify the curvautures, e.g., x 10 to increase the numerical value. 


	bool verbose;

	CPFMatchingParams()
	{
		knn_serach_radius = 0.1f;

		cpf_angle_bins = 12;
		cpf_multiplier = 10.0f;
		verbose = false;
	}
}CPFMatchingParams;



class CPFMatching
{
public:
	
	

	/*
	The class calculates descriptors for the reference model(s). 
	It uses the point cloud as stored in data, and populates the variables 'descriptors' and 'curvatures'
	@param data - CPF model data
	@param params - a parameter set.
	*/
	static void CalculateDescriptors(CPFModelData& data, CPFMatchingParams params );


	/*
	Naive implementation of the descriptor calculator. 
	@param data - CPF model data
	@param params - a parameter set.
	*/
	static void CalculateDescriptorsNaive(CPFModelData& data, CPFMatchingParams params);


	/*
	Match a set of reference descriptors with a swet of scene descriptors. 
	The result is a list of associations between scenen and reference points with similar descriptors. 
	@param model_data - a data set containing curvatures and decriptors of model data of type CPFModelData
	@param scene_data - a data set containing curvatures and decriptors of model data of type CPFSceneData
	@param results - matching point pairs. 
	@param params - parameter for the process. 
	*/
	static void  MatchDescriptors(CPFModelData& model_data, CPFSceneData& scene_data,  CPFMatchingData& results, CPFMatchingParams params);



	/*
	Match a set of reference descriptors with a swet of scene descriptors.
	The result is a list of associations between scenen and reference points with similar descriptors.
	@param model_data - a data set containing curvatures and decriptors of model data of type CPFModelData
	@param scene_data - a data set containing curvatures and decriptors of model data of type CPFSceneData
	@param results - matching point pairs.
	@param params - parameter for the process.
	*/
	static void  MatchDescriptorsNaive(CPFModelData& model_data, CPFSceneData& scene_data, CPFMatchingData& results, CPFMatchingParams params);

private:

	
	

};

