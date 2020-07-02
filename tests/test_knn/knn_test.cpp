/*
@file knn_test.cpp

This file tests the kd-tree and the k-nearest neighbors methods.
The kd-tree is a cuda implementation. The test compares the cuda implementation (O( n log(n) )
vs. a naive O(n^2) implementation. 

The test runs multiple times with different random datasets. 

Note that error < 4% can be expected. The cuda version stops to backtrack adjacent 
branches at one point to increase performance. Also, it uses a Radix sort with integers, 
and the conversion results in inaccuracies that yield some errors. 
The kd-tree was develop with point clouds in mind, so the camera tolerances introduce larger errors. 

Rafael Radkowski
Iowa State University
rafael@iastate.edu
January 2020
MIT License
-----------------------------------------------------------------------------------------------------------------------------
Last edited:



*/

// STL
#include <iostream>
#include <string>
#include <Windows.h>
#include <fstream>
#include <thread>
#include <mutex>
#include <cmath>

// GLM include files
#define GLM_FORCE_INLINE
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtx/transform.hpp>  // transformation
#include <glm/gtx/quaternion.hpp> // quaternions


// TrackingExpert
#include "trackingx.h"
#include "graphicsx.h"
#include "ReaderWriterOBJ.h"
#include "ReaderWriterPLY.h"
#include "Cuda_KdTree.h"  // the ICP class to test
#include "KNN.h"
#include "RandomGenerator.h"

using namespace texpert;


//-------------------------------------------------------------
// The knn tool
KNN* knn;

PointCloud	cameraPoints0;
PointCloud	cameraPoints1;


std::vector<MyMatches> matches0;
std::vector<MyMatches> matches1;


/*
Function to generate random point clouds and normal vectors. Note that the normal vectors are just 
points. 
@param pc - reference to the location for the point cloud.
@param num_points - number of points to generatel
@param min, max - the minimum and maximum range of the points. 
*/
void GenerateRandomPointCloud( PointCloud& pc, int num_points, float min = -2.0, float max = 2.0)
{
	pc.points.clear();
	pc.normals.clear();

	for (int i = 0; i < num_points; i++) {
		vector<float> p = RandomGenerator::FloatPosition(min, max);

		pc.points.push_back(Eigen::Vector3f(p[0], p[1], p[2]));
		pc.normals.push_back(Eigen::Vector3f(p[0], p[1], p[2]));
	}
	pc.size();
}

/*
Calculate the distance between two points. 
@param p0 - the first point as Eigen Vector3f (x, y, z)
@param p1 - the second point as Eigen Vector3f (x, y, z)
@return - the distance as float. 
*/
float Distance(Eigen::Vector3f p0, Eigen::Vector3f p1) {

	return std::sqrt( std::pow( p0.x() - p1.x(),2)  + std::pow( p0.y() - p1.y(),2) + std::pow( p0.z() - p1.z(),2)); 

}

/*
Find the nearest neighbors between a search point set and a second one. For each point in pc_search, the function output a 
nearest neighbor from pc_cam. 
@param pc_search - the search point cloud
@param pc_cam - the other point clud
@param k - currently not in use.
@param matches - a location to store the matches.
*/
bool FindKNN_Naive(PointCloud& pc_search, PointCloud& pc_cam, int k, std::vector<MyMatches>& matches)
{

	int s = pc_search.size();
	int p = pc_cam.size();

	matches.clear();
	matches.resize(s);


	for (int i = 0; i < s; i++) {
		float min_distance = 10000000.00;
		int min_idx = -1;
		for (int j = 0; j < p; j++) {
			float d = Distance(pc_search.points[i], pc_cam.points[j]);
			if ( d < min_distance) {
				min_distance = d;
				min_idx = j;
			}
		}
		matches[i].matches[0].first = i;
		matches[i].matches[0].second = min_idx;
		matches[i].matches[0].distance = min_distance;

	}
	return true;
}

/*
Compare two set of matches. For each search point, the naive method and the kd-tree should 
find the identical match. Thus, the function compares the point indices and reports an error, 
if the point-pairs do not match. 
@param matches0 - the location with the first set of matches
@param matches1 - the location with the second set of matches. 
*/
int CompareMatches(std::vector<MyMatches>& matches0, std::vector<MyMatches>& matches1)
{
	int s0 = matches0.size();
	int s1 = matches1.size();

	if (s0 != s1) {
		std::cout << "Error - matches have not the same size " << s0 << " to " << s1 << endl;
	}

	int error_count = 0;

	for (int i = 0; i < s0; i++) {
		if (matches0[i].matches[0].second != matches1[i].matches[0].second) {
			//std::cout << "Found error for i = " << i << " with gpu " << matches0[i].matches[0].second << " and naive " << matches1[i].matches[0].second  << " with distance " << matches0[i].matches[0].distance << " and " << matches1[i].matches[0].distance * matches1[i].matches[0].distance << std::endl;
			error_count++;
		}
	}

	float error_percentage =  float(error_count)/float(s0) * 100.0;

	std::cout << "[INFO] - Found " << error_count << " in total (" << error_percentage << "%)" << std::endl;

	// When working with the kd-tree, some minor errors can be expected. Those are the result of a integer conversion, the tree
	// works with a Radix search. Also, the tree does not backtrack into adjacent branches indefinitely. 
	// The error does not matter when working with point cloud data from cameras, since the camera tolerances yield larger variances. 
	// The error was never larger than 5%. If you encounter a larger error, this requires furter investigation but may not point to a bug, etc. 
	if (error_percentage > 5.0) {
		std::cout << "[ERROR] - The last run yielded an error > 5% with " << error_percentage << "%. That is higher than expected." << std::endl;
	}


	return error_count;
}

/*
Run the test.
The function runs the test.
*/
int RunTest(int num_points, float min_range, float max_range) {

	
	// generate two set of random point clouds. 
	GenerateRandomPointCloud( cameraPoints0, num_points, min_range, max_range);
	GenerateRandomPointCloud( cameraPoints1, num_points, min_range, max_range);

	matches0.clear();
	matches1.clear();

	// populate the kd-tree
	knn->reset();
	knn->populate(cameraPoints0);

	// search for the neaarest neighbors
	knn->knn(cameraPoints1,1,matches0);

	// run the naive knn method. 
	FindKNN_Naive(cameraPoints1, cameraPoints0, 1, matches1);

	// compare the results
	return CompareMatches( matches0,  matches1);
}



int main(int argc, char** argv)
{

	std::cout << "KNN Test.\n" << std::endl;
	std::cout << "This application implements a k-nearest neighbors test using a kd-tree." << std::endl;
	std::cout << "The kd-tree uses cuda to construct the tree and to find nearest neighbors. " << std::endl;
	std::cout << "The test compares the kd-tree solution with a naive solution. \nThe application runs 17 test in 3-stages, using different data set complexities.\n" << std::endl;
	
	std::cout << "Rafael Radkowski\nIowa State University\nrafael@iastate.edu" << std::endl;
	std::cout << "-----------------------------------------------------------------------------------------\n" << std::endl;


	// create the knn
	knn = new KNN();


	//-------------------------------------------------
	// First test round

	std::cout << "[Info] 1. Testing with the same points." << endl;

	// generate
	GenerateRandomPointCloud( cameraPoints0, 100);

	// KNN
	knn->populate(cameraPoints0);
	knn->knn(cameraPoints0,1,matches0);

	// naive
	FindKNN_Naive(cameraPoints0, cameraPoints0, 1, matches1);

	// compare
	CompareMatches( matches0,  matches1);

	//-------------------------------------------------
	// Second test round

	std::cout << "\n[Info] 2. Testing with different points." << endl;

	// generate
	GenerateRandomPointCloud( cameraPoints1, 100);

	knn->knn(cameraPoints1,1,matches0);

	// naive
	FindKNN_Naive(cameraPoints1, cameraPoints0, 1, matches1);

	// compare
	CompareMatches( matches0,  matches1);


	//-------------------------------------------------
	// Third test round

	std::cout << "\n[Info] 3. Testing with different points and increased point size." << endl;

	for(int i = 0; i< 15; i++){
		int k = 10000;

		RunTest( k, -1.0, 1.0);
	}

	delete knn;

}




