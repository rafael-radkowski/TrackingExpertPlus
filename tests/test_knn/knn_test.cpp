

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



KNN* knn;

PointCloud	cameraPoints0;
PointCloud	cameraPoints1;


std::vector<MyMatches> matches0;
std::vector<MyMatches> matches1;


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


float Distance(Eigen::Vector3f p0, Eigen::Vector3f p1) {

	return std::sqrt( std::pow( p0.x() - p1.x(),2)  + std::pow( p0.y() - p1.y(),2) + std::pow( p0.z() - p1.z(),2)); 

}


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
			std::cout << "Found error for i = " << i << " with gpu " << matches0[i].matches[0].second << " and naive " << matches1[i].matches[0].second  << " with distance " << matches0[i].matches[0].distance << " and " << matches1[i].matches[0].distance * matches1[i].matches[0].distance << std::endl;
			error_count++;
		}
	}

	std::cout << "Found " << error_count << " in total (" <<  float(error_count)/float(s0) * 100.0 << "%)" << std::endl;



	return error_count;
}


int RunTest(int num_points, float min_range, float max_range) {

	
	// generate
	GenerateRandomPointCloud( cameraPoints0, num_points, min_range, max_range);
	GenerateRandomPointCloud( cameraPoints1, num_points, min_range, max_range);

	matches0.clear();
	matches1.clear();

	knn->reset();
	knn->populate(cameraPoints0);

	knn->knn(cameraPoints1,1,matches0);

	// naive
	FindKNN_Naive(cameraPoints1, cameraPoints0, 1, matches1);


	// compare
	return CompareMatches( matches0,  matches1);
}



int main(int argc, char** argv)
{

	knn = new KNN();


	//-------------------------------------------------
	// First test round

	std::cout << "1. Testing with the same points." << endl;

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

	std::cout << "\n2. Testing with different points." << endl;

	// generate
	GenerateRandomPointCloud( cameraPoints1, 100);

	knn->knn(cameraPoints1,1,matches0);

	// naive
	FindKNN_Naive(cameraPoints1, cameraPoints0, 1, matches1);

	// compare
	CompareMatches( matches0,  matches1);


	//-------------------------------------------------
	// Third test round

	std::cout << "\n3. Testing with different points and increased point size." << endl;

	for(int i = 0; i< 15; i++){
		int k = 10000;

		RunTest( k, -1.0, 1.0);
	}

	delete knn;

}




