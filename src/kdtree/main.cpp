#include <iostream>
#include <stdio.h>
#include <vector>
#include <ctime>
#include <fstream>
#include <chrono>
#include <algorithm>
#include <limits>
#include <random>
#include <vector>

#include "Cuda_KdTree.h"


extern int  SORT_TPB;


using namespace std;
using namespace std::chrono;


vector<high_resolution_clock::time_point> g_times_t0(10); // the start values
vector<high_resolution_clock::time_point> g_times_t1(10); // the start values



std::vector<float> GenerateDataFloat(size_t size, float min, float max)
{
	using value_type = float;

	// We use static in order to instantiate the random engine
	// and the distribution once only.
	// It may provoke some thread-safety issues.
	static std::uniform_real_distribution<value_type> distribution(min, max);
	static std::default_random_engine generator;

	std::vector<value_type> data(size);
	std::generate(data.begin(), data.end(), []() { return distribution(generator); });

	return data;

}


std::vector<Cuda_Point> generateRandomPoints(int amount) {
	auto xs = GenerateDataFloat(amount, 0, 1);
	auto ys = GenerateDataFloat(amount, 0, 1);
	auto zs = GenerateDataFloat(amount, 0, 1);

	std::vector<Cuda_Point> points;
	for (int i = 0; i < amount; ++i) {
		Cuda_Point p = Cuda_Point(xs[i], ys[i], zs[i]);
		p._id = i;
		points.push_back(p);
	}
	return points;
}


//const 
std::string GetCurrentDateTime2()
{
	time_t     now = time(0);
	struct tm  tstruct;
	char       buf[80];
	tstruct = *localtime(&now);

	// Visit http://en.cppreference.com/w/cpp/chrono/c/strftime
	// for more information about date/time format
	strftime(buf, sizeof(buf), "%Y-%m-%d_%I-%M-%S", &tstruct);

	return buf;
}



int main(int argc, char** argv)
{

	//Cuda_Helpers::CheckDeviceCapabilities();
	Cuda_Helpers::CudaSetDevice(0);



	vector<Cuda_Point> data;// = generateRandomPoints(N);

	
	data.push_back(Cuda_Point(4.0, 5.0, 6.0)); data[0]._id = 0;
	data.push_back(Cuda_Point(1.0, 2.0, 3.0)); data[1]._id = 1;
	data.push_back(Cuda_Point(7.0, 8.0, 9.0)); data[2]._id = 2;
	data.push_back(Cuda_Point(10.0, 11.0, 12.0)); data[3]._id = 3;
	data.push_back(Cuda_Point(13.0, 14.0, 15.0)); data[4]._id = 4;
	data.push_back(Cuda_Point(16.0, 17.0, 18.0)); data[5]._id = 5;
	data.push_back(Cuda_Point(19.0, 20.0, 21.0)); data[6]._id = 6;
	data.push_back(Cuda_Point(22.0, 23.0, 24.0)); data[7]._id = 7;
	data.push_back(Cuda_Point(25.0, 26.0, 27.0)); data[8]._id = 8;
	data.push_back(Cuda_Point(28.0, 29.0, 30.0)); data[9]._id = 9;
	data.push_back(Cuda_Point(31.0, 32.0, 33.0)); data[10]._id = 10;
	data.push_back(Cuda_Point(34.0, 35.0, 36.0)); data[11]._id = 11;
	data.push_back(Cuda_Point(37.0, 38.0, 39.0)); data[12]._id = 12;
	data.push_back(Cuda_Point(40.0, 41.0, 42.0)); data[13]._id = 13;
	data.push_back(Cuda_Point(43.0, 44.0, 45.0)); data[14]._id = 14;
	int N = data.size();


	Cuda_KdTree* tree = new Cuda_KdTree();


	tree->initialize(data);


	vector<Cuda_Point> querry;
	querry.push_back(Cuda_Point(1.1, 2.2, 3.1)); querry[0]._id = 0;
	querry.push_back(Cuda_Point(4.1, 5.1, 6.1)); querry[1]._id = 1;
	querry.push_back(Cuda_Point(34.20, 35.10, 36.20)); querry[2]._id = 2;

	vector<MyMatches> matches;

	
//#define RADIUS_SEARCH
#ifdef RADIUS_SEARCH
	tree->radius_search(querry, matches, 10.0);
#else
	int k = 3;
	tree->knn(querry, matches, k);
#endif

	for (int i = 0; i < matches.size(); i++)
	{
		MyMatches m = matches[i];
		printf("For point %d, found:\n", i);
		for (int j = 0; j < KNN_MATCHES_LENGTH; j++)
		{
			MyMatch n = m.matches[j];
			if (n.distance > 0){
				printf("\t%d -> %d - dist: %lf\n", n.first, n.second, std::sqrt(n.distance));
			}
		}

	}

	delete tree;
	return 1;
}