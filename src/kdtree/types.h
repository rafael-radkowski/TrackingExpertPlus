#pragma once
#include <iostream>
#include <vector>


using namespace std;

// This variable limits the number of 
// matches the code can find. Cuda memory is fixed size. 
// Thus, one need to pre-define how many points one expects. 
#define KNN_MATCHES_LENGTH 21


class MyPoint
{
public:


	float	_data[3];


	int		_id;


	__host__ __device__ MyPoint()
		//_data[0](0.0), data[1](0.0), data[2](0.0)
	{
		_data[0] = 0.0;
		_data[1] = 0.0;
		_data[2] = 0.0;
		_id = -1;
	}


	__host__ __device__ MyPoint(float x, float y, float z)
	{
		_data[0] = x;
		_data[1] = y;
		_data[2] = z;
	}



	__host__ __device__ float& operator[]( int index)
	{
		return _data[index];
	}

	__host__ __device__ const float operator[](int index)const
	{
		return _data[index];
	}
};


typedef MyPoint Cuda_Point;


// type for the matches
// first - the distance, second - the point data.
//typedef std::pair<double, dPoint> dMatch;
typedef struct _MyMatch {
	int first; // index of query point
	int second; // index of search point. 
	//MyPoint second; // Matching point
	double distance;

	__host__ __device__ _MyMatch()
	{
		first = -1;
		second = -1;
		distance = 10000.0;
		//second._id = -1;
	}
} MyMatch;



typedef struct MyMatches
{
	MyMatch		matches[KNN_MATCHES_LENGTH];

} MyMatches;