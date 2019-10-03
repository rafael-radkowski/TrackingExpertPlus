#pragma once
/*
class RandomGenerator

The class generate random numbers. 
This class is a part of the performance analysis example. It is used to 
generate random positions and orientations for test objects. 

Note that this is an tool which is not necessary to enable the main functionality.  

Features:
- Generate random numbers

Rafael Radkowski
Iowa State University
rafael@iastate.edu
+1 (515) 294 7044
MIT License
------------------------------------------------------
Last Changes:

*/

#include <algorithm>
#include <iostream>
#include <limits>
#include <random>
#include <vector>
#include <algorithm>
#include <random>


class RandomGenerator
{
public:

	/*
	Return a vector of random integer values.
	@param size - the number of integer values to return.
	@return - a vector with random integers of size 'size'
	*/
	static std::vector<int> GenerateDataInt(size_t size);

	/*
	Return a vector of random float values.
	@param size - the number of integer values to return.
	@param min - the smallest number to be returned; default is the numerical min float.
	@param max - the largest number to be returned; default is the numerical max float.
	@return - a vector with random floats of size 'size'
	*/
	static std::vector<float> GenerateDataFloat(size_t size, float min = std::numeric_limits<float>::min(), float max = std::numeric_limits<float>::max());

	/*
	Return a random position as vector. The vector contains three numners to be interpreted as ( x, y, z ).
	@param min - the smallest number to be returned; default is the numerical min float.
	@param max - the largest number to be returned; default is the numerical max float.
	@return - a vector with three float elements to be interpreted as ( x, y, z ).
	*/
	static std::vector<float> FloatPosition(float min = std::numeric_limits<float>::min(), float max = std::numeric_limits<float>::max());
};
