#include "RandomGenerator.h"



//static 
std::vector<int> RandomGenerator::GenerateDataInt(size_t size)
{
	using value_type = int;

	// We use static in order to instantiate the random engine
	// and the distribution once only.
	// It may provoke some thread-safety issues.
	static std::uniform_int_distribution<value_type> distribution(
		std::numeric_limits<value_type>::min(),
		std::numeric_limits<value_type>::max());
	static std::default_random_engine generator;

	std::vector<value_type> data(size);
	std::generate(data.begin(), data.end(), []() { return distribution(generator); });

	return data;
}


//static 
std::vector<float> RandomGenerator::GenerateDataFloat(size_t size, float min, float max)
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



//static 
std::vector<float> RandomGenerator::FloatPosition(float min, float max)
{
	std::random_device		rd;
	std::mt19937 generator(rd());
    std::uniform_real_distribution<float> distribution_x(min,max);
	std::uniform_real_distribution<float> distribution_y(min,max);
	std::uniform_real_distribution<float> distribution_z(min,max);
    float x = distribution_x(generator); 
	float y = distribution_y(generator); 
	float z = distribution_z(generator); 


	std::vector<float> data;
	
	data.push_back(x);
	data.push_back(y);
	data.push_back(z);

	return data;
}