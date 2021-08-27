#pragma once
/*
@class GLColorCoder

The class converts a curvature value into a color code. 

The class CPFMatchingExp provides two apis to access the curvatures.
	bool getModelCurvature(const int model_id, std::vector<uint32_t>& model_cu);
and
	bool getSceneCurvature(std::vector<uint32_t>& scene_cu);

These curvatures can be color encoded using this class and passed to the point cloud renderer for rendering. 


Rafael Radkowski
Iowa State University
rafael@iastate.edu

MIT License
---------------------------------------------------------------------------------------------------------------
Last edited:

*/

// STL
#include <iostream>
#include <string>
#ifdef _WIN32
#include <Windows.h>
#endif
#include <fstream>
#include <unordered_map>
#include <algorithm>    
#include <vector>   

// glm
#include <glm/glm.hpp>




using namespace std;

class GLColorCoder
{
public:

	/*
	Encode a curvature value given a uint32_t as an RGB color value. 
	@param desc - vector with curvatures. 
	@param color - vectors with vec3 containing RGB color values. 
	*/
	static void CPF2Color(std::vector<uint32_t>& desc, std::vector<glm::vec3>& colors);


private:

	/*
	 Internal color mapping. 
	 @param p - the curvature value
	 @param np - the number of color/value increments
	 @param r, g, b - locations to store the color values. 
	*/
	static void getcolor(std::uint32_t p, std::uint32_t np, float& r, float& g, float& b);
};
