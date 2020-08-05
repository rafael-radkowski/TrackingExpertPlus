#pragma once

// STL
#include <iostream>
#include <string>
#include <Windows.h>
#include <fstream>
#include <unordered_map>
#include <algorithm>    
#include <vector>   

// Eigen 3
#include <Eigen\Dense>
#include <Eigen\Geometry>

// glm
#include <glm\glm.hpp>

// local
//#include "CPFTypes.h"
//#include "Types.h"


using namespace std;

class ColorCoder
{
public:

	
	static void CPF2Color(std::vector<uint32_t>& desc, std::vector<glm::vec3>& colors);


};