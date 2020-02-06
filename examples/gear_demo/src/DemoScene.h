#pragma once
/*

*/


// STL
#include <iostream>
#include <string>
#include <fstream>
#include <vector>


// TrackingExpert
#include "graphicsx.h"
#include "ModelOBJ.h"



class DemoScene
{
	public:
		DemoScene();
		~DemoScene();


		/*
		Create the demo scene
		*/
		void create(void);



		void draw(glm::mat4 projectionMatrix, glm::mat4 viewMatrix);

	private:

		// vector containing the models
		std::vector<cs557::OBJModel> _3d_models;
};