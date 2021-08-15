#pragma once
/*
This class implements a video background canvas. 



Rafael Radkowski
Iowa State University
rafael@iastate.edu
MIT License

----------------------------------------------------------------------
Last edited:




*/
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

// GLEW include
#include <GL/glew.h>

// GLM include files
#define GLM_FORCE_INLINE
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>


// local
#include "VertexBuffers.h"			// create vertex buffer object
#include "ShaderProgram.h"			// create a shader program
#include "Texture2D.h"				// create a texture object

namespace isu_gfx {

	class GLVideoCanvas
	{
	public:
		GLVideoCanvas();
		~GLVideoCanvas();


		bool create(int rows, int cols, unsigned char* video_ptr, bool fullscreen);


		/*
		Draw the video background
		@param viewMatrix - a view matrix object
		@param modelMatrix - a model matrix object.
		*/
		void draw(glm::mat4 projectionMatrix, glm::mat4 viewMatrix, glm::mat4 modelMatrix);


		/*
		Return the shader program
		@return - int containing the shader program
		*/
		int getProgram(void){return program;}

	private:
	
		/*
		Update the video content
		*/
		void updateVideo(void);
		


		//---------------------------------------------------------


		int vaoID[1]; // Our Vertex Array Object
		int vboID[2]; // Our Vertex Buffer Object
		
		// the shader program that renders this object
		int program;


		int viewMatrixLocation;
		int modelMatrixLocation;
		int projMatrixLocation;


		float		_width;
		float		_height; 

		int			_rows;
		int			_cols;

		unsigned int _texture_id;

		unsigned char*	_video_ptr;
	};

}; //isu_gfx
