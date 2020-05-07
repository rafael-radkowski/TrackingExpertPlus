#ifndef __GL_POINT_CLOUD_RENDERER__
#define __GL_POINT_CLOUD_RENDERER__
/*
class: GLPointCloudRenderer

files: GLPointCloudRenderer.h/.cpp

@brief: The class renders a point cloud and normal vectors. 
An instance of this class requires the lcoation of the point cloud. 
It can automatically update the gpu data set before its renders. 

Input data must come in form of two index-aligned Eigen::Vector3f vectors 
with points and normal vectors. 

Note that the class uses glBufferSubData to update the points.

Note that the class uses a glsl vertex and fragment shader program to render the points. 
The normal vectors require a geometry shader for rendering. 
All shaders are hard-coded in GLPointCloudRenderer.cpp (no external files required).

Features:
- Creates and renders points
- Renders normal vectors
- Automatically updates the dataset. 

Rafael Radkowski
Iowa State University
rafael@iasate.edu
Oct, 2019

MIT License
//-----------------------------------------------------------------------
Last edits:

Feb 21, 2020, RR
- Added a geoemetry shader to render normal vectors. 

Feb 22, 2020, RR
- Added a parameter to change between static and dynamic vertex buffer usage.


*/

// stl include
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <mutex>
#include <algorithm>

// GLEW include
#include <GL/glew.h>

// GLM include files
#define GLM_FORCE_INLINE
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>


// local
#include "VertexBuffers.h"			// create vertex buffer object
#include "ShaderProgram.h"			// create a shader program
#include "Types.h"
#include "GLVertexBufferExt.h"		// helpers to create vertex buffers. 
#include "GLTypes.h"

using namespace std;

namespace isu_ar {

	class GLPointCloudRenderer {

	public:

		/*
		Create an point cloud model from a point cloud dataset
		@param src_points - a reference to the lcoation of the point dataset.
		@param scr_normals - a reference to the lcoation of the normal vector dataset.
		@param usage - indicate whether the vertex buffer is used statically or dynamically. Default is STATIC. 
			Set to DYNAMIC if the data changes a lot. 
		*/
		GLPointCloudRenderer(vector<Eigen::Vector3f>& src_points, vector<Eigen::Vector3f>& scr_normals, texpertgfx::GLDataUsage usage = texpertgfx::STATIC);



		/*
		Update the points using the existing references.
		*/
		void updatePoints(void);


		/*
		Draw the obj model
		@param viewMatrix - a view matrix object
		@param projectionMatrix - a  4x4 projection matrix .
		*/
		void draw(glm::mat4 projectionMatrix, glm::mat4 viewMatrix);


		/*
		Return the shader program
		@return - int containing the shader program
		*/
		int getProgram(void) { return program; }


		/*
		Set the model matrix for this object.
		@param m - glm 4x4 matrix
		*/
		void setModelmatrix(glm::mat4 m) { modelMatrix = m; }


		/*
		Set the point cloud color. 
		@param color  -  a color values in RGB format with each value in the range [0,1].
		*/
		void setPointColor(glm::vec3 color);


		/*
		Set the color for the normal vectors
		@param color  -  a color values in RGB format with each value in the range [0,1].
		*/
		void setNormalColor(glm::vec3 color);


		/*
		Set the normal vector rendering length
		@param length - a value > 0.0
		*/
		void setNormalGfxLength(float length);


		/*
		Enable normal rendering. 
		Note that the normal vectors require a geometry shader. 
		@param draw - true renders the normals. 
		*/
		void enableNormalRendering(bool draw);


		/*
		Render the points.. 
		@param draw - true renders the points. 
		*/
		void enablePointRendering(bool draw);



		/*
		Enable automatic point update. 
		The renderer updates the points before each draw if set to 'true'
		Otherwise, one has to call the api 'updatePoints' to manually update the points. 
		Default is 'true'
		@param enable - true enables auto update, false disables it. 
		*/
		void enableAutoUpdate(bool enable);


	private:


	//-------------------------------------------------------------------

		int vaoID[1]; // Our Vertex Array Object
		int vboID[2]; // Our Vertex Buffer Object
		int iboID[1]; // Our Index  Object
		texpertgfx::GLDataUsage _data_usage; // indicates whether the vertex buffer data is static or dynamic. 

		int program;
		int program_normals;
		int program_normals_locations[6];

		int viewMatrixLocation;
		int modelMatrixLocation;
		int projMatrixLocation;

		bool _draw_normals;
		bool _draw_points;
		bool _auto_update;

		std::vector<glm::vec3> points;
		std::vector<glm::vec3> normals;

		int _pos_location;
		int _norm_location;

		std::mutex	_block; // to prevent concurrent access when updating points

		glm::mat4  modelMatrix;
		glm::mat4	_projectionMatrix;
		glm::mat4	_viewMatrix;

		glm::vec3	_point_color;
		glm::vec3	_normal_color;
		float		_normal_length;

		int _N; // number of vertices
		int _I; // number indices


		vector<Eigen::Vector3f>& _points;
		vector<Eigen::Vector3f>& _normals;


	};


}
#endif