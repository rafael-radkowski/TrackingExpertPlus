#pragma once
/*
class: GLLineRenderer

files: GLLineRenderer.h/.cpp

@brief: The class renders lines between point pairs. Its purpose is to debug nearest neighbors
in point clouds. Create an instance of this class, provide the point cloud location and the 
links, and draw. 

The class was created as a debug helper, so it is not the most efficient way to render lines between points. 
Especially since the points get copied a second time to the graphics card (the point cloud renderer already maintains
vbo's with points.).

The point set must be manually updated if the points change. Use updatePoints() for this purpose. 
The api uses glBufferSubData to update the points and the vbo size is currently limited to MAX_LINES 10000.

The line renderer is DISABLED by DEFAULT. Use  enableRenderer(true); to start rendering. 

Note that the class uses a glsl vertex and fragment shader program to render the points. 
All shaders are hard-coded in GLLineRenderer.cpp (no external files required).

Features:
- Creates and renders lines between points


Rafael Radkowski
Iowa State University
rafael@iasate.edu
March, 2020

MIT License
//-----------------------------------------------------------------------
Last edits:

June 11, 2020, RR
- Update the function updatePoints, to pass the latest points to the gpu. 

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



namespace isu_ar
{ 


class GLLineRenderer
{

public:

	/*
	Draws lines between points, as a result of a knn search. Provide the locations of the points and the 
	locations indicating the connections and the class renders them. 
	@param src_points0 - a vector location containing the first point set of type Vector3. 
	@param src_points1 - a vector location containing the second point set of type Vector3
	@param knn_matches - a vector with index pairs pointing from points0 to points1/
	*/
	GLLineRenderer(vector<Eigen::Vector3f>& src_points0, vector<Eigen::Vector3f>& src_points1, std::vector<std::pair<int, int>>& knn_matches, int program = -1 );
	
	/*
	Destructor
	*/
	~GLLineRenderer();


	/*
	Update the points using the existing references.
	This will copy the points to a vertex buffer object and render them. 
	*/
	void updatePoints(void);


	/*
	Update the points using the existing references.
	This will copy the points to a vertex buffer object and render them. 
	*/
	void updatePoints(vector<Eigen::Vector3f>& src_points0, vector<Eigen::Vector3f>& src_points1, std::vector<std::pair<int, int>>& knn_matches);


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
	int getProgram(void) { return _program; }


	/*
	Enable or disable line rendering.
	@param enable - true enables the renderer. The default value is false. 
	*/
	void enableRenderer(bool enable);



	/*
	Set the color for all lines. 
	@param color  -  a color values in RGB format with each value in the range [0,1].
	*/
	void setLineColor(glm::vec3 color);




private:

	// init the graphics content. 
	void init(void);


	//------------------------------------------------------------------------------------------------------

	// the point cloud. Note that these are the original points. 
	// the gl points are re-organized depending on the lines to draw. 
	vector<Eigen::Vector3f>					_src_points0;
	vector<Eigen::Vector3f>					_src_points1;

	// reference that stores the links between the point clouds. 
	// The container contains a list with indices pointing from _src_points0 to _src_points1
	std::vector<std::pair<int, int>>		_knn_matches;

	// the graphics content
	std::vector<glm::vec3>					_gl_points0;
	std::vector<glm::vec3>					_gl_normals0;
	std::vector<int>						_gl_indices;


	int _vaoID[1]; // Our Vertex Array Object
	int _vboID[2]; // Our Vertex Buffer Object
	int _iboID[1]; // Our Index  Object

	int _program;
	int _program_locations[6];

	int viewMatrixLocation;
	int modelMatrixLocation;
	int projMatrixLocation;

	bool _draw_lines;

	int _pos_location;
	int _norm_location;

	std::mutex	_block;

	glm::mat4	_modelMatrix;
	glm::mat4	_projectionMatrix;
	glm::mat4	_viewMatrix;

	glm::vec3	_line_color;

	int _N; // number of vertices
	int _I; // number indices


};

}