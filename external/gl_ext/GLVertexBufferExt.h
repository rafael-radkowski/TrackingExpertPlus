/*
@file: GLVertexBufferEct.h/.cpp

This file contains helper function to create a vertex buffer object and to 
load data into the vertex buffer. It can simplify the content creation since
the vertex buffer call is mostly repetative. 

The class extends the functions in VertexBuffer by buffers with GL_DYNAMIC_DRAW usage,
since camera data is changed repeatedly. 

Rafael Radkowski
Iowa State University
rafael@iastate.edu
+1 (515) 294-7044
MIT License

Feb 2020.
-------------------------------------------------------------------------------
Last edited:

. 

*/
#pragma once


// stl include
#include <iostream>
#include <string>

// GLEW include
#include <GL/glew.h>

// GLM include files
#define GLM_FORCE_INLINE
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtx/transform.hpp>

// glfw includes
#include <GLFW/glfw3.h>



using namespace std;


namespace texpertgfx
{


	/*
	Create a vertex array object and vertex buffer object for vertices of size 3 (x, y, z)  along with colors of size 3: (r, g, b)
	@param vaoID - address to store the vertex array object
	@param vboID - address to store the vertex buffer objects. Note, TWO spaces are required to create buffers of vertices and colors. 
	@param vertices - pointer to an array containing vertices as [x0, y0, z0, x1, y1, z1, ...]
	@param colors - pointer to an array containning color as [r0, g0, b0, r1, g1, b1, .....]
	@param N - the number of vertices and colors, NOT THE LENGTH OF THE ARRAY. Note that the vector sizes MUST match. 
	@param static_data_usage - indicate whether the usage is static or dynamic. 'true' sets static useage and 'false' dynamic.
	@param vertices_location - the GLSL vertices location 
	@param normals_location - the GLSL normal vectors locations
	*/
	bool CreateVertexObjects33(int* vaoID, int* vboID, float* vertices, float* colors, int N, bool static_data_usage = true,
								int vertices_location = 0, int normals_location = 1);


}