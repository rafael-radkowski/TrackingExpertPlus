#include "GLVertexBufferExt.h"


using namespace texpertgfx;



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
bool texpertgfx::CreateVertexObjects33(int* vaoID, int* vboID, float* vertices, float* colors, int N, bool static_data_usage, int vertices_location , int normals_location )
{
	if (vertices == NULL || colors == NULL)
	{
		std::cout << "[ERROR] - CreateVertexObjects33: No vertices or color information given." << std::endl;
		return false;
	}

	glGenVertexArrays(1, (GLuint*)vaoID); // Create our Vertex Array Object
	glBindVertexArray(*vaoID); // Bind our Vertex Array Object so we can use it

	if (vaoID[1] == -1){
		std::cout << "[ERROR] - Vertex array object was not generated." << std::endl;
		return false;
	}

	glGenBuffers(2, (GLuint*)vboID); // Generate our Vertex Buffer Object


	if (vboID[0] == -1 || vboID[1] == -1){
		std::cout << "[ERROR] - One or both vertex buffer objects were not generated." << std::endl;
		return false;
	}

	GLenum data_usage = GL_STATIC_DRAW;
	if(!static_data_usage)  data_usage = GL_DYNAMIC_DRAW;


	// vertices
	glBindBuffer(GL_ARRAY_BUFFER, vboID[0]); // Bind our Vertex Buffer Object
	glBufferData(GL_ARRAY_BUFFER, N * 3 * int(sizeof(GLfloat)), vertices, data_usage); // Set the size and data of our VBO and set it to STATIC_DRAW
	

	glVertexAttribPointer((GLuint)vertices_location, 3, GL_FLOAT, GL_FALSE, 0, 0); // Set up our vertex attributes pointer
	glEnableVertexAttribArray(vertices_location); // Disable our Vertex Array Object


	//Color or normal vectors
	glBindBuffer(GL_ARRAY_BUFFER, vboID[1]); // Bind our second Vertex Buffer Object
	glBufferData(GL_ARRAY_BUFFER, N * 3 * int(sizeof(GLfloat)), colors, data_usage); // Set the size and data of our VBO and set it to STATIC_DRAW

	glVertexAttribPointer((GLuint)normals_location, 3, GL_FLOAT, GL_FALSE, 0, 0); // Set up our vertex attributes pointer
	glEnableVertexAttribArray(normals_location); // Enable the second vertex attribute array

	glBindVertexArray(0); // Disable our Vertex Buffer Object



	return true;

}
