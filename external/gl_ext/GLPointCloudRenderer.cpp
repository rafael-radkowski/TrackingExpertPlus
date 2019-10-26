#include "GLPointCloudRenderer.h"


#define MAX_POINTS 500000


namespace nsGLPointCloudRenderer
{

	static const string vs_string_410 =
		"#version 410 core                                                 \n"
		"                                                                   \n"
		"uniform mat4 projectionMatrix;                                    \n"
		"uniform mat4 viewMatrix;                                           \n"
		"uniform mat4 modelMatrix;                                          \n"
		"in vec3 in_Position;                                               \n"
		"in vec3 in_Normal;                                                  \n"
		"out vec3 pass_Color;                                               \n"
		"out vec2 pass_Texture;												\n"
		"                                                                  \n"
		"void main(void)                                                   \n"
		"{                                                                 \n"
		"    gl_Position = projectionMatrix * viewMatrix * modelMatrix * vec4(in_Position, 1.0);  \n"
		"    pass_Color = in_Normal;                                         \n"
		"	gl_PointSize = 4.0;												\n"	
		"}                                                                 \n";

	// Fragment shader source code. This determines the colors in the fragment generated in the shader pipeline. In this case, it colors the inside of our triangle specified by our vertex shader.
	static const string fs_string_410 =
		"#version 410 core                                                 \n"
		"                                                                  \n"
		"in vec3 pass_Color;                                                 \n"
		"in vec2 pass_Texture;												\n"
		"out vec4 color;                                                    \n"
		"void main(void)                                                   \n"
		"{                                                                 \n"
		"    color = vec4(1.0,0.0,0.0, 1.0);                               \n"
		"}                                                                 \n";



};


using namespace nsGLPointCloudRenderer;
using namespace isu_ar;

 /*
Create an point cloud model from a point cloud dataset
@param pc - the point cloud dataset 
@param shader_program - overwrite the default shader program by passing a hander to the constructor
*/
GLPointCloudRenderer::GLPointCloudRenderer(vector<Eigen::Vector3f>& dst_points, vector<Eigen::Vector3f>& dst_normals):
	_points(dst_points), _normals(dst_normals)
{


	_block = false;
    program = -1;
	_N = 0;
	_I = 0;



	float center_x = 0.0;
	float center_y = 0.0;
	float center_z = 0.0;


	std::vector<int> indices;

	program = -1;
	// create a shader program only if the progrm was not overwritten. 
	if(program == -1)
		program = cs557::CreateShaderProgram(vs_string_410, fs_string_410);

    // Find the id's of the related variable name in your shader code. 
	projMatrixLocation = glGetUniformLocation(program, "projectionMatrix"); // Get the location of our projection matrix in the shader
	viewMatrixLocation = glGetUniformLocation(program, "viewMatrix"); // Get the location of our view matrix in the shader
	modelMatrixLocation = glGetUniformLocation(program, "modelMatrix"); // Get the location of our model matrix in the shader
	_pos_location = glGetAttribLocation(program, "in_Position");
	_norm_location = glGetAttribLocation(program, "in_Normal");

    points.clear();
    normals.clear();

	int size = MAX_POINTS;

	for(int i=0; i<size; i++)
	{
        points.push_back( glm::vec3(0.0,0.0, i * 0.01 ) );
        normals.push_back(glm::vec3(0.0,0.0, i * 0.01 ) );
	}
	_N = points.size();

	// create a vertex buffer object
	cs557::CreateVertexObjects33(vaoID, vboID, &points[0].x, &normals[0].x, _N, _pos_location, _norm_location );

    modelMatrix = glm::translate(glm::mat4(1.0f), glm::vec3(0.0f, 0.0f, 0.0f));
}


/*
Update the points using the existing references. 
*/
void GLPointCloudRenderer::updatePoints(void)
{
	_N = _points.size();

	_block = true;
	    // Enable the shader program
	glUseProgram(program);

	glBindBuffer(GL_ARRAY_BUFFER, vboID[0]); // Bind our Vertex Buffer Object
	glBufferSubData(GL_ARRAY_BUFFER, 0, (GLsizeiptr)(_N *3* sizeof(GLfloat)),(void*)&_points[0]); // Set the size and data of our VBO and set it to STATIC_DRAW

								  //Color
	glBindBuffer(GL_ARRAY_BUFFER, vboID[1]); // Bind our second Vertex Buffer Object
	glBufferSubData(GL_ARRAY_BUFFER, 0, (GLsizeiptr)(_N * 3* sizeof(GLfloat)), (void*) &_normals[0]); // Set the size and data of our VBO and set it to STATIC_DRAW

	glBindVertexArray(0); // Disable our Vertex Buffer Object

	_block = false;

	//draw(_projectionMatrix, _viewMatrix);

}


/*
Draw the obj model
@param viewMatrix - a view matrix object
@param modelMatrix - a model matrix object.
*/
void GLPointCloudRenderer::draw(glm::mat4 projectionMatrix, glm::mat4 viewMatrix)
{

	updatePoints();
	_projectionMatrix = projectionMatrix;
	_viewMatrix = viewMatrix;

	if (_block) return;

    // Enable the shader program
	glUseProgram(program);

	// this changes the camera location
	glUniformMatrix4fv(viewMatrixLocation, 1, GL_FALSE, &viewMatrix[0][0]); // send the view matrix to our shader
	glUniformMatrix4fv(modelMatrixLocation, 1, GL_FALSE, &modelMatrix[0][0]); // send the model matrix to our shader
	glUniformMatrix4fv(projMatrixLocation, 1, GL_FALSE, &projectionMatrix[0][0]); // send the projection matrix to our shader

	// Bind the buffer and switch it to an active buffer
	glBindVertexArray(vaoID[0]);

	// Draw the triangles
 	glDrawArrays(GL_POINTS, 0, _N);
	
	// Unbind our Vertex Array Object
	glBindVertexArray(0);

	// Unbind the shader program
	glUseProgram(0);

}