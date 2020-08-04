#include "GLPointCloudRenderer.h"


#define MAX_POINTS 500000
#define _USE_PER_VERTEX_COLOR

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
		"in vec3 in_Color;                                                  \n"
		"out vec3 pass_Color;                                               \n"
		"out vec3 pass_Normal;                                               \n"
		"out vec2 pass_Texture;												\n"
		"                                                                  \n"
		"	out Vertex														 \n"
		"	{																 \n"
		"	  vec3 pos;	 \n"
		"	  vec3 normal;	 \n"
		"	  vec3 color;	 \n"
		"	} vertex;	 \n"
		"void main(void)                                                   \n"
		"{                                                                 \n"
		"    gl_Position = projectionMatrix * viewMatrix * modelMatrix * vec4(in_Position, 1.0);  \n"
		"    pass_Color = in_Color;                                        \n"
		"    pass_Normal = in_Normal;                                       \n"
		"	 gl_PointSize = 8.0;											\n"	
		"	vertex.normal = in_Normal;  \n"
		"	vertex.color = in_Color;  \n"
		"	vertex.pos = in_Position; \n"	
		"}                                                                 \n";


	static const string gs_string_410 =
		"#version 410 core                                                 \n"
		"uniform mat4 projectionMatrix;                                    \n"
		"uniform mat4 viewMatrix;                                           \n"
		"uniform mat4 modelMatrix;                                          \n"
		"uniform float normal_length;                                          \n"
		"																	\n"
		"layout(points) in;													\n"
		"//layout(points, max_vertices=1) out;								\n"
		"layout(line_strip, max_vertices=2) out;								\n"
		"	in Vertex														 \n"
		"	{																 \n"
		"	  vec3 pos;	 \n"
		"	  vec3 normal;	 \n"
		"	  vec3 color;	 \n"
		"	} vertex[];	 \n"
		"																	\n"
		"void main()														\n"
		"{																	\n"
		"	for(int i=0; i<1; i++)											\n"
		"	{																\n"
		"		gl_Position = projectionMatrix * viewMatrix * modelMatrix * vec4(vertex[0].pos,1.0);							\n"
		"		//gl_PointSize = 4.0;											\n"
		"		EmitVertex();												\n"
		"		gl_Position = projectionMatrix * viewMatrix * modelMatrix *  (vec4(vertex[0].pos,1.0)+ normal_length  * vec4(vertex[0].normal,0.0));		\n"
		"		//gl_PointSize = 4.0;											\n"
		"		EmitVertex();												\n"
		"	}																\n"
		"EndPrimitive();													\n"
		"}																	\n";



	// Fragment shader source code. This determines the colors in the fragment generated in the shader pipeline. In this case, it colors the inside of our triangle specified by our vertex shader.
	static const string fs_string_410 =
		"#version 410 core                                                 \n"
		"                                                                  \n"
		"uniform vec3 pointcolor;                                          \n"
		"in vec2 pass_Texture;												\n"
		"in vec3 pass_Color;                                               \n"
		"out vec4 color;                                                    \n"
		"void main(void)                                                   \n"
		"{                                                                 \n"
#ifdef _USE_PER_VERTEX_COLOR
		"    color = vec4(pass_Color, 1.0);                               \n"
#else
		"    color = vec4(pointcolor, 1.0);                               \n"
#endif
		"}                                                                 \n";



};


using namespace nsGLPointCloudRenderer;
using namespace isu_ar;



 /*
Create an point cloud model from a point cloud dataset
@param pc - the point cloud dataset 
@param shader_program - overwrite the default shader program by passing a hander to the constructor
*/
GLPointCloudRenderer::GLPointCloudRenderer(vector<Eigen::Vector3f>& src_points, vector<Eigen::Vector3f>& src_normals, texpertgfx::GLDataUsage usage):
	_points(src_points), _normals(src_normals), _data_usage(usage)
{


	//_block.unlock();
    program = -1;
	_N = 0;
	_I = 0;

	// default point color
	_point_color = glm::vec3(1.0,0.0,0.0);
	_normal_color = glm::vec3(0.0,0.2,0.8);
	_normal_length = 0.05f;

	_draw_normals = false;
	_draw_points = true;
	_auto_update = true;

	float center_x = 0.0;
	float center_y = 0.0;
	float center_z = 0.0;


	std::vector<int> indices;

	program = -1;
	// create a shader program only if the progrm was not overwritten. 
	if(program == -1)
		program = cs557::CreateShaderProgram(vs_string_410,  fs_string_410);

    // Find the id's of the related variable name in your shader code. 
	projMatrixLocation = glGetUniformLocation(program, "projectionMatrix"); // Get the location of our projection matrix in the shader
	viewMatrixLocation = glGetUniformLocation(program, "viewMatrix"); // Get the location of our view matrix in the shader
	modelMatrixLocation = glGetUniformLocation(program, "modelMatrix"); // Get the location of our model matrix in the shader
	_pos_location = glGetAttribLocation(program, "in_Position");
	_norm_location = glGetAttribLocation(program, "in_Normal");
	_color_location = glGetAttribLocation(program, "in_Color");

	// default color
	glUniform3f(glGetUniformLocation(program, "pointcolor") , (GLfloat)1.0f, (GLfloat)0.0f, (GLfloat)1.0f );

    points.clear();
    normals.clear();

	int size = MAX_POINTS;

	for(int i=0; i<size; i++)
	{
		points.push_back( glm::vec3(0.0,0.0, i * 0.01 ) );
        normals.push_back(glm::vec3(0.0,0.0, i * 0.01 ) );
		colors.push_back(glm::vec3(0.0,0.0,1.0));
	}
	_N = points.size();


	// create a vertex buffer object
	bool data_usage = false;
	if(_data_usage == texpertgfx::DYNAMIC) data_usage = false;
#ifdef _USE_PER_VERTEX_COLOR
	texpertgfx::CreateVertexObjects333(vaoID, vboID, &points[0].x, &normals[0].x, &colors[0].x, _N, data_usage, _pos_location, _norm_location, _color_location );
#else
	texpertgfx::CreateVertexObjects33(vaoID, vboID, &points[0].x, &normals[0].x, _N, data_usage, _pos_location, _norm_location );
#endif
    modelMatrix = glm::translate(glm::mat4(1.0f), glm::vec3(0.0f, 0.0f, 0.0f));

	

	//----------------------------------------------------------------
	// normal vectors
	program_normals = cs557::CreateShaderProgram(vs_string_410, gs_string_410, fs_string_410);

	glUseProgram(program_normals);
	glBindVertexArray(vaoID[0]);

	
	glBindAttribLocation(program_normals, _pos_location, "in_Position");
	glBindAttribLocation(program_normals, _norm_location, "in_Normal");
	glBindAttribLocation(program_normals, _color_location, "in_Color");

	program_normals_locations[0]  = glGetUniformLocation(program_normals, "projectionMatrix"); // Get the location of our projection matrix in the shader
	program_normals_locations[1]  = glGetUniformLocation(program_normals, "viewMatrix"); // Get the location of our view matrix in the shader
	program_normals_locations[2]  = glGetUniformLocation(program_normals, "modelMatrix"); // Get the location of our model matrix in the shader
	program_normals_locations[3]  = glGetAttribLocation(program_normals, "in_Position");
	program_normals_locations[4]  = glGetAttribLocation(program_normals, "in_Normal");
	program_normals_locations[5]  = glGetUniformLocation(program_normals, "pointcolor");
	program_normals_locations[6]  = glGetAttribLocation(program_normals, "in_Color");

	// binds the vbos to the shader
	glBindBuffer(GL_ARRAY_BUFFER, vboID[0]);
	glVertexAttribPointer((GLuint)program_normals_locations[3] , 3, GL_FLOAT, GL_FALSE, 0, 0); 
	glEnableVertexAttribArray(program_normals_locations[3] );

	glBindBuffer(GL_ARRAY_BUFFER, vboID[1]);
	glVertexAttribPointer((GLuint)	program_normals_locations[4], 3, GL_FLOAT, GL_FALSE, 0, 0); 
	glEnableVertexAttribArray(	program_normals_locations[4]);

#ifdef _USE_PER_VERTEX_COLOR
	glBindBuffer(GL_ARRAY_BUFFER, vboID[2]);
	glVertexAttribPointer((GLuint)	program_normals_locations[6], 3, GL_FLOAT, GL_FALSE, 0, 0); 
	glEnableVertexAttribArray(	program_normals_locations[6]);
#endif	
	
	// default color
	glUniform3fv( program_normals_locations[5], 1, &_normal_color[0]);
	glUniform1f(  glGetUniformLocation(program_normals, "normal_length"), _normal_length );
	
}




/*
Update the points using the existing references. 
*/
void GLPointCloudRenderer::updatePoints(void)
{
	if(!_auto_update)return;

	_N = _points.size();

	if(_N <= 0) return;

	_block.lock();

//	glUseProgram(program);

	glBindBuffer(GL_ARRAY_BUFFER, vboID[0]); // Bind our Vertex Buffer Object
	glBufferSubData(GL_ARRAY_BUFFER, 0, (GLsizeiptr)(_N *3* sizeof(GLfloat)),(void*)&_points[0]); // Set the size and data of our VBO and set it to STATIC_DRAW

	//Normal vectors
	glBindBuffer(GL_ARRAY_BUFFER, vboID[1]); // Bind our second Vertex Buffer Object
	glBufferSubData(GL_ARRAY_BUFFER, 0, (GLsizeiptr)(_N * 3* sizeof(GLfloat)), (void*) &_normals[0]); // Set the size and data of our VBO and set it to STATIC_DRAW

	glBindVertexArray(0); // Disable our Vertex Buffer Object

	_block.unlock();

}


/*
Draw the obj model
@param viewMatrix - a view matrix object
@param modelMatrix - a model matrix object.
*/
void GLPointCloudRenderer::draw(glm::mat4 projectionMatrix, glm::mat4 viewMatrix)
{
	// update the points
	updatePoints();

	_projectionMatrix = projectionMatrix;
	_viewMatrix = viewMatrix;

	//if (!_block.try_lock()) return;
	_block.lock();

    // Enable the shader program
	glUseProgram(program);

	// this changes the camera location
	glUniformMatrix4fv(viewMatrixLocation, 1, GL_FALSE, &viewMatrix[0][0]); // send the view matrix to our shader
	glUniformMatrix4fv(modelMatrixLocation, 1, GL_FALSE, &modelMatrix[0][0]); // send the model matrix to our shader
	glUniformMatrix4fv(projMatrixLocation, 1, GL_FALSE, &projectionMatrix[0][0]); // send the projection matrix to our shader
	glUniform3fv(glGetUniformLocation(program, "pointcolor"), 1 , &_point_color[0] );
	

	// Bind the buffer and switch it to an active buffer
	glBindVertexArray(vaoID[0]);
	

	// Draw the triangles
	if(_draw_points)
 		glDrawArrays(GL_POINTS, 0, _N);
	
	// Unbind our Vertex Array Object
	//glBindVertexArray(0);

	// Unbind the shader program
	//glUseProgram(0);

	if (!_draw_normals) {

		
		// Unbind our Vertex Array Object
		glBindVertexArray(0);

		// Unbind the shader program
		glUseProgram(0);

		_block.unlock();
		return;
	}

	glUseProgram(program_normals);
	// this changes the camera location
	glUniformMatrix4fv(program_normals_locations[1] , 1, GL_FALSE, &viewMatrix[0][0]); // send the view matrix to our shader
	glUniformMatrix4fv(program_normals_locations[2] , 1, GL_FALSE, &modelMatrix[0][0]); // send the model matrix to our shader
	glUniformMatrix4fv(program_normals_locations[0] , 1, GL_FALSE, &projectionMatrix[0][0]); // send the projection matrix to our shader
	glUniform3fv(  glGetUniformLocation(program_normals, "pointcolor") ,  1, &_normal_color[0] );


	// Bind the buffer and switch it to an active buffer
	//glBindVertexArray(vaoID[0]);

	//glUseProgram(program_normals);
	
	// Draw the triangles
 	glDrawArrays(GL_POINTS, 0, _N);
	
	
	// Unbind our Vertex Array Object
	glBindVertexArray(0);


	// Unbind the shader program
	glUseProgram(0);

	// unlock the mutex
	_block.unlock();
}


/*
Set the point cloud color. 
@param color  -  a color values in RGB format with each value in the range [0,1].
*/
void GLPointCloudRenderer::setPointColor(glm::vec3 color)
{
	// check limits
	color.r = std::min(1.0f,  std::max(0.0f, color.r) );
	color.g = std::min(1.0f,  std::max(0.0f, color.g) );
	color.b = std::min(1.0f,  std::max(0.0f, color.b) );

	// the color is used during runtime since the shader pogram is re-used. 
	_point_color = color;

#ifdef _USE_PER_VERTEX_COLOR
	int size = MAX_POINTS;
	for(int i=0; i<size; i++)
	{
		colors[i] = color;
	}

	//Normal vectors
	glBindBuffer(GL_ARRAY_BUFFER, vboID[2]); // Bind our second Vertex Buffer Object
	glBufferSubData(GL_ARRAY_BUFFER, 0, (GLsizeiptr)(_N * 3* sizeof(GLfloat)), (void*) &colors[0]); // Set the size and data of our VBO and set it to STATIC_DRAW

	glBindVertexArray(0); // Disable our Vertex Buffer Object
#endif

}



/*
Set a color value per vertex.
Note, this requires to define #define _USE_PER_VERTEX_COLOR in GLPointCloudRenderer.cpp
@param per_vertex_color - a list with one color per vertex. 
*/
void GLPointCloudRenderer::setPointColors(vector<glm::vec3> per_vertex_color)
{
#ifdef _USE_PER_VERTEX_COLOR
	int size = per_vertex_color.size();
	for(int i=0; i<size; i++)
	{
		colors[i] = per_vertex_color[i];
	}

	//Normal vectors
	glBindBuffer(GL_ARRAY_BUFFER, vboID[2]); // Bind our second Vertex Buffer Object
	glBufferSubData(GL_ARRAY_BUFFER, 0, (GLsizeiptr)(_N * 3* sizeof(GLfloat)), (void*) &colors[0]); // Set the size and data of our VBO and set it to STATIC_DRAW

	glBindVertexArray(0); // Disable our Vertex Buffer Object
#else

	std::cout << "[ERROR] - per vertex color for point cloud rendering is disabled. The function setPointColors(...) has not effect." << std::endl;
#endif
}


/*
Set the color for the normal vectors
@param color  -  a color values in RGB format with each value in the range [0,1].
*/
void  GLPointCloudRenderer::setNormalColor(glm::vec3 color)
{
	// check limits
	color.r = std::min(1.0f,  std::max(0.0f, color.r) );
	color.g = std::min(1.0f,  std::max(0.0f, color.g) );
	color.b = std::min(1.0f,  std::max(0.0f, color.b) );

	_normal_color = color;
}


/*
Enable normal rendering. 
Note that the normal vectors require a geometry shader. 
@param draw - true renders the normals. 
*/
void GLPointCloudRenderer::enableNormalRendering(bool draw)
{
	_draw_normals = draw;
}

/*
Render the points.. 
@param draw - true renders the points. 
*/
void GLPointCloudRenderer::enablePointRendering(bool draw)
{
	_draw_points = draw;
}


/*
Enable automatic point update. 
The renderer updates the points before each draw if set to 'true'
Otherwise, one has to call the api 'updatePoints' to manually update the points. 
Default is 'true'
@param enable - true enables auto update, false disables it. 
*/
void GLPointCloudRenderer::enableAutoUpdate(bool enable)
{
	_auto_update = enable;
}



/*
Set the normal vector rendering length
@param length - a value > 0.0
*/
void GLPointCloudRenderer::setNormalGfxLength(float length)
{
	_normal_length = std::max(length, 0.0001f);
	glUseProgram(program_normals);
	glUniform1f(  glGetUniformLocation(program_normals, "normal_length"), _normal_length );
	glUseProgram(0);
}