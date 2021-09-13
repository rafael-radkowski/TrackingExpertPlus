#include "GLLineRenderer.h"

#define MAX_LINES 10000


using namespace isu_ar;


namespace nsGLLineRenderer
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
		"out vec3 pass_Normal;                                               \n"
		"                                                                  \n"
		"void main(void)                                                   \n"
		"{                                                                 \n"
		"    gl_Position = projectionMatrix * viewMatrix * modelMatrix * vec4(in_Position, 1.0);  \n"
		"    pass_Color = in_Normal;                                        \n"
		"    pass_Normal = in_Normal;                                       \n"
		"}                                                                 \n";




	// Fragment shader source code. This determines the colors in the fragment generated in the shader pipeline. In this case, it colors the inside of our triangle specified by our vertex shader.
	static const string fs_string_410 =
		"#version 410 core                                                 \n"
		"                                                                  \n"
		"uniform vec3 pointcolor;                                          \n"
		"out vec4 color;                                                    \n"
		"void main(void)                                                   \n"
		"{                                                                 \n"
		"    color = vec4(pointcolor, 1.0);                               \n"
		"}                                                                 \n";



};


using namespace nsGLLineRenderer;


GLLineRenderer::GLLineRenderer(vector<Eigen::Vector3f>& src_points0, vector<Eigen::Vector3f>& src_points1, std::vector<std::pair<int, int>>& knn_matches, int program):
	_src_points0(src_points0), _src_points1(src_points1), _knn_matches(knn_matches), _program(program)
{
	init();
}


GLLineRenderer::~GLLineRenderer()
{

}


// init the graphics content. 
void GLLineRenderer::init(void)
{
	//_block.unlock();
    _program = -1;
	_N = 0;
	_I = 0;

	// default point color
	_line_color = glm::vec3(0.0,0.0,1.0);
	_modelMatrix = glm::translate(glm::mat4(1.0f), glm::vec3(0.0f, 0.0f, 0.0f));
	_draw_lines = false;
	_line_width = 1.0;


	
	int size = MAX_LINES * 2;

	for(int i=0; i<size; i++)
	{
		_gl_points0.push_back(glm::vec3(0.0,i * 0.05, i * 0.05 ) );
        _gl_normals0.push_back(_line_color ); // normal vector slot is used to render colors
	}
	_N = _gl_points0.size();


	//----------------------------------------------------------------
	// points vectors
	//_program = cs557::CreateShaderProgram(vs_string_410, gs_string_410, fs_string_410);
	_program = cs557::CreateShaderProgram(vs_string_410, fs_string_410);

	glUseProgram(_program);
	glBindVertexArray(_vaoID[0]);


	_pos_location = glGetAttribLocation(_program, "in_Position");
	_norm_location = glGetAttribLocation(_program, "in_Normal");

	
	glBindAttribLocation(_program, _pos_location, "in_Position");
	glBindAttribLocation(_program, _norm_location, "in_Normal");


	// create a vertex buffer object
	texpertgfx::CreateVertexObjects33(_vaoID, _vboID, &_gl_points0[0].x, &_gl_normals0[0].x, _N, texpertgfx::DYNAMIC, _pos_location, _norm_location );

 
	_program_locations[0]  = glGetUniformLocation(_program, "projectionMatrix"); // Get the location of our projection matrix in the shader
	_program_locations[1]  = glGetUniformLocation(_program, "viewMatrix"); // Get the location of our view matrix in the shader
	_program_locations[2]  = glGetUniformLocation(_program, "modelMatrix"); // Get the location of our model matrix in the shader
	_program_locations[3]  = glGetAttribLocation(_program, "in_Position");
	_program_locations[4]  = glGetAttribLocation(_program, "in_Normal");
	_program_locations[5]  = glGetUniformLocation(_program, "pointcolor");

	
	// default color
	glUniform3fv( _program_locations[5], 1, &_line_color[0]);



	updatePoints();
	
}



/*
Update the points using the existing references.
*/
void GLLineRenderer::updatePoints(void)
{

	if(_src_points0.size() == 0 || _src_points1.size() == 0) {
		std::cout << "[GLLineRenderer]  - ERROR: insufficient points" << std::endl;
		return;
	}
		

	// update the points
	int size = _knn_matches.size();

	//cout << "[GLLineRenderer] Info - update " << size << " valid matches" << endl;

	if (size > MAX_LINES) {
		size = MAX_LINES;
		std::cout << "[GLLineRenderer]  - ERROR: insufficient space for points" << std::endl;
	}

	int count  = 0;
	for(int i=0; i<size; i++)
	{
		int p0 = _knn_matches[i].first;
		int p1 = _knn_matches[i].second;

		_gl_points0[i*2] = glm::vec3( _src_points0[p0].x(),  _src_points0[p0].y(),  _src_points0[p0].z());
		_gl_points0[i*2+1] = glm::vec3( _src_points1[p1].x(),  _src_points1[p1].y(),  _src_points1[p1].z());
		_gl_normals0[i*2] = _line_color;
		_gl_normals0[i*2+1] = _line_color;

		count+=2;
      
	}
	_N = count;

	_block.lock();

	glUseProgram(_program);

	glBindBuffer(GL_ARRAY_BUFFER, _vboID[0]); // Bind our Vertex Buffer Object
	glBufferSubData(GL_ARRAY_BUFFER, 0, (GLsizeiptr)(_N *3* sizeof(GLfloat)),(void*)&_gl_points0[0]); // Set the size and data of our VBO and set it to STATIC_DRAW

	//Normal vectors
	glBindBuffer(GL_ARRAY_BUFFER, _vboID[1]); // Bind our second Vertex Buffer Object
	glBufferSubData(GL_ARRAY_BUFFER, 0, (GLsizeiptr)(_N * 3* sizeof(GLfloat)), (void*) &_gl_normals0[0]); // Set the size and data of our VBO and set it to STATIC_DRAW

	glBindVertexArray(0); // Disable our Vertex Buffer Object


	glUseProgram(_program);
	
	_block.unlock();

}


/*
Update the points using the existing references.
This will copy the points to a vertex buffer object and render them. 
*/
void GLLineRenderer::updatePoints(vector<Eigen::Vector3f>& src_points0, vector<Eigen::Vector3f>& src_points1, std::vector<std::pair<int, int>>& knn_matches)
{
	// update the points
	int size = knn_matches.size();
	if(size == 0) return;


	//_src_points0 = src_points0;
	//_src_points1 = src_points1;

	//cout << "[GLLineRenderer] Info - update " << size << " valid matches" << endl;

	if (size > MAX_LINES) {
		size = MAX_LINES;
		std::cout << "[GLLineRenderer]  - ERROR: insufficient space for points" << std::endl;
	}

	int count  = 0;
	for(int i=0; i<size; i++)
	{
		int p0 = knn_matches[i].first;
		int p1 = knn_matches[i].second;

		_gl_points0[i*2] = glm::vec3( src_points0[p0].x(),  src_points0[p0].y(),  src_points0[p0].z());
		glm::vec4 dst =   glm::vec4( src_points1[p1].x(),  src_points1[p1].y(),  src_points1[p1].z(), 1.0f);

		_gl_points0[i*2+1] = glm::vec3(dst.x, dst.y,  dst.z);
		_gl_normals0[i*2] = _line_color;
		_gl_normals0[i*2+1] = _line_color;

		count+=2;
      
	}
	_N = count;

	_block.lock();

	glUseProgram(_program);

	glBindBuffer(GL_ARRAY_BUFFER, _vboID[0]); // Bind our Vertex Buffer Object
	glBufferSubData(GL_ARRAY_BUFFER, 0, (GLsizeiptr)(_N *3* sizeof(GLfloat)),(void*)&_gl_points0[0]); // Set the size and data of our VBO and set it to STATIC_DRAW

	//Normal vectors
	glBindBuffer(GL_ARRAY_BUFFER, _vboID[1]); // Bind our second Vertex Buffer Object
	glBufferSubData(GL_ARRAY_BUFFER, 0, (GLsizeiptr)(_N * 3* sizeof(GLfloat)), (void*) &_gl_normals0[0]); // Set the size and data of our VBO and set it to STATIC_DRAW

	glBindVertexArray(0); // Disable our Vertex Buffer Object


	glUseProgram(_program);
	
 	_block.unlock();
}


void GLLineRenderer::updateMatches(std::vector<std::pair<int, int>>& knn_matches)
{
	_knn_matches = knn_matches;

	updatePoints();
}


/*
Draw the obj model
@param viewMatrix - a view matrix object
@param modelMatrix - a model matrix object.
*/
void GLLineRenderer::draw(glm::mat4 projectionMatrix, glm::mat4 viewMatrix)
{

	if(!_draw_lines) return;
	
	_projectionMatrix = projectionMatrix;
	_viewMatrix = viewMatrix;

	glEnable(GL_LINE_SMOOTH);
	glUseProgram(_program);

	// Bind the buffer and switch it to an active buffer
	glBindVertexArray(_vaoID[0]);

	// this changes the camera location
	glUniformMatrix4fv(_program_locations[1] , 1, GL_FALSE, &viewMatrix[0][0]); // send the view matrix to our shader
	glUniformMatrix4fv(_program_locations[2] , 1, GL_FALSE, &_modelMatrix[0][0]); // send the model matrix to our shader
	glUniformMatrix4fv(_program_locations[0] , 1, GL_FALSE, &projectionMatrix[0][0]); // send the projection matrix to our shader
	

	glLineWidth(_line_width);
	// Draw the triangles
 	glDrawArrays(GL_LINES, 0, _N);
	
	
	// Unbind our Vertex Array Object
	glBindVertexArray(0);


}


/*
Enable or disable line rendering.
@param enable - true enables the renderer. The default value is false. 
*/
void GLLineRenderer::enableRenderer(bool enable)
{
	_draw_lines = enable;
}

/*
Set the color for all lines. 
@param color  -  a color values in RGB format with each value in the range [0,1].
*/
void GLLineRenderer::setLineColor(glm::vec3 color)
{
	_line_color = color;
	glUseProgram(_program);
	glUniform3fv(  glGetUniformLocation(_program, "pointcolor") ,  1, &_line_color[0] );
}


/*
Set the line width for the renderer
*/
void GLLineRenderer::setLineWidth(float line_width)
{
	_line_width = (std::max)( 1.0f, (std::min)( line_width, 10.0f));
}