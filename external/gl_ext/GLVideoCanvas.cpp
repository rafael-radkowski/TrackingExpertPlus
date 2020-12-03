#include "GLVideoCanvas.h"





namespace ns_GLVideoCanvas
{

	static const std::string vs_string_canvas_410 =
		"#version 410 core                                                 \n"
		"                                                                   \n"
		"uniform mat4 projectionMatrix;                                    \n"
		"uniform mat4 viewMatrix;                                           \n"
		"uniform mat4 modelMatrix;                                          \n"
		"in vec3 in_Position;                                               \n"
		"in vec2 in_Texture;                                                 \n"
		"in vec3 in_Normal;                                                  \n"
		"out vec3 pass_Normal;                                              \n"
		"out vec2 pass_Texture;												\n"
		"                                                                  \n"
		"void main(void)                                                   \n"
		"{                                                                 \n"
		"    //gl_Position = projectionMatrix * viewMatrix * modelMatrix * vec4(in_Position, 1.0);  \n"
		"    gl_Position =  modelMatrix * vec4(in_Position, 1.0);  \n"
		"    pass_Normal = in_Normal;                                         \n"
		"	 pass_Texture = in_Texture;										\n"
		"}                                                                 \n";

	// Fragment shader source code. This determines the colors in the fragment generated in the shader pipeline. In this case, it colors the inside of our triangle specified by our vertex shader.
	static const std::string fs_string_canvas_410 =
		"#version 410 core                                                 \n"
		"                                                                  \n"
		"uniform sampler2D video; 											\n"
		"in vec3 pass_Normal;                                                 \n"
		"in vec2 pass_Texture;												\n"
		"out vec4 color;                                                    \n"
		"void main(void)                                                   \n"
		"{                                                                 \n"
		"    color =  texture(video, pass_Texture);                         \n"
		"    gl_FragDepth =  0.99999;                        \n"
		"}                                                                 \n";



};

using namespace isu_gfx;
using namespace ns_GLVideoCanvas;



GLVideoCanvas::GLVideoCanvas()
{
	_video_ptr = NULL;
	_width = 0;
	_height = 0; 
	_rows = 0;
	_cols = 0;
}
		
		
GLVideoCanvas::~GLVideoCanvas()
{

}


/*
Create a simple coordinate system in the centroid
@param length - the length of each unit axis
*/
bool GLVideoCanvas::create(int rows, int cols, unsigned char* video_ptr, bool fullscreen)
{
	_video_ptr = video_ptr;

	float width = 2.0;
	float height = 2.0;

	// This overwrite the default shader program
	program = -1;

	_width = cols;
	_height = rows; 

	float center_x = 0.0;
	float center_y = 0.0;
	float center_z = 0.0;


	float vertices[] = { 
		//--------------------------------------------------------
		// xy-plane, positive z direction, texture coordinates
		-width / 2.0f + center_x, -height / 2.0f + center_y, 0.0f, 0.0f, 1.0f, // 0k
		-width / 2.0f + center_x, height / 2.0f + center_y, 0.0f, 0.0f, 0.0f,
		width / 2.0f + center_x, -height / 2.0f + center_y, 0.0f, 1.0f, 1.0f, // ok
		width / 2.0f + center_x, height / 2.0f + center_y,  0.0f, 1.0f, 0.0f
		
	};


	float normals[] = { 0.0f, 0.0f, 1.0f, //
		0.0f, 0.0f, 1.0f, 
		0.0f, 0.0f, 1.0f, 
		0.0f, 0.0f, 1.0f,
	};


	// create a shader program only if the progrm was not overwritten. 
	if(program == -1)
		program = cs557::CreateShaderProgram(vs_string_canvas_410, fs_string_canvas_410);

	glUseProgram(program);

	int pos_location = glGetAttribLocation(program, "in_Position");
	int normal_location = glGetAttribLocation(program, "in_Normal");
	int texture_location = glGetAttribLocation(program, "in_Texture");


	// create a vertex buffer object
	cs557::CreateVertexObjects53(vaoID, vboID, vertices, normals, 4, 
						  pos_location, texture_location, normal_location);


	// Find the id's of the related variable name in your shader code. 
	projMatrixLocation = glGetUniformLocation(program, "projectionMatrix"); // Get the location of our projection matrix in the shader
	viewMatrixLocation = glGetUniformLocation(program, "viewMatrix"); // Get the location of our view matrix in the shader
	modelMatrixLocation = glGetUniformLocation(program, "modelMatrix"); // Get the location of our model matrix in the shader

	
	glBindAttribLocation(program, pos_location, "in_Position");
	glBindAttribLocation(program, texture_location, "in_Texture");
	glBindAttribLocation(program, normal_location, "in_Normal");


	// create a texture using the video
	cs557::CreateTexture2D(_width, _height, 3, (unsigned char*)_video_ptr, &_texture_id,  GL_CLAMP_TO_BORDER, GL_TEXTURE0);


	// Activate the texture unit and bind the texture. 
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, _texture_id);


    // Fetch the texture location and set the parameter to 0.
    // Note that 0 is the number of the texture unit GL_TEXTURE0.
    int video_location = glGetUniformLocation(program, "video");
    glUniform1i(video_location, 0);
	glUseProgram(0);

	return true;
}





/*
Draw the coordinate system
@param viewMatrix - a view matrix object
@param modelMatrix - a model matrix object.
*/
void GLVideoCanvas::draw(glm::mat4 projectionMatrix, glm::mat4 viewMatrix, glm::mat4 modelMatrix)
{


	// Enable the shader program
	glUseProgram(program);


	// update the video texture
	updateVideo();


	// this changes the camera location
	glUniformMatrix4fv(viewMatrixLocation, 1, GL_FALSE, &viewMatrix[0][0]); // send the view matrix to our shader
	glUniformMatrix4fv(modelMatrixLocation, 1, GL_FALSE, &modelMatrix[0][0]); // send the model matrix to our shader
	glUniformMatrix4fv(projMatrixLocation, 1, GL_FALSE, &projectionMatrix[0][0]); // send the projection matrix to our shader


	 // Bind the buffer and switch it to an active buffer
	glBindVertexArray(vaoID[0]);

	glLineWidth((GLfloat)3.0);

	// Draw the triangles
	glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
	
	// Unbind our Vertex Array Object
	glBindVertexArray(0);

	// Unbind the shader program
	glUseProgram(0);


}


/*
Update the video content
*/
void GLVideoCanvas::updateVideo(void)
{
	if(_texture_id == 0 || _video_ptr == NULL)
		return;

	glTextureSubImage2D(_texture_id, 0, 0,0,_width,_height, GL_BGR, GL_UNSIGNED_BYTE, _video_ptr); 

}
