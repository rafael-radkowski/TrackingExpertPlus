#pragma once
/*




----------------
Latest edits:

March 20, 2019, RR
- Added a 'stop' function to close the window from code. 
Feb 5, 2020, RR
- Swapped window with and height to see the window nicely shaped. 
*/


#include <iostream>
#include <string>
#include <time.h>
#include <thread>       
#include <vector>
#include <functional>

// GLEW include
#include <GL/glew.h>

// GLM include files
#define GLM_FORCE_INLINE
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtx/transform.hpp>  // transformation
#include <glm/gtx/quaternion.hpp> // quaternions

// glfw includes
#include <GLFW/glfw3.h>

// local
#include "Window.h" // the windows
#include "OpenGLDefaults.h" // some open gl and glew defaults
#include "OpenGLDefaults.h" // some open gl and glew defaults
#include "VertexBuffers.h"  // create vertex buffer object
#include "ShaderProgram.h"  // create a shader program
#include "CommonTypes.h"  // types that all projects use
#include "ModelCoordinateSystem.h"


using namespace std::placeholders;
using namespace std;

namespace isu_ar {

	class GLViewer
	{
	public:

		GLViewer();
		~GLViewer();

		/*
		Create the renderer instance
		@param window_width - width of the window in pixel
		@param window_height - height of the window in pixel
		@param name - label for the window as string
		*/
		bool create(int window_width, int window_height, string name);


		bool addRenderFcn(std::function<void(glm::mat4 pm, glm::mat4 vm)> function);

		/*
		Add a keyboard callback of type
		void name (int key, int action)
		*/
		bool addKeyboardCallback(std::function<void(int, int)> function);

		/*
		Set a view matrix
		@param vm - 4x4 view matrix
		*/
		bool setViewMatrix(glm::mat4 vm);


		/*
		Set the clear color for the applicatoin
		@param clear_color - vector with rgba values .
		*/
		void setClearColor(glm::vec4 clear_color);

		/*
		Start the renderer
		*/
		bool start(void);


		/*
		Stop the renderer
		*/
		bool stop(void);


		/*
		Enable or disable the moveable camera. 
		@param enable - true enables the camera. 
		*/
		bool enableCameraControl(bool enable);


	private:

		/*
		The main draw loop
		*/
		void draw_loop(void);


		//------------------------------------------------------------
		// Members

		// The handle to the window object
		GLFWwindow*						_window;
		glm::mat4						_projMatrix; //  the projection matrix
		glm::mat4						_viewMatrix;  //  the view matrix



		// Set up our green background color
		GLfloat							_clear_color[4];
		GLfloat							_clear_depth[4];


		int								_window_width;
		int								_window_height;

		bool							_running;
		bool							_init_ready;
		bool							_camera_control;

		// a coordinate system
		cs557::CoordinateSystem			_cs;


		vector<std::function<void(glm::mat4 pm, glm::mat4 vm)> > _render_calls;
	};

}