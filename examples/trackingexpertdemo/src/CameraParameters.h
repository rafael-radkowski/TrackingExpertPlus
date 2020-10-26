#ifndef __CAMERAPARAMS__
#define __CAMERAPARAMS__
/*
class CameraParameters

The class reads and writes intrinsic camera parameters from/to a file. 
The class uses OpenCV FileStorage for that and uses a json file to store these parameters. 

The intrinsic matrix is expected to be a 3x3 matrix of type CV_64FC1 containing
		[ fx 0  ux ]
		[ 0  fc uy ]
		[ 0  0  1  ]

and the distortion parameters come in the opencv typical form as
		[ k1, k2, p1, p2, k3 ]

with k, the radial distortion parameters, and p, the tangential distortion parameters.

Note that this application does not use the distortion parameters since we assume that the 
camera image one work with is undistorted. So those can be kept [0,0,0,0,0]

The methods of this class are static. 

Rafael Radkowski
Iowa State University
rafael@iastate.edu
+1 (515) 294 7044
May 2019
All copyrights reserved
---------------------------------------------------------------------------------------------------
Last edits:

Dec 10, 2019, RR:
- Added FileUtils.h to address the deprecation of experimental/filesystem

*/
// stl
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <algorithm>
#include <random>
#include <time.h>

// opencv
#include <opencv2/opencv.hpp>

// local
#include "FileUtils.h"

using namespace std;

class CameraParameters {

public:

	/*@brief: read camera parameters from a json file. 

	The function reads the camera matrix, distortion parameters, and the camera width
	and height from a json file. Use the keywords intrinsic, dist, and imgSize to 
	indicate the related variables. 

	@param path_and_file - string with the relative or absolute path.
	@param verbose - outputs the parameters to display if true.
	@return: true, if the parameters were successfully loaded. False otherwise.
	*/
	static bool Read(string path_and_file, bool verbose = false);

	/*@brief: Write the camera parameters to a file. 
	@param path_and_file - string with the relative or absolute path.
	@return: true, if the parameters were successfully written. False otherwise.
	*/
	static bool Write(string path_and_file);

	/*@brief: return the intrinsic matrix as 3x3 matrix of type CV_64FC1
	@return the intrinsic 3x3 matrix of type CV_64FC1
	*/
	static cv::Mat getIntrinsic(void);

	/*@brief: return the distortion parameters as 5x1 matrix of type CV_64FC1
	@return: matrix with the distortion parameters. 
	*/
	static cv::Mat getDistortions(void);

	/*@brief: return the image width and height
	@return the image dimensions as cv::Size ( width, height ) in pixels. 
	*/
	static cv::Size getImgSize(void);

	/*@brief: return the field of view from the intrinsic matrix as angles in degree.
	The function needs a valid intrinsic matrix. Thus, one needs to read the matrix with Read(...) first.
	The return is invalid otherwise. 
	@return: the field of view (foyx, fovy) in degree. (0,0) if no intrinsic matrix is given. 
	*/
	static cv::Size2f getIntrinsicAsFoV(void);


private:

};




#endif