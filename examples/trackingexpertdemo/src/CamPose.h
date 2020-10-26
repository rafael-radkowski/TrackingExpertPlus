#pragma once
/*!
@class   CamPose

@brief  Finds the pose of a multitude of cameras relative to an origin camera 
given images of checkerboards taken by those cameras.

As long as the number of cameras whose positions are being found are provided to
the computer and their respective AzureViewers are available, this class
will calculate the pose of the cameras relative to the origin camera (camera 0)
and give a vector of matrices representing those poses.  

Each camera should be associated with an AzureViewer in order to parse its
intrinsics.  These are used to undistort the camera's image and find its
pose relative to the checkerboard marker.

All pose matrices output in this program are given the following format:

		T =	[r00, r01, r02, tx]
			[r10, r11, r12, ty]
			[r20, r21, r22, tz]
			[  0,   0,   0,  1]

With r being the rotation matrix of the camera and t being its translation vector.

Translation and rotation vectors are given in the formats:

		t = [x]			r = [roll]
			[y]				[pitch]
			[z]				[yaw]

respectively.

Both the translation and rotation vectors are written into .txt files that store them
for future use.  The user chooses the path of where these files are written.


Features:
- Process pictures to find pose matrices of their respective cameras
- Parse translation and rotation vectors from the pose matrices
- Store poses into date_time titled files for future use


William Blanchard
Iowa State University
wsb@iastate.edu
Mobile/Work: +1 (847) 707 - 1421

-------------------------------------------------------------------------------
Last edited:

July 23, 2020, William Blanchard
- Camera intrinsics are now found through the AzureViewer class rather than through
	xml files.

*/
//stl
#include <string>
#include <vector>
#include <stdio.h>
#include <fstream>
#include <iostream>
#include <ctime>

//opencv
#include <opencv2/opencv.hpp>

//local
#include "CameraParameters.h"
#include "KinectAzureCaptureDevice.h"



class CamPose 
{
private:
	int i_numCameras;
	vector<vector<float>> calibrations;
	int icbX;
	int icbY;
	float fcbSquareSize;
	char* lastFile;
	vector<vector<vector<float>>> rgvPoses;

	/*!
	Uses an image of a marker to find its chessboard corners and uses those points to find the distance of the camera
	away from it.
	@param cboardPic - the picture of the marker in CV_8UC3 format.  Other objects may be in the picture, as long as 
		the marker is unobstructed.
	@param cboard3DPts - the scaled 3-dimensional points of the marker's chessboard corners in meters. Format should be as follows:
				[x0, y0, z0]
				[x1, y0, z0]
				[x2, y0, z0]
				...
				[xN, yN, zN]
				where x corresponds to distance from the bottom of the chessboard and y corresponds to the distance from the left
				of the chessboard
	@param calibFilepath - The filepath of the camera's calibration XML file.
	@param debug - if true, displays the corresponding picture with the chessboard points marked for each image taken.
	@return - the camera's pose matrix relative to the marker's bottom-left corner in the format:
			T =	[r00, r01, r02, tx]
				[r10, r11, r12, ty]
				[r20, r21, r22, tz]
				[  0,   0,   0,  1]
	*/
	cv::Affine3f checkerboardPnP(cv::Mat& cboardPic, cv::Mat& cboard3DPts, vector<float> calibs, bool debug = false);

public:

	/*!
	Constructor
	@param numCams - the number of cameras being used by the computer
	@param x - the number of inner corners per row
	@param y - the number of inner corners per column
	@param len - the length of the checkerboard square sides in meters
	*/
	CamPose(int numCams, int x, int y, float len);
	/*!
	Constructor
	*/
	CamPose();

	/*!
	Destructor
	*/
	~CamPose();


	/*!
	Start the calibration program.
	Creates matrices of camera poses relative to an origin camera and places those
	matrices in the order in which they were assigned by the computer.  The translation
	and rotation vectors parsed from these matrices are stored in a .txt file which is 
	written in the folder specified by the path parameter and the pose matrices 
	themselves are returned from this function as a vector.  If the path parameter is
	empty, it will just return a vector of matrices given as follows:
	({T0->T1, T0->T2, ...}).
	@param snaps - the vector of pictures in CV_8UC3 format.
	@param path - the path of the directory the calibration file should be written into.
	@param fName - the name of the file that will be written.  Will be a default name if left empty.
	@param debug - if true, displays each picture taken with the checkerboard points marked with green circles.
	*/
	void start(vector<cv::Mat> snaps, const char* path, bool debug = false, const char* fName = " ");

	/*!
	Read one of the pose files created by the start function and store the information.
	Reads the entirety of a CamPose file and stores the data in the member vectors.  The
	data in these vectors can then be quickly accessed from the getTrans and getRot 
	functions.
	@param path - the path of the pose file (defaults to last file created with this
		CamPose if left empty)
	*/
	void readFile(const char* path = " ");


	/*!
	Return the pose matrix of the specified camera.
	Gives the translation of the specified camera relative to the origin camera
	as a vector from the file specified by the path parameter. Returns zeros 
	(the origin camera position) if cam == 0.  If path is left empty, path
	becomes the filepath of the last pose file.
	
	@param cam - the camera number
	@param path - the path of the pose file
	@return - The 4x4 float pose matrix of the camera
	*/
	vector<vector<float>> getPose(int cam);


	/*!
	Sets the amount of squares on a marker as well as its side length.
	@param x - the number of squares on each column of the marker
	@param y - the number of squares on each row of the marker
	@param len - the length (in meters) of the side of each chessboard square
	@return - true if everything has been successfully set
	*/
	bool setSquares(int x, int y, float len);


	/*!
	Sets the camera calibrations. Its a vector of floats, needs to be in this order:
	float 0;            < Principal point in image, x
	float 1;            < Principal point in image, y
	float 2;            < Focal length x
	float 3;            < Focal length y
	float 4;            < k1 radial distortion coefficient
	float 5;            < k2 radial distortion coefficient
	float 6;            < k3 radial distortion coefficient
	float 7;            < k4 radial distortion coefficient
	float 8;            < k5 radial distortion coefficient
	float 9;            < k6 radial distortion coefficient
	float 10;			< Center of distortion in Z=1 plane, x (only used for Rational6KT)
	float 11;			< Center of distortion in Z=1 plane, y (only used for Rational6KT)
	float 12;           < Tangential distortion coefficient 2
	float 13;           < Tangential distortion coefficient 1
	@param camNum - the camera whose calibration is being changed.
	@param calib - the camera at that index, used to retrieve its calibration.
	@return - true if successful.
	*/
	bool addCameraCalibrations(int camNum, std::vector<float> calib);


	/*!
	Sets the number of cameras being used.
	@param num - the number of cameras to be used.
	@return - a true if the number of cameras in this object was changed.
	*/
	bool setNumCameras(int num);
};