#define _CRT_SECURE_NO_WARNINGS

#include "CamPose.h"

CamPose::CamPose(int numCams, int x, int y, float len)
	:i_numCameras((numCams >= 0) ? numCams : numCams * -1), icbX(x), icbY(y), fcbSquareSize(len)
{
	lastFile = (char*)malloc(100 * sizeof(char));
	//changed here, it automatically creates devices if the vector constructor has a size arg and it cannot do that
	calibrations = vector<vector<float>>(numCams);
	rgvPoses = vector<vector<vector<float>>>(1, vector<vector<float>>(3, vector<float>(3, 0)));
}
CamPose::CamPose()
{
	lastFile = (char*)malloc(100 * sizeof(char));
	rgvPoses = vector<vector<vector<float>>>(1, vector<vector<float>>(3, vector<float>(3, 0)));
}

CamPose::~CamPose() {}



void CamPose::start(vector<cv::Mat> snaps, const char* path, bool debug, const char* fName)
{
	cv::setUseOptimized(true);
	//Generate the 3D points of the checkerboard
	cv::Mat cboard3DPts = cv::Mat::zeros((icbX * icbY), 3, cv::DataType<float>::type);
	int i_arrPos = 0;
	
	//Return empty vector if the values for icby and icbX aren't larger than 0.
	if (!(icbY > 0 || icbX > 0))
	{
		cout << "ERROR: The values of icbY and icbX must be larger than zero" << endl;
		return;
	}

	if (calibrations.empty())
	{
		cout << "ERROR: Please provide the calibrations to associate with your cameras" << endl;
		return;
	}

	/*for (int i = 0; i < calibrations.size(); i++)
	{
		if (!rgavViewers.at(i).isOpen())
		{
			cout << "ERROR: All AzureViewers must be connected to a camera." << endl;
			return;
		}
	}*/

	for (int i = 0; i < icbY; i++)
	{
		for (int j = 0; j < icbX; j++, i_arrPos++) 
		{
			cboard3DPts.at<float>(i_arrPos, 0) = i * fcbSquareSize;
			cboard3DPts.at<float>(i_arrPos, 1) = j * fcbSquareSize;
			cboard3DPts.at<float>(i_arrPos, 2) = 0;
		}
	}

	//Check if the number of cameras is larger than 0 and return if it is empty. 
	if (i_numCameras <= 0)
	{
		cout << "ERROR: The number of cameras being used must be larger than zero." << endl;
		return;
	}

	
	vector<cv::Affine3f> camPositions = vector<cv::Affine3f>(i_numCameras);

	//Finds and loads the pose matrices of each camera into camPositions using checkerboardPnP
	for (int camNum = 0; camNum < i_numCameras; camNum++)
		camPositions.at(camNum) = checkerboardPnP(snaps.at(camNum), cboard3DPts, calibrations.at(camNum), debug);

	//Finds the inverse of the origin camera pose matrix
	cv::Affine3f origin = camPositions.at(0);

	cv::Matx33f R = origin.rotation();

	cv::Vec3f t = origin.translation();

	R = R.t();
	t = -1 * R * t;

	cv::Affine3f originInverse(R, t);

	char* title = (char*)malloc(100 * sizeof(char));

	if (std::strcmp(fName, " ") == 0) {
		time_t t = time(0);
		tm* ltm = localtime(&t);
		//Create title of the file
		sprintf(title, "%s\\%d%d%d_%d%d%d.txt", path,
			ltm->tm_mday,			//Current day
			1 + (ltm->tm_mon),		//Current month
			1900 + (ltm->tm_year),	//Current year
			ltm->tm_hour,			//Current hour
			ltm->tm_min,			//Current minute
			ltm->tm_sec);			//Current second
	}
	else {
		sprintf(title, "%s\\%s.txt", path, fName);
	}

	ofstream file(title);

	if (!file.is_open()) {
		cout << "ERROR: Directory cannot be opened" << endl;
		return;
	}
	else lastFile = title;
	file << i_numCameras << " =  numCameras\n";
	char* line = (char*)malloc(300 * sizeof(char));
	//Calculates the distance of each camera from the origin camera and displays that distance on the screen
	for (int camNum = 0; camNum < i_numCameras; camNum++)
	{
		cv::Affine3f compMat = camPositions.at(camNum);
		cv::Affine3f result = originInverse.concatenate(compMat);

		cv::Matx44f res = result.matrix;

		//switched x and Z axis, and negate them. Not sure why this was necesarry? ______________________________________________________________________________________________________
		//float temp = res(0, 3);
		//res(0, 3) = -1*res(2, 3);
		//res(2, 3) = -1* res(2, 3);
		
		
		//Write to file if the file exists
		if (file.is_open()) {
			
			sprintf(line, "%d = index\n[%f, %f, %f, %f]\n[%f, %f, %f, %f]\n[%f, %f, %f, %f]\n[%f, %f, %f, %f]\n", camNum, 

				res(0, 0), res(0, 1), res(0, 2), res(0, 3),
				res(1, 0), res(1, 1), res(1, 2), res(1, 3), 
				res(2, 0), res(2, 1), res(2, 2), res(2, 3),
				res(3, 0), res(3, 1), res(3, 2), res(3, 3));
			file << line;
		}
	}
	if(file.is_open()) file.close();
}



void CamPose::readFile(const char* path)
{
	if (path == " ") path = lastFile;

	ifstream ifs(path);

	if (!ifs.is_open())
	{
		vector<vector<float>> idMatrix = vector<vector<float>>(4, vector<float>(4, 0));
		idMatrix.at(0).at(0) = 1;
		idMatrix.at(1).at(1) = 1;
		idMatrix.at(2).at(2) = 1;
		idMatrix.at(3).at(3) = 1;
		rgvPoses = vector<vector<vector<float>>>(i_numCameras, idMatrix);

		cout << "ERROR - Cannot find pose file at given path" << endl;

		return;
	}

	char* line = (char*)malloc(50 * sizeof(char));
	ifs.getline(line, 50);

	int numCams = 0;
	sscanf(line, "%d", &numCams);

	rgvPoses = vector<vector<vector<float>>>(numCams);

	vector<vector<float>> curPose(4);
	vector<float> curLine(4);

	for (int i = 0; i < numCams; i++)
	{
		ifs.getline(line, 50);
		ifs.getline(line, 50);
		sscanf(line, "[%f, %f, %f, %f]", &curLine.at(0), &curLine.at(1), &curLine.at(2), &curLine.at(3));
		curPose.at(0) = vector<float>(curLine);
		ifs.getline(line, 50);
		sscanf(line, "[%f, %f, %f, %f]", &curLine.at(0), &curLine.at(1), &curLine.at(2), &curLine.at(3));
		curPose.at(1) = vector<float>(curLine);
		ifs.getline(line, 50);
		sscanf(line, "[%f, %f, %f, %f]", &curLine.at(0), &curLine.at(1), &curLine.at(2), &curLine.at(3));
		curPose.at(2) = vector<float>(curLine);
		ifs.getline(line, 50);
		sscanf(line, "[%f, %f, %f, %f]", &curLine.at(0), &curLine.at(1), &curLine.at(2), &curLine.at(3));
		curPose.at(3) = vector<float>(curLine);

		rgvPoses.at(i) = vector<vector<float>>(curPose);
	}
}



vector<vector<float>> CamPose::getPose(int cam)
{
	return rgvPoses.at(cam);
}


bool CamPose::setNumCameras(int num)
{
	//Check if num is >= 0. It should not be negative. 
	if (num < 0)
	{
		cout << "ERROR: The number of cameras must be larger than zero." << endl;
		return false;
	}
	
	if (i_numCameras != num) {
		i_numCameras = num;
		calibrations.resize(num);
		return true;
	}
	return false;
}



bool CamPose::setSquares(int x, int y, float len) {
	if (x < 0 || y < 0 || len < 0)
	{
		cout << ("ERROR: Values negative.  Please only use positive values.");
		return false;
	}

	icbX = x;
	icbY = y;
	fcbSquareSize = len;

	return true;
}



bool CamPose::addCameraCalibrations(int camNum, std::vector<float> calib)
{
	

	calibrations.at(camNum) = calib;
	return true;
}



cv::Affine3f CamPose::checkerboardPnP(cv::Mat& cboardPic, cv::Mat& cboard3DPts, vector<float> calibs, bool debug)
{
	//Check if icbX and icbY is larger than 0
	if (!(icbX > 0 || icbY > 0))
	{
		return NULL;
	}

	//Find the marker corners for reference points
	cv::Mat detectedCorners;
	if (!cv::findChessboardCorners(cboardPic, cv::Size(icbX, icbY), detectedCorners))
	{
		cout << "ERROR: Cannnot detect chessboard corners" << endl;
		return NULL;
	}

	//Get the camera parameters from the AzureViewer passed into the program
	//vector<float> camCalib = azureCam.getCalibration(texpert::COLOR);
	vector<float> camCalib = calibs;
	if (calibs.size() == 1)
	{
		std::cout << "Camera could not return calibrations. No file can be generated. \n";
		return cv::Affine3f(0);
	}
	cv::Mat intrinsics = (cv::Mat_<float>(3, 3) <<
		camCalib.at(2), 0.0f,			camCalib.at(0),
		0.0f,			camCalib.at(3), camCalib.at(1),
		0.0f,			0.0f,			1.0f);
	cv::Mat dist = (cv::Mat_<float>(8, 1) <<
		camCalib.at(4), camCalib.at(5), camCalib.at(13), camCalib.at(12), camCalib.at(6), camCalib.at(7), camCalib.at(8), camCalib.at(9));

	//Find the rotation and translation vectors of the cameras using solvePnP.  
	cv::Mat camRot = (cv::Mat_<float>(3, 1) << 0.5, 0.5, 0.5);
	cv::Mat camTrans = (cv::Mat_<float>(3, 1) << 0.5, 0.5, 0.5);

	cv::solvePnP(cboard3DPts, detectedCorners, intrinsics, dist, camRot, camTrans, false);
	cv::solvePnP(cboard3DPts, detectedCorners, intrinsics, dist, camRot, camTrans, true);


	cout << camRot.at<float>(0)*180/3.14 << " "<< camRot.at<float>(1) * 180 / 3.14 << " " << camRot.at<float>(2) * 180 / 3.14 << " angle\n";
	cout << camTrans.at<float>(0) << " " << camTrans.at<float>(1) << " " << camTrans.at<float>(2) << " translation\n";


	//DEBUG: Write points found to png file, output camRot and camTrans vectors to console.
	
	if (debug)
	{
		cv::Mat debugPts;
		cv::projectPoints(cboard3DPts, camRot, camTrans, intrinsics, dist, debugPts);

		for (int i = 0; i < debugPts.rows; i++)
		{
			cv::Point pt = cv::Point_<float>(debugPts.at<float>(i, 0), debugPts.at<float>(i, 1));

			cv::circle(cboardPic, pt, 10, cv::Scalar(0, 0, 255), 5);
		}

		char* title = (char*)malloc(100 * sizeof(char));
		sprintf(title, "test\\test%f.png", intrinsics.at<float>(0, 0));

		cv::Size size(1280, 720);

		cv::namedWindow(title);

		cv::Mat img;
		cv::resize(cboardPic, img, size);
		cv::imshow(title, img);
		cv::moveWindow(title, 0, 0);

		cv::waitKey(0);
	}
	
	//Return 4x4 pose matrix
	return cv::Affine3f(camRot, camTrans);
}
