#pragma once

/*
This file keeps the data for different sensor configurations such as the focal length, the projection parameters, etc.



Focal length
- Kinect V1: 520.0
- Kinect V2: 361.1
- P60U:






Rafael Radkowski
Iowa State University
Aug. 1st, 2017
rafael@iastate.edu
*/

typedef enum _SensorType
{
	NONE,
	KINECT_V1,
	KINECT_V2,
	STRUCTUR_IO,
	P60U,
	REC_KINECT_V1,
	REC_STRUCTUR_IO,
	REC_P60U

}SensorType;




/*********************************************************************************
Kinect V1
**********************************************************************************/
const struct
{
	const double	focal_length = 520.0;
	const int		depth_width = 640;
	const int		depth_height = 480;

}Kinect_V1;


/*********************************************************************************
Kinect V2
**********************************************************************************/
const struct
{
	const double	focal_length = 361.1;
	const int		depth_width = 512;
	const int		depth_height = 424;
	const int		rgb_width = 960;
	const int		rgb_height = 540;
	const int		rgb_fps = 30;

}Kinect_V2;



/*********************************************************************************
Structure Sensor
**********************************************************************************/
const struct
{
	const double	focal_length = 520.0;
	const int		depth_width = 640;
	const int		depth_height = 480;

}StructureIO;



/*********************************************************************************
P60U
**********************************************************************************/
const struct
{
	const double	focal_length_x = 577.0;
	const double	focal_length_y = 577.0;
	const double	principle_x = 0.0;
	const double	principle_y = 0.0;
	const int		depth_width = 640;
	const int		depth_height = 480;
	const int		rgb_width = 640;
	const int		rgb_height = 480;
	const int		rgb_fps = 30;

}Fotonic_P60;