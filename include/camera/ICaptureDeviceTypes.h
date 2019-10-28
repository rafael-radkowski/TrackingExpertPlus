#pragma once


namespace texpert{

/*
Datatype to specify the capture device.
*/
typedef enum CaptureDeviceType{
	StructureCore,
	IntelRealSense,
	KinectV2,
	KinectAzure,
	Fotonic
}CaptureDeviceType;


/*
Datatype to specify the camera component. 
*/
typedef enum CaptureDeviceComponent{
	COLOR,
	DEPTH
}CaptureDeviceComponent;


}//namespace texpert{