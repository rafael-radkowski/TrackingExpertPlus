#pragma once

/*
-----------------------------------------------------------
Last edits:

Aug 5, 2020, RR
- Added a device type Noen to CaptureDeviceType
*/

namespace texpert{

/*
Datatype to specify the capture device.
*/
typedef enum CaptureDeviceType{
	StructureCore,
	IntelRealSense,
	KinectV2,
	KinectAzure,
	Fotonic,
	None
}CaptureDeviceType;


/*
Datatype to specify the camera component. 
*/
typedef enum CaptureDeviceComponent{
	COLOR,
	DEPTH
}CaptureDeviceComponent;


}//namespace texpert{