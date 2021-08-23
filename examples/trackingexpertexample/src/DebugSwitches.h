/*
The file defines a set of debug switches that can be used to debug 
individual features of TrackingExpert+

Note that this file is a file used during development. 
It will disapear at one point without warning. So do not rely on its existence. 

Rafael Radkowski
Aug 8, 2021
MIT License
radkowski.dev@google.com
*/
#pragma once


/*
Enable or disable registration features. 
*/
#define _WITH_REGISTRATION


/*
Enable or disable detection features.
*/
//#define _WITH_DETECTION


/*
Render the camera images.
Note that this function is not thread safe. 
It may just work for a few seconds. But the only purpose is to visually
check if we have the right camera images. 
*/
//#define _WITH_AZURE_OUTPUT


/*
Sequential Processing
Processes the camera updates & tracking and the render loop sequentially. 
Removing this switch runs the process in parallel. 
*/
//#define _WITH_SEQUENTIAL
