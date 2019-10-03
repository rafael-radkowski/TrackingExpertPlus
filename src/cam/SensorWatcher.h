/*
class SensorWatcher 

SensorWatcher.h
DenseModelTracking

Created by David Wehr on 11/7/14.

Rafael Radkowski
Iowa State University
rafael@iastate.edu
MIT License
---------------------------------------------------------------
*/

#ifndef __DenseModelTracking__SensorWatcher__
#define __DenseModelTracking__SensorWatcher__

#include <iostream>
//#include "ICPTrackingSim.h"

class ICPTrackingSim;

class SensorWatcher {
private:
    static ICPTrackingSim* simulation;
public:
    static void setSimulation(ICPTrackingSim*);
    static void newFrame(void);
};

#endif /* defined(__DenseModelTracking__SensorWatcher__) */
