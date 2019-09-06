//
//  SensorWatcher.h
//  DenseModelTracking
//
//  Created by David Wehr on 11/7/14.
//  Copyright (c) 2014 Dr. Rafael Radkowski. All rights reserved.
//

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
