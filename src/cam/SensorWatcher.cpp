//
//  SensorWatcher.cpp
//  DenseModelTracking
//
//  Created by David Wehr on 11/7/14.
//  Copyright (c) 2014 Dr. Rafael Radkowski. All rights reserved.
//

#include "SensorWatcher.h"

#ifdef WIN32
ICPTrackingSim* g_the_simulation;
#endif

void SensorWatcher::setSimulation(ICPTrackingSim* sim_to_notify) {
  
#ifdef WIN32
	 g_the_simulation = sim_to_notify;
#else
	simulation = sim_to_notify;
#endif
}

void SensorWatcher::newFrame(void ) {
#ifdef WIN32
    //TODO FIX ME */g_the_simulation->newFrameReady();*/
#else
	simulation->newFrameReady();
#endif
}