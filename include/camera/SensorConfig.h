/*
class SensorConfig

Rafael Radkowski
Iowa State University
rafael@iastate.edu
MIT License
---------------------------------------------------------------
*/
#include "ICP.h"
#include "PointCloud.h"
#include "ICPConfig.h"

#ifdef WITH_PCA
#include "PrincipalComponentAnalysis.h"
#endif


#ifdef _RECORD_DATA
// writer for experiment results
#include "ICPResultData.h"
#endif

