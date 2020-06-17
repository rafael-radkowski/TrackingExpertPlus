#pragma once

// stl
#include <iostream>
#include <vector>
#include <numeric>
#include <algorithm>
#include <functional>

// cuda
#include <curand.h>
#include <curand_kernel.h>

// local
#include "Cuda_Types.h"


class cuICPMemory
{
public:

	static bool AllocateMemory(void);


	static bool FreeMemory(void);


	static Cuda_Point*	GetCameraDataPtr(void);


	static void SetCameraDataPtr(Cuda_Point* ptr);


	static Cuda_Point*	GetQuerryDataPtr(void);


	static void SetQuerryDataPtr(Cuda_Point* ptr);


	static MyMatches*	GetSearchResultsPtr(void);


	static void	SetSearchResultsPtr(MyMatches*);


	static float3* GetQuerryNormalPtr(void);


	static float3* GetCameraNormalPtr(void);


	static float* GetOutlierResultsPtr(void);

private:


};