#pragma once
#include <iostream>
#include <string>
#include <vector>

#include "Cuda_KdTree.h"



class ResourceManager
{
public:


	static Cuda_KdTree* GetKDTree(void);


	static bool UnrefKDTree(Cuda_KdTree* tree);

private:

};

