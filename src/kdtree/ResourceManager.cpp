#include "ResourceManager.h"


namespace ns_ResourceManager{

	Cuda_KdTree*		g_kdtree_ref = NULL;
	int					g_kdtree_ref_cout = 0;



}


using namespace ns_ResourceManager;


//static 
bool ResourceManager::SetKDTree(Cuda_KdTree* kdtree)
{
	if (g_kdtree_ref_cout == 0) {
		g_kdtree_ref = kdtree;
		g_kdtree_ref_cout++;

		return true;
	}

	return false;
}


//static 
Cuda_KdTree* ResourceManager::GetKDTree(void)
{
	if (g_kdtree_ref_cout > 0 ) {
		g_kdtree_ref_cout++;
		return g_kdtree_ref;
	}

	return NULL;
}

//static 
bool ResourceManager::UnrefKDTree(Cuda_KdTree* tree)
{
	if (tree != NULL) {
		g_kdtree_ref_cout--;
		if(g_kdtree_ref_cout == 0) delete g_kdtree_ref;
		return true;
	}
	return false;
}