#include "CPFToolsGPU.h"
#include "CPFTools.h"

void main()
{
	PointCloud pc;
	CPFToolsGPU::AllocateMemory(20);

	Eigen::Vector3f vec0 = Eigen::Vector3f(0.3, 1.1, 1.8);
	Eigen::Vector3f vec1 = Eigen::Vector3f(0.6, 0.1, 0.55);

	cout << CPFTools::AngleBetween(vec0, vec1) << endl;
	cout << CPFToolsGPU::AngleBetween(vec0, vec1) << endl;
	CPFToolsGPU::DeallocateMemory();
}