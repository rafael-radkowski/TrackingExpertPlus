#include "CPFToolsGPU.h"
#include "CPFTools.h"

void main()
{
	cout << "------------------------------Begin CPFGPU Test--------------------------------" << endl;
	PointCloud pc;
	pc.resize(5);
	CPFToolsGPU::AllocateMemory(5);
	cout << "pc size: " << pc.size() << endl;
	cout << endl;

	pc.points.at(0) = Eigen::Vector3f(-0.5, 0.2, 0.9);
	pc.points.at(1) = Eigen::Vector3f(-0.1, -0.82, 0.6);
	pc.points.at(2) = Eigen::Vector3f(-0.3, -0.3, -0.5);
	pc.points.at(3) = Eigen::Vector3f(0.4, -0.6, -0.9);
	pc.points.at(4) = Eigen::Vector3f(0, 0, 0);

	pc.normals.at(0) = Eigen::Vector3f(0, 0, 1);
	pc.normals.at(1) = Eigen::Vector3f(1, 0, 0);
	pc.normals.at(2) = Eigen::Vector3f(0, 1, 0);
	pc.normals.at(3) = Eigen::Vector3f(0.59, 0.25, 0.76);
	pc.normals.at(4) = Eigen::Vector3f(-0.77, 0.31, -0.56);

	cout << "AngleBetween Test--------------------------------------------------------------" << endl;

	Eigen::Vector3f vec0 = Eigen::Vector3f(0.3, 1.1, 1.8);
	Eigen::Vector3f vec1 = Eigen::Vector3f(0.6, 0.1, 0.55);

	cout << "Regular:" << endl;
	cout << CPFTools::AngleBetween(vec0, vec1) << endl;
	cout << endl;
	cout << "GPU:" << endl;
	cout << CPFToolsGPU::AngleBetween(vec0, vec1) << endl;
	cout << endl << endl;


	cout << "GetRefFrame Test--------------------------------------------------------------" << endl;

	vector<Eigen::Affine3f> refFramesGPU = vector<Eigen::Affine3f>(pc.size());

	cout << "Regular:" << endl;
	for (int i = 0; i < pc.size(); i++)
		cout << CPFTools::GetRefFrame(pc.points.at(i), pc.normals.at(i)).matrix() << endl;
	cout << endl;
	cout << "GPU:" << endl;
	CPFToolsGPU::GetRefFrames(refFramesGPU, pc.points, pc.normals);
	for (int i = 0; i < pc.size(); i++)
		cout << refFramesGPU.at(i).matrix() << endl;
	cout << endl << endl;
	

	CPFToolsGPU::DeallocateMemory();
}