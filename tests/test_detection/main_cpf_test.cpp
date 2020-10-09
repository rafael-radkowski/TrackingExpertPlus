#include "CPFToolsGPU.h"
#include "CPFTools.h"

void main()
{
	cout << "------------------------------Begin CPFGPU Test--------------------------------" << endl;
	bool error = false;
	PointCloud pc;
	pc.resize(5);
	CPFToolsGPU::AllocateMemory(5);
	cout << "PC size: " << pc.size() << endl;
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

	vector<Matches> matches;
	matches.clear();
	KNN knn = KNN();

	knn.radius(pc, 2.0, matches);
	for (int i = 0; i < matches.size(); i++)
	{
		cout << matches.at(0).matches << endl;
	}
	cout << endl << endl;

	cout << "AngleBetween Test--------------------------------------------------------------" << endl;

	Eigen::Vector3f vec0 = Eigen::Vector3f(0.3, 1.1, 1.8);
	Eigen::Vector3f vec1 = Eigen::Vector3f(0.6, 0.1, 0.55);

	cout << "Regular: ";
	cout << CPFTools::AngleBetween(vec0, vec1) << endl;
	cout << endl;
	cout << "GPU: ";
	cout << CPFToolsGPU::AngleBetween(vec0, vec1) << endl;
	cout << endl;
	if (CPFTools::AngleBetween(vec0, vec1) == CPFToolsGPU::AngleBetween(vec0, vec1))
		cout << "AngleBetween successful!" << endl;
	else
		cout << "AngleBetween failed" << endl;
	cout << endl << endl;


	cout << "GetRefFrame Test--------------------------------------------------------------" << endl;

	vector<Eigen::Affine3f> refFramesGPU;
	Eigen::Affine3f curRefCPU;

	CPFToolsGPU::GetRefFrames(refFramesGPU, pc.points, pc.normals);
	for (int i = 0; i < pc.size(); i++) {
		curRefCPU = CPFTools::GetRefFrame(pc.points.at(i), pc.normals.at(i));

		cout << i << ":" << endl;
		cout << curRefCPU.matrix() << endl;
		cout << refFramesGPU.at(i).matrix() << endl << endl;

		if (!(curRefCPU.matrix().isApprox(refFramesGPU.at(i).matrix())))
			error = true;
	}
	if (!error)
		cout << "GetRefFrame successful!" << endl;
	else
		cout << "GetRefFrame failed" << endl;
	cout << endl << endl;
	
	cout << "DiscretizeCurvature Test--------------------------------------------------------------" << endl;

	error = false;
	vector<uint32_t> CPU_curvatures;
	CPU_curvatures.clear();
	CPU_curvatures.reserve(pc.size());
	vector<uint32_t> GPU_curvatures;
	GPU_curvatures.clear();
	GPU_curvatures.reserve(pc.size());

	CPFToolsGPU::DiscretizeCurvature(GPU_curvatures, pc.normals, pc, matches, 10.0);
	for (int i = 0; i < pc.size(); i++)
	{
		cout << i << ": " << endl;
		float cur_vature = CPFTools::DiscretizeCurvature(pc.points[i], pc.normals[i], pc, matches.at(i), 10);

		cout << "CPU: " << cur_vature << endl;
		cout << "GPU: " << GPU_curvatures.at(i) << endl << endl;

		if (cur_vature != GPU_curvatures.at(i))
			error = true;

		CPU_curvatures.at(i) = cur_vature;
	}
	if (!error)
		cout << "DiscretizeCurvature successful!" << endl;
	else
		cout << "DiscretizeCurvature failed" << endl;
	cout << endl << endl;

	CPFToolsGPU::DeallocateMemory();
}