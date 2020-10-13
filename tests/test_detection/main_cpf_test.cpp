#include "CPFToolsGPU.h"
#include "CPFTools.h"

void main()
{
	cout << "------------------------------Begin CPFGPU Test--------------------------------" << endl;
	bool error = false;
	PointCloud pc;
	pc.resize(7);
	CPFToolsGPU::AllocateMemory(5);
	cout << "PC size: " << pc.size() << endl;
	cout << endl;

	pc.points.at(0) = Eigen::Vector3f(-0.5, 0.2, 0.9);
	pc.points.at(1) = Eigen::Vector3f(-0.1, -0.82, 0.6);
	pc.points.at(2) = Eigen::Vector3f(-0.3, -0.3, -0.5);
	pc.points.at(3) = Eigen::Vector3f(0.4, -0.6, -0.9);
	pc.points.at(4) = Eigen::Vector3f(0, 0, 0);
	pc.points.at(5) = Eigen::Vector3f(0.5, 0, 0);
	pc.points.at(6) = Eigen::Vector3f(0.05, -0.35, 0.6);

	pc.normals.at(0) = Eigen::Vector3f(0, 0, 1);
	pc.normals.at(1) = Eigen::Vector3f(1, 0, 0);
	pc.normals.at(2) = Eigen::Vector3f(0, 1, 0);
	pc.normals.at(3) = Eigen::Vector3f(0.59, 0.25, 0.76);
	pc.normals.at(4) = Eigen::Vector3f(-0.77, 0.31, -0.56);
	pc.normals.at(5) = Eigen::Vector3f(-0.77, 0.31, -0.56);
	pc.normals.at(6) = Eigen::Vector3f(1, 0, 0);

	vector<Matches> matches;
	matches.clear();
	KNN knn = KNN();

	knn.populate(pc);
	knn.radius(pc, 2.0, matches);

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
	vector<Eigen::Affine3f> refFramesCPU;
	Eigen::Affine3f curRefCPU;

	CPFToolsGPU::GetRefFrames(refFramesGPU, pc.points, pc.normals);
	for (int i = 0; i < pc.size(); i++) {
		curRefCPU = CPFTools::GetRefFrame(pc.points.at(i), pc.normals.at(i));

		cout << i << ":" << endl;
		cout << curRefCPU.matrix() << endl;
		cout << refFramesGPU.at(i).matrix() << endl << endl;

		refFramesCPU.push_back(curRefCPU);
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
		CPU_curvatures.push_back(CPFTools::DiscretizeCurvature(pc.points[i], pc.normals[i], pc, matches.at(i), 10));

	for (int i = 0; i < CPU_curvatures.size() && i < GPU_curvatures.size(); i++)
	{
		cout << i << ": " << endl;
		cout << "CPU: " << CPU_curvatures.at(i) << endl;
		cout << "GPU: " << GPU_curvatures.at(i) << endl << endl;

		if (CPU_curvatures.at(i) != GPU_curvatures.at(i))
			error = true;
	}
	if (!error)
		cout << "DiscretizeCurvature successful!" << endl;
	else
		cout << "DiscretizeCurvature failed" << endl;
	cout << endl << endl;

	cout << "DiscretizeCPF Test--------------------------------------------------------------" << endl;
	error = false;

	vector<CPFDiscreet> GPU_cpf;
	vector<CPFDiscreet> CPU_cpf;

	CPFToolsGPU::DiscretizeCPF(GPU_cpf, GPU_curvatures, matches, pc.points, refFramesGPU);

	for (int i = 0; i < pc.size(); i++)
	{
		uint32_t cur1 = CPU_curvatures.at(i);
		Eigen::Affine3f T = refFramesCPU.at(i);
		const int num_matches = KNN_MATCHES_LENGTH;

		for (int j = 0; j < num_matches; j++)
		{
			if (matches.at(i).matches[j].distance > 0.0)
			{
				int id = matches.at(i).matches[j].second;
				uint32_t cur2 = CPU_curvatures.at(id);

				Eigen::Vector3f pt = T * pc.points[id];

				CPFDiscreet cpf = CPFTools::DiscretizeCPF(cur1, cur2, pc.points.at(i), pt);

				cpf.point_idx = i;

				float alpha_m = atan2(-pt(2), pt(1));
				cpf.alpha = alpha_m;

				CPU_cpf.push_back(cpf);
			}
		}
	}

	for (int i = 0; i < CPU_cpf.size() && i < GPU_cpf.size(); i++)
	{
		cout << i << ": " << endl;
		cout << "CPU: " << CPU_cpf.at(i).data << endl;
		cout << "GPU: " << GPU_cpf.at(i).data << endl << endl;
		if (!(CPU_cpf.at(i) == GPU_cpf.at(i)))
			error = true;
	}
	if (!error)
		cout << "DiscretizeCPF successful!" << endl;
	else
		cout << "DiscretizeCPF failed" << endl;
	cout << endl << endl;

	CPFToolsGPU::DeallocateMemory();
}