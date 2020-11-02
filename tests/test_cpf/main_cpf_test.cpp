#include "CPFToolsGPU.h"
#include "CPFTools.h"
#include "CPFMatchingExp.h"
#include "CPFMatchingExpGPU.h"
#include "RandomGenerator.h"
#include "CPFMatchingWrapper.h"

using namespace texpert;

void run_non_stress()
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

	vector<CPFDiscreet> GPU_cpf = vector<CPFDiscreet>();
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

	CPFDiscreet curCPU;
	CPFDiscreet curGPU;

	for (int i = 0; i < CPU_cpf.size() && i < GPU_cpf.size(); i++)
	{
		curCPU = CPU_cpf.at(i);
		curGPU = GPU_cpf.at(i);
		cout << i << ": " << endl;
		cout << "CPU: " << curCPU.data[0] << ", " << curCPU.data[1] << ", " << curCPU.data[2] << ", " << curCPU.data[3] << endl;
		cout << "GPU: " << curGPU.data[0] << ", " << curGPU.data[1] << ", " << curGPU.data[2] << ", " << curGPU.data[3] << endl << endl;
		for (int j = 0; j < CPU_cpf.size(); j++)
		{
			if (!(CPU_cpf.at(j) == curGPU)) 
			{
				error = true;
			}
			else
			{
				error = false;
				break;
			}
		}
	}
	if (!error)
		cout << "DiscretizeCPF successful!" << endl;
	else
		cout << "DiscretizeCPF failed" << endl;
	cout << endl << endl;

	CPFToolsGPU::DeallocateMemory();
}

/*
-----------------------------Stress Tests----------------------------------------------
*/

KNN* knn;

PointCloud*	points;

std::vector<MyMatches> matches;

float tolerance = 0.0000001;

/*
Function to generate random point clouds and normal vectors. Note that the normal vectors are just
points.
@param pc - reference to the location for the point cloud.
@param num_points - number of points to generatel
@param min, max - the minimum and maximum range of the points.
*/
void GenerateRandomPointCloud(PointCloud& pc, int num_points, float min = -2.0, float max = 2.0)
{
	pc.points.clear();
	pc.normals.clear();

	for (int i = 0; i < num_points; i++) {
		vector<float> p = RandomGenerator::FloatPosition(min, max);

		pc.points.push_back(Eigen::Vector3f(p[0], p[1], p[2]));
		pc.normals.push_back(Eigen::Vector3f(p[0], p[1], p[2]));
	}
	pc.size();
}

bool AbsCPFEquals(CPFDiscreet& a, CPFDiscreet& b)
{
	float tolerance = 0.00001f;

	if (a == b && abs(a.alpha - b.alpha) <= tolerance && a.point_idx == b.point_idx) return true;

	return false;
}

/*
Compare two set of matches. For each search point, the naive method and the kd-tree should
find the identical match. Thus, the function compares the point indices and reports an error,
if the point-pairs do not match.
@param matches0 - the location with the first set of matches
@param matches1 - the location with the second set of matches.
*/
int CompareMatches(std::vector<MyMatches>& matches0, std::vector<MyMatches>& matches1)
{
	int s0 = matches0.size();
	int s1 = matches1.size();

	if (s0 != s1) {
		std::cout << "Error - matches have not the same size " << s0 << " to " << s1 << endl;
	}

	int error_count = 0;

	for (int i = 0; i < s0; i++) {
		if (matches0[i].matches[0].second != matches1[i].matches[0].second) {
			//std::cout << "Found error for i = " << i << " with gpu " << matches0[i].matches[0].second << " and naive " << matches1[i].matches[0].second  << " with distance " << matches0[i].matches[0].distance << " and " << matches1[i].matches[0].distance * matches1[i].matches[0].distance << std::endl;
			error_count++;
		}
	}

	float error_percentage = float(error_count) / float(s0) * 100.0;

	std::cout << "[INFO] - Found " << error_count << " in total (" << error_percentage << "%)" << std::endl;

	// When working with the kd-tree, some minor errors can be expected. Those are the result of a integer conversion, the tree
	// works with a Radix search. Also, the tree does not backtrack into adjacent branches indefinitely. 
	// The error does not matter when working with point cloud data from cameras, since the camera tolerances yield larger variances. 
	// The error was never larger than 5%. If you encounter a larger error, this requires furter investigation but may not point to a bug, etc. 
	if (error_percentage > 5.0) {
		std::cout << "[ERROR] - The last run yielded an error > 5% with " << error_percentage << "%. That is higher than expected." << std::endl;
	}


	return error_count;
}

void CompareValues(std::vector<uint32_t> values0, std::vector<uint32_t> values1)
{
	int s0 = values0.size();
	int s1 = values1.size();

	if (s0 != s1) {
		std::cout << "Error - matches have not the same size " << s0 << " to " << s1 << endl;
	}

	int error_count = 0;

	for (int i = 0; i < s0; i++)
	{
		if (values0.at(i) != values1.at(i))
			error_count++;
	}

	float error_percentage = float(error_count) / float(s0) * 100.0;

	if (error_percentage > 0)
		std::cout << "[ERROR] - " << error_percentage << "% of values don't match" << endl;
}

void run_stress(PointCloud pc, int iteration)
{
	cout << "Test " << iteration << ": " << endl;
	matches.clear();
	knn = new KNN();

	knn->populate(pc);
	knn->radius(pc, 2.0, matches);

	int ref_errors = 0;
	vector<Eigen::Affine3f> refFramesGPU;
	vector<Eigen::Affine3f> refFramesCPU;
	Eigen::Affine3f curRefCPU;

	CPFToolsGPU::GetRefFrames(refFramesGPU, pc.points, pc.normals);
	if (refFramesGPU.empty())
	{
		cout << "GetRefFrames Failed!" << endl;
		return;
	}

	for (int i = 0; i < pc.size(); i++) {
		curRefCPU = CPFTools::GetRefFrame(pc.points.at(i), pc.normals.at(i));

		refFramesCPU.push_back(curRefCPU);
		if (!(curRefCPU.matrix().isApprox(refFramesGPU.at(i).matrix())))
			ref_errors++;
	}

	float err_ratio = ((float)ref_errors / (float)pc.size()) * 100;
	cout << "GetRefFrame: Found " << ref_errors << " ( about " << err_ratio << "% ) errors." << endl;

	int curve_error = 0;
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
		if (CPU_curvatures.at(i) != GPU_curvatures.at(i))
			curve_error++;
	}

	err_ratio = ((float)curve_error / (float)CPU_curvatures.size()) * 100;
	cout << "DiscretizeCurvature: Found " << curve_error << " ( about " << err_ratio << "% ) errors." << endl;

	int cpf_error = 0;

	vector<CPFDiscreet> GPU_cpf = vector<CPFDiscreet>();
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

				for (int k = 0; k < 3; k++)
				{
					if (!(abs(pt(k)) > 10e-6))
						pt(k) = 0;
				}

				CPFDiscreet cpf = CPFTools::DiscretizeCPF(cur1, cur2, pc.points.at(i), pt);

				cpf.point_idx = i;

				float alpha_m = atan2(-pt(2), pt(1));
				cpf.alpha = alpha_m;

				CPU_cpf.push_back(cpf);
			}
		}
	}

	CPFDiscreet curCPU;
	CPFDiscreet curGPU;
	vector<CPFDiscreet> discErr;
	vector<CPFDiscreet> discErrCPU;
	bool has_error;

	if (CPU_cpf.size() != GPU_cpf.size())
		cout << "WARNING: Result of CPU and GPU DiscretizeCPF functions not of the same size." << endl;

	for (int i = 0; i < CPU_cpf.size() && i < GPU_cpf.size(); i++)
	{
		has_error = true;
		curCPU = CPU_cpf.at(i);
		curGPU = GPU_cpf.at(i);

		if (!(AbsCPFEquals(curCPU, curGPU)))
		{
			cpf_error++;
			discErr.push_back(curGPU);
			discErrCPU.push_back(curCPU);
		}

		//for (int j = 0; j < CPU_cpf.size(); j++)
		//{
		//	if ((CPU_cpf.at(j).point_idx == curGPU.point_idx) && (CPU_cpf.at(j) == curGPU))
		//	{
		//		has_error = false;
		//		break;
		//	}
		//}

		//if (has_error) {
		//	cpf_error++;
		//	discErr.push_back(curGPU);
		//	discErrCPU.push_back(curCPU);
		//}
	}

	err_ratio = ((float)cpf_error / (float)CPU_cpf.size()) * 100;
	cout << "DiscretizeCPF: Found " << cpf_error << " ( about " << err_ratio << "% ) errors." << endl;
	//for (int i = 0; i < discErr.size(); i++)
	//{
	//	cout << discErr.at(i).data[0] << ", " << discErr.at(i).data[1] << ", " << discErr.at(i).data[2] << ", " << discErr.at(i).data[3] << " &&& " << 
	//		discErrCPU.at(i).data[0] << ", " << discErrCPU.at(i).data[1] << ", " << discErrCPU.at(i).data[2] << ", " << discErrCPU.at(i).data[3] << endl;
	//}
	for (int i = 0; i < discErr.size(); i++)
	{
		cout << discErr.at(i).alpha << " &&& " << discErrCPU.at(i).alpha << endl;
	}



	knn->reset();
}

void main()
{
	//run_non_stress();

	float tolerance = 0.0000001f;

	//1. Test AngleBetween
	cout << "-----Begin AngleBetween Test-----" << endl;
	for (int i = 0; i < 20; i++)
	{
		vector<float> vec0 = RandomGenerator::FloatPosition(-i, i);
		vector<float> vec1 = RandomGenerator::FloatPosition(-i, i);

		Eigen::Vector3f v0 = Eigen::Vector3f(vec0[0], vec0[1], vec0[2]);
		Eigen::Vector3f v1 = Eigen::Vector3f(vec1[0], vec1[1], vec1[2]);

		float acpu = CPFTools::AngleBetween(v0, v1);
		float agpu = CPFToolsGPU::AngleBetween(v0, v1);

		if (!(agpu <= acpu + tolerance || agpu >= acpu - tolerance))
			std::cout << "[ERROR] - AngleBetween fails for CPFToolsGPU for vectors " << v0 << " and " << v1 << endl
				<< "CPU: " << acpu << ", GPU: " << agpu << endl;
	}
	cout << endl;

	points = new PointCloud();

	//2. Test normal range, small point size
	cout << "-----Begin stress test: Normal range, small point size-----" << endl;
	CPFToolsGPU::AllocateMemory(100);
	for (int i = 0; i < 20; i++)
	{
		GenerateRandomPointCloud(*points, 100);
		run_stress(*points, i);
	}
	CPFToolsGPU::DeallocateMemory();
	cout << endl;

	//CPFToolsGPU::AllocateMemory(10000);
	////3. Test normal range, large point size
	//cout << "-----Begin stress test: Normal range, large point size-----" << endl;
	//for (int i = 0; i < 5; i++)
	//{
	//	GenerateRandomPointCloud(*points, 10000);
	//	run_stress(*points, i);
	//}
	//cout << endl;

	////4. Test large range, large point size
	//cout << "-----Begin stress test: Large range, large point size-----" << endl;
	//for (int i = 0; i < 5; i++)
	//{
	//	GenerateRandomPointCloud(*points, 10000, -3.0, 3.0);
	//	run_stress(*points, i);
	//}
	//cout << endl;

	////5. Test small range, large point size
	//cout << "-----Begin stress test: Small range, large point size-----" << endl;
	//for (int i = 0; i < 5; i++)
	//{
	//	GenerateRandomPointCloud(*points, 10000, -1.0, 1.0);
	//	run_stress(*points, i);
	//}
	//cout << endl;
	//CPFToolsGPU::DeallocateMemory();


	//Begin CPFMatchingExpGPU tests

	CPFParams cpfParams = CPFParams();

	CPFMatchingExp* cpuMatching = new CPFMatchingExp();
	CPFMatchingExpGPU* gpuMatching = new CPFMatchingExpGPU();
	cpuMatching->setVerbose(true, 2);
	gpuMatching->setVerbose(true, 2);

	cpuMatching->setParams(cpfParams);
	gpuMatching->setParams(cpfParams);

	GenerateRandomPointCloud(*points, 1000, -1.0, 1.0);

	PointCloud* scene = new PointCloud();
	GenerateRandomPointCloud(*scene, 1000, -1.0, 1.0);

	cpuMatching->setScene(*scene);
	gpuMatching->setScene(*scene);

	int cpu_id = cpuMatching->addModel(*points, "Cloud1");
	int gpu_id = gpuMatching->addModel(*points, "Cloud1");

	if (cpu_id == -1) cout << "ERROR: CPFMatchingExp: Could not add model" << endl;
	if (gpu_id == -1) cout << "ERROR: CPFMatchingExpGPU: Could not add model" << endl;

	if (!cpuMatching->match(cpu_id)) cout << "ERROR: CPFMatchingExp: match function did not work" << endl;
	if (!gpuMatching->match(gpu_id)) cout << "ERROR: CPFMatchingExpGPU: match function did not work" << endl;

	vector<Eigen::Affine3f> poses_gpu, poses_cpu;
	vector<int> pose_votes_gpu, pose_votes_cpu;

	cpuMatching->getPose(cpu_id, poses_cpu, pose_votes_cpu);
	gpuMatching->getPose(gpu_id, poses_gpu, pose_votes_gpu);

	//for (int i = 0; i < poses_gpu.size() && i < poses_cpu.size(); i++)
	//{
	//	cout << i << ": GPU: Votes: " << pose_votes_gpu.at(i) << endl;
	//	cout << poses_gpu.at(i).matrix() << endl;
	//	cout << "CPU: Votes: " << pose_votes_cpu.at(i) << endl;
	//	cout << poses_cpu.at(i).matrix() << endl;
	//}

	//CPFMatchingWrapper test
	CPFMatchingWrapper* wrapped = new CPFMatchingExp();
	wrapped->setVerbose(true, 2);
	wrapped->setParams(cpfParams);

	wrapped->setScene(*scene);
	cpu_id = wrapped->addModel(*points, "Cloud2");
	if (cpu_id == -1) cout << "ERROR: CPFMatchingExp: Could not add model" << endl;
	if (!wrapped->match(cpu_id)) cout << "ERROR: CPFMatchingExp: match function did not work" << endl;
	wrapped->getPose(cpu_id, poses_cpu, pose_votes_cpu);

	wrapped = new CPFMatchingExpGPU();
	wrapped->setVerbose(true, 2);
	wrapped->setParams(cpfParams);

	wrapped->setScene(*scene);
	gpu_id = wrapped->addModel(*points, "Cloud2");
	if (gpu_id == -1) cout << "ERROR: CPFMatchingExp: Could not add model" << endl;
	if (!wrapped->match(gpu_id)) cout << "ERROR: CPFMatchingExp: match function did not work" << endl;
	wrapped->getPose(gpu_id, poses_gpu, pose_votes_gpu);

	for (int i = 0; i < poses_gpu.size() && i < poses_cpu.size(); i++)
	{
		cout << i << ": GPU: Votes: " << pose_votes_gpu.at(i) << endl;
		cout << poses_gpu.at(i).matrix() << endl;
		cout << "CPU: Votes: " << pose_votes_cpu.at(i) << endl;
		cout << poses_cpu.at(i).matrix() << endl;
	}
}