#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda/cutil_math.h>
#include "CPFToolsGPU.h"

namespace nsCPFToolsGPU {
	int pc_size;

	int* angle_bins;
	float* max_ang_value;
	float* min_ang_value;

	float3* vectorsA;
	float3* vectorsB;
	float3* pcN;
	Matches* pt_matches;
	int* int_p;

	//GetReferenceFrames
	float4* RefFrames;

	//DiscretizeCurvature
	double2* curvature_pairs;
	int* discretized_curvatures;

	//DiscretizeCPF
	CPFDiscreet* discretized_cpfs;
}

using namespace texpert;
using namespace nsCPFToolsGPU;

void CPFToolsGPU::AllocateMemory(uint32_t size)
{
	pc_size = size;

	cudaMallocManaged(&angle_bins, sizeof(int));
	cudaMallocManaged(&max_ang_value, sizeof(int));
	cudaMallocManaged(&min_ang_value, sizeof(int));
	*angle_bins = 12;
	*max_ang_value = 0.0;
	*min_ang_value = 10000000.0;

	cudaMallocManaged(&vectorsA, size * sizeof(float3));
	cudaMallocManaged(&vectorsB, size * sizeof(float3));
	cudaMallocManaged(&pcN, size * sizeof(float3));
	cudaMallocManaged(&pt_matches, size * sizeof(Matches));
	cudaMallocManaged(&int_p, sizeof(int));

	cudaMallocManaged(&RefFrames, size * 4 * sizeof(float4));

	cudaMallocManaged(&curvature_pairs, size * sizeof(double2));
	cudaMallocManaged(&discretized_curvatures, size * sizeof(int));
	
	cudaMallocManaged(&discretized_cpfs, size * KNN_MATCHES_LENGTH * sizeof(CPFDiscreet));
}

void CPFToolsGPU::DeallocateMemory()
{
	cudaFree(angle_bins);
	cudaFree(max_ang_value);
	cudaFree(min_ang_value);

	cudaFree(vectorsA);
	cudaFree(vectorsB);
	cudaFree(pcN);
	cudaFree(pt_matches);
	cudaFree(int_p);

	cudaFree(RefFrames);

	cudaFree(curvature_pairs);
	cudaFree(discretized_curvatures);

	cudaFree(discretized_cpfs);
}

void vecToPointer3F(float3* dst, const vector<Eigen::Vector3f>& src)
{
	Eigen::Vector3f curVec;
	for (int i = 0; i < src.size(); i++) {
		curVec = src.at(i);
		dst[i] = make_float3(curVec[0], curVec[1], curVec[2]);
	}
}

void pointerToVecM4F(vector<Eigen::Affine3f>& dst, float4* src, int num_src)
{

	Eigen::Affine3f affine;
	for (int i = 0; i < num_src; i++)
	{
		affine.matrix() << src[i * 4].x, src[i * 4].y, src[i * 4].z, src[i * 4].w,
				src[(i * 4) + 1].x, src[(i * 4) + 1].y, src[(i * 4) + 1].z, src[(i * 4) + 1].w,
				src[(i * 4) + 2].x, src[(i * 4) + 2].y, src[(i * 4) + 2].z, src[(i * 4) + 2].w,
				src[(i * 4) + 3].x, src[(i * 4) + 3].y, src[(i * 4) + 3].z, src[(i * 4) + 3].w;

		dst.push_back(Eigen::Affine3f(affine));
	}
}

void vecToPointerM4F(float4* dst, vector<Eigen::Affine3f>& src)
{
	Eigen::Matrix4f curMatrix;
	for (int i = 0; i < src.size(); i++)
	{
		curMatrix = src.at(i).matrix();
		dst[i * 4] = make_float4(curMatrix(0), curMatrix(4), curMatrix(8), curMatrix(12));
		dst[(i * 4) + 1] = make_float4(curMatrix(1), curMatrix(5), curMatrix(9), curMatrix(13));
		dst[(i * 4) + 2] = make_float4(curMatrix(2), curMatrix(6), curMatrix(10), curMatrix(14));
		dst[(i * 4) + 3] = make_float4(curMatrix(3), curMatrix(7), curMatrix(11), curMatrix(15));
	}
}

void vecToPointerI(int* dst, std::vector<uint32_t>& src)
{
	for (int i = 0; i < src.size(); i++)
	{
		dst[i] = src.at(i);
	}
}

void pointerToVecI(std::vector<uint32_t>& dst, int* src, int num_e)
{
	for (int i = 0; i < num_e; i++)
		dst.push_back(src[i]);
}



//static
__host__ __device__
float AngleBetweenGPU(float3 a, float3 b)
{
	if (a.x == b.x && a.y == b.y && a.z == b.z)
		return 0.0f;

	float an = sqrtf(powf(a.x, 2) + powf(a.y, 2) + powf(a.z, 2));
	float bn = sqrtf(powf(b.x, 2) + powf(b.y, 2) + powf(b.z, 2));
	float3 a_norm = make_float3(a.x / an, a.y / an, a.z / an);
	float3 b_norm = make_float3(b.x / bn, b.y / bn, b.z / bn);
	float3 c = cross(a_norm, b_norm);


	double c_norm = sqrt(powf(c.x, 2) + powf(c.y, 2) + powf(c.z, 2));
	double ab_dot = (a_norm.x * b_norm.x) + (a_norm.y * b_norm.y) + (a_norm.z * b_norm.z);

	//printf("%f \n", atan2(c_norm, ab_dot));
	return atan2(c_norm, ab_dot);
}

float CPFToolsGPU::AngleBetween(const Eigen::Vector3f& a, const Eigen::Vector3f& b)
{
	return AngleBetweenGPU(make_float3(a(0), a(1), a(2)), make_float3(b(0), b(1), b(2)));
}



__global__
void GetRefFrameGPU(float3* p, float3* n, int numPts, float4* res)
{
	int i = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (i >= numPts) return;

	float3 point = p[i] * -1; //the point is never used in its original form, only in its negated form
	float3 normal = n[i];

	float3 axis = cross(normal, make_float3(1, 0, 0));

	if (axis.x == 0.0f && axis.y == 0.0f && axis.z == 0.0f)
	{
		axis = make_float3(1, 0, 0);
	}
	else {
		float an = sqrt(powf(axis.x, 2) + powf(axis.y, 2) + powf(axis.z, 2));
		axis = make_float3(axis.x / an, axis.y / an, axis.z / an);
	}

	// create an angle axis transformation that rotates A degrees around the axis.
	float theta =	AngleBetweenGPU(normal, make_float3(1, 0, 0)); //Angle between the surface normal and the x axis

	float cost =	cosf(theta); // cos(theta)
	float omcost =	1 - cost; // 1 - cos(theta)
	float sint =	sinf(theta); // sin(theta)

	float xy =		axis.x * axis.y; // ux * uy
	float yz =		axis.y * axis.z; // uy * uz
	float xz =		axis.x * axis.z; // ux * uz

	//Find the rotation matrix of this reference frame
	float3* rot = (float3*)malloc(3 * sizeof(float3));
	rot[0] = make_float3(cost + (powf(axis.x, 2) * omcost), (xy * omcost) - (axis.z * sint), (xz * omcost) + (axis.y * sint));
	rot[1] = make_float3((xy * omcost) + (axis.z * sint), cost + (powf(axis.y, 2) * omcost), (yz * omcost) - (axis.x * sint));
	rot[2] = make_float3((xz * omcost) - (axis.y * sint), (yz * omcost) + (axis.x * sint), cost + (powf(axis.z, 2) * omcost));

	//Translate the point to fit in the new rotation frame
	float3 tinf = make_float3(rot[0].x * point.x + rot[0].y * point.y + rot[0].z * point.z,
		rot[1].x * point.x + rot[1].y * point.y + rot[1].z * point.z,
		rot[2].x * point.x + rot[2].y * point.y + rot[2].z * point.z);

	// create the reference frame
	res[i * 4] =		make_float4(rot[0].x, rot[0].y, rot[0].z, tinf.x);
	res[(i * 4) + 1] =  make_float4(rot[1].x, rot[1].y, rot[1].z, tinf.y);
	res[(i * 4) + 2] =  make_float4(rot[2].x, rot[2].y, rot[2].z, tinf.z);
	res[(i * 4) + 3] =  make_float4(0, 0, 0, 1);
}

//static
void CPFToolsGPU::GetRefFrames(vector<Eigen::Affine3f>& dst, vector<Eigen::Vector3f>& p, vector<Eigen::Vector3f>& n)
{
	//Map Eigen vectors to float3 pointers
	vecToPointer3F(vectorsA, p);
	vecToPointer3F(vectorsB, n);

	int threads = 32;
	int blocks = ceil((float) p.size() / threads);
	GetRefFrameGPU<<<blocks, threads>>>(vectorsA, vectorsB, p.size(), RefFrames);
	cudaDeviceSynchronize();

	cudaError error = cudaGetLastError();
	if (error)
		cout << "ERROR: CPFToolsGPU: GetRefFrames: " << cudaGetErrorString(error) << endl;

	//cout << "Before the ref frame" << endl;
	//for (int i = 0; i < p.size(); i++)
	//{
	//	cout << RefFrames[(i * 4)].x << ", " << RefFrames[(i * 4)].y << ", " << RefFrames[(i * 4)].z << ", " << RefFrames[(i * 4)].w << endl;
	//	cout << RefFrames[(i * 4) + 1].x << ", " << RefFrames[(i * 4) + 1].y << ", " << RefFrames[(i * 4) + 1].z << ", " << RefFrames[(i * 4) + 1].w << endl;
	//	cout << RefFrames[(i * 4) + 2].x << ", " << RefFrames[(i * 4) + 2].y << ", " << RefFrames[(i * 4) + 2].z << ", " << RefFrames[(i * 4) + 2].w << endl;
	//	cout << RefFrames[(i * 4) + 3].x << ", " << RefFrames[(i * 4) + 3].y << ", " << RefFrames[(i * 4) + 3].z << ", " << RefFrames[(i * 4) + 3].w << endl;
	//}

	pointerToVecM4F(dst, RefFrames, p.size());
}


__global__
void DiscretizeCurvatureGPU(double2* dst, float3* n1, float3* n, Matches* matches, int num_pts, double range, int iteration) 
{
	int i = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (i > num_pts) return;
	if (matches[i].matches[iteration].distance > 0.0)
	{
		double prevAng = dst[i].x;
		float prevNum = dst[i].y;



		int id = matches[i].matches[iteration].second;
		//printf("%f \n", AngleBetweenGPU(n1[i], n[id]));
		double curAng = AngleBetweenGPU(n1[i], n[id]) * range;

		dst[i].x = prevAng + curAng; 
		dst[i].y = prevNum + 1;
	}
}

__global__
void CalculateDiscCurve(int* dst, double2* src, int num_curve)
{
	int i = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (i > num_curve) return;
	dst[i] = src[i].x / src[i].y;
}

//static
void CPFToolsGPU::DiscretizeCurvature(vector<uint32_t>& dst, const vector<Eigen::Vector3f>& n1, PointCloud& pc, vector<Matches> matches, const float range)
{
	vecToPointer3F(vectorsA, n1);
	vecToPointer3F(pcN, pc.normals);
	for (int i = 0; i < matches.size(); i++)
	{
		pt_matches[i] = matches.at(i);
	}
	for (int i = 0; i < n1.size(); i++)
		curvature_pairs[i] = make_double2(0, 0);

	int threads = 64;
	int blocks = ceil((float) pc.normals.size() / threads);

	//Iterate through all 21 matches in each match
	for (int i = 0; i < 21; i++)
	{
		//Store number of valid matches and the sum of the angle between their normals
		DiscretizeCurvatureGPU<<<blocks, threads>>>(curvature_pairs, vectorsA, pcN, pt_matches, pc.normals.size(), range, i);
		cudaDeviceSynchronize();
	}

	cudaError error = cudaGetLastError();
	if (error)
		cout << "ERROR: CPFToolsGPU: DiscretizeCurvature: DiscretizeCurvatureGPU: " << error << endl;

	//curvature_pairs.x / curvature_pairs.y
	CalculateDiscCurve<<<blocks, threads>>>(discretized_curvatures, curvature_pairs, pc.normals.size());
	cudaDeviceSynchronize();
	error = cudaGetLastError();
	if (error)
		cout << "ERROR: CPFToolsGPU: DiscretizeCurvature: CalculateDiscCurve: " << error << endl;

	pointerToVecI(dst, discretized_curvatures, n1.size());
}

__global__
void DiscretizeCPFGPU(CPFDiscreet* dst, uint32_t* curvatures, float4* ref_frames, float3* pts, int num_pts, Matches* matches, int iteration, float* max_angle_val, float* min_angle_val, int* ang_bins)
{
	int i = (blockIdx.x * blockDim.x) + threadIdx.x;

	if (i > num_pts)
		return;

	if (matches[i].matches[iteration].distance > 0.0) {
		int id = matches[i].matches[iteration].second;
		int cur1 = curvatures[i];
		int cur2 = curvatures[id];
		int idx = (i * KNN_MATCHES_LENGTH) + iteration;

		float3 p01;

		float3 pt = pts[id];
		float4* ref_frame = ref_frames + (i * 4);

		dst[idx].point_idx = i;
		//get the angle.
		//The point pt is in the frame origin.  n is aligned with the x axis.
		dst[idx].alpha = atan2f(-pt.z, pt.y);

		float3 pt_trans = make_float3(
			(ref_frame[0].x * pt.x) + (ref_frame[0].y * pt.y) + (ref_frame[0].z * pt.z) + ref_frame[0].w,
			(ref_frame[1].x * pt.x) + (ref_frame[1].y * pt.y) + (ref_frame[1].z * pt.z) + ref_frame[1].w, 
			(ref_frame[2].x * pt.x) + (ref_frame[2].y * pt.y) + (ref_frame[2].z * pt.z) + ref_frame[2].w);



		pt = pts[i];

		float pn = sqrt(powf(pt.x, 2) + powf(pt.y, 2) + powf(pt.z, 2));
		float3 p_norm = make_float3(pt.x / pn, pt.y / pn, pt.z / pn);

		float ptrn = sqrt(powf(pt_trans.x, 2) + powf(pt_trans.y, 2) + powf(pt_trans.z, 2));
		float3 ptr_norm = make_float3(pt_trans.x / ptrn, pt_trans.y / ptrn, pt_trans.z / ptrn);

		float ang = (p_norm.x * ptr_norm.x) + (p_norm.y * ptr_norm.y) + (p_norm.z * ptr_norm.z);

		if (isnan(ang)) ang = 0;

		dst[idx].data[0] = cur1;
		dst[idx].data[1] = cur2;
		dst[idx].data[2] = ((double)(ang + 1.0) * (double)(*ang_bins / 2.0));
		dst[idx].data[3] = 0; //cur1 - cur2

		if (ang > *max_angle_val)
			*max_angle_val = ang;
		if (ang < *min_angle_val)
			*min_angle_val = ang;
		return;
	}
}

//static
void CPFToolsGPU::DiscretizeCPF(vector<CPFDiscreet>& dst, vector<uint32_t>& curvatures, vector<Matches> matches, vector<Eigen::Vector3f> pts, vector<Eigen::Affine3f> ref_frames)
{
	vecToPointerI(discretized_curvatures, curvatures);
	vecToPointer3F(vectorsA, pts);
	vecToPointerM4F(RefFrames, ref_frames);
	for (int i = 0; i < matches.size(); i++)
	{
		pt_matches[i] = matches.at(i);
		discretized_cpfs[i] = CPFDiscreet();
	}

	int threads = 64;
	int blocks = ceil((float) pts.size() / threads);
	for (int i = 0; i < KNN_MATCHES_LENGTH; i++)
	{
		DiscretizeCPFGPU<<<blocks, threads>>>(discretized_cpfs, (uint32_t*)discretized_curvatures, RefFrames, vectorsA, pts.size(), pt_matches, i, max_ang_value, min_ang_value, angle_bins);
		cudaError error = cudaGetLastError();
		if (error)
			cout << "ERROR: CPFToolsGPU: DiscretizeCPF: " << cudaGetErrorString(error) << " on iteration " << i << endl;
		cudaDeviceSynchronize();
	}

	CPFDiscreet null_CPF = CPFDiscreet();
	for (int i = 0; i < pts.size() * KNN_MATCHES_LENGTH; i++)
	{
		if(!(discretized_cpfs[i] == null_CPF))
			dst.push_back(discretized_cpfs[i]);
	}
}