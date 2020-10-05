#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda/cutil_math.h>
#include <cuda/cutil_matrix.h>
#include "CPFToolsGPU.h"

#define M_PI 3.14159265359

namespace nsCPFToolsGPU {
	int angle_bins = 12;

	float3* vectorsA;
	float3* vectorsB;
	float3* pcN;
	Matches* pt_matches;
	int* int_p;

	//GetReferenceFrames
	float4* RefFrames;

	//DiscretizeCurvature
	float2* curvature_pairs;
	int* discretized_curvatures;

	//DiscretizeCPF
	CPFDiscreet discretized_cpfs;
}

using namespace texpert;
using namespace nsCPFToolsGPU;

void vecToPointer3F(float3* dst, const vector<Eigen::Vector3f>& src)
{
	Eigen::Vector3f curVec;
	for (int i = 0; i < src.size(); i++) {
		curVec = src.at(i);
		dst[i] = make_float3(curVec(0), curVec(1), curVec(2));
	}
}

void pointerToVecM4F(vector<Eigen::Affine3f>& dst, float4* src)
{
	Eigen::Matrix4f curMatrix;
	for (int i = 0; i < dst.size(); i++)
	{
		curMatrix <<
			(src[i * 4].x, src[i * 4].y, src[i * 4].z, src[i * 4].w,
				src[(i * 4) + 1].x, src[(i * 4) + 1].y, src[(i * 4) + 1].z, src[(i * 4) + 1].w,
				src[(i * 4) + 2].x, src[(i * 4) + 2].y, src[(i * 4) + 2].z, src[(i * 4) + 2].w,
				src[(i * 4) + 3].x, src[(i * 4) + 3].y, src[(i * 4) + 3].z, src[(i * 4) + 3].w);

		dst.at(i) = curMatrix;
	}
}

void pointerToVecI(std::vector<uint32_t>& dst, int* src)
{
	for (int i = 0; i < dst.size(); i++)
		dst.at(i) = src[i];
}



//static
__host__ __device__
float AngleBetweenGPU(const float3 a, const float3 b)
{
	float3 a_norm = normalize(a);
	float3 b_norm = normalize(b);
	float3 c = cross(a, b);
	return atan2f(sqrtf(powf(c.x, 2) + powf(c.y, 2) + powf(c.z, 2)), dot(a_norm, b_norm));
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
		axis = normalize(axis);
	}

	// create an angle axis transformation that rotates A degrees around the axis.
	float theta =	AngleBetweenGPU(normal, make_float3(1, 0, 0)); //Angle between the surface normal and the x axis

	float cost =	cosf(theta); // cos(theta)
	float omcost =	1 - cost; // 1 - cos(theta)
	float sint =	sinf(theta); // sin(theta)

	float xy =		axis.x * axis.y; // ux * uy
	float yz =		axis.y * axis.z; // uy * uz
	float xz =		axis.x * axis.z; // ux * uz


	float3* rot = (float3*)malloc(3 * sizeof(float3));
	rot[0] = make_float3(cost + (powf(axis.x, 2) * omcost), (xy * omcost) - (axis.z * sint), (xz * omcost) + (axis.y * sint));
	rot[1] = make_float3((xy * omcost) + (axis.z * sint), cost + (powf(axis.y, 2) * omcost), (yz * omcost) - (axis.x * sint));
	rot[2] = make_float3((xz * omcost) - (axis.y * sint), (yz * omcost) + (axis.x * sint), cost + (powf(axis.z, 2) * omcost));

	// create the reference frame
	res[i * 4] =		make_float4(rot[0], point.x);
	res[(i * 4) + 1] =  make_float4(rot[1], point.y);
	res[(i * 4) + 2] =  make_float4(rot[2], point.z);
	res[(i * 4) + 3] =  make_float4(0, 0, 0, 1);
}

//static
void CPFToolsGPU::GetRefFrames(vector<Eigen::Affine3f>& dst, vector<Eigen::Vector3f>& p, vector<Eigen::Vector3f>& n)
{
	//Map Eigen vectors to float3 pointers
	vecToPointer3F(vectorsA, p);
	vecToPointer3F(vectorsB, n);

	int threads = 32;
	int blocks = p.size() / threads;
	GetRefFrameGPU<<<blocks, threads>>>(vectorsA, vectorsB, p.size(), RefFrames);

	cudaDeviceSynchronize();

	std::vector<Eigen::Affine3f> frames = std::vector<Eigen::Affine3f>(p.size());
	pointerToVecM4F(dst, RefFrames);
}



//TODO
__global__
void DiscretizeCurvatureGPU(float2* dst, float3* n1, float3* n, Matches* matches, int* range, int iteration) 
{
	int i = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (matches[i].matches[iteration].distance > 0.0)
	{
		int id = matches[i].matches[iteration].second;
		dst[i].x += AngleBetweenGPU(n1[i], n[id]);
		dst[i].y++;
	}
}

__global__
void CalculateDiscCurve(int* dst, float2* src)
{
	int i = (blockIdx.x * blockDim.x) + threadIdx.x;
	dst[i] = src[i].x / src[i].y;
}

//static
void CPFToolsGPU::DiscretizeCurvature(vector<uint32_t>& dst, const vector<Eigen::Vector3f>& n1, PointCloud& pc, const Matches* matches, const float range)
{
	vecToPointer3F(vectorsA, n1);
	vecToPointer3F(pcN, pc.normals);
	curvature_pairs = (float2*)malloc(n1.size() * sizeof(float2));
	cudaMemcpy(pt_matches, matches, n1.size() * sizeof(Matches), cudaMemcpyHostToDevice);
	*int_p = range;
	for (int i = 0; i < n1.size(); i++)
		curvature_pairs[i] = make_float2(0, 0);

	int threads = 64;
	int blocks = pc.size() / threads;
	for (int i = 0; i < 21; i++)
	{
		DiscretizeCurvatureGPU<<<blocks, threads>>>(curvature_pairs, vectorsA, pcN, pt_matches, int_p, i);
	}

	CalculateDiscCurve<<<blocks, threads>>>(discretized_curvatures, curvature_pairs);

	cudaDeviceSynchronize();

	pointerToVecI(dst, discretized_curvatures);
}



//TODO
__global__
void DiscretizeCPFGPU(CPFDiscreet* dst, uint32_t* curvatures, Matches* matches, int iteration)
{
	int i = (blockIdx.x * blockDim.x) + threadIdx.x;

	if (matches[i].matches[iteration].distance > 0.0) {
		int id = matches[i].matches[iteration].second;
		int cur1 = curvatures[i];
		int cur2 = curvatures[id];

		float3 pt = ref_frames[i] * pts[id];
	}


}

//TODO
//static
void CPFToolsGPU::DiscretizeCPF(vector<CPFDiscreet>& dst, vector<uint32_t>& curvatures, Matches* matches, int num_matches, vector<Eigen::Vector3f> pts, vector<Eigen::Affine3f> ref_frames)
{
	for (int i = 0; i < num_matches; i++)
	{
		DiscretizeCPFGPU()
	}
}