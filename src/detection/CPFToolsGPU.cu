#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda/cutil_math.h>
#include <cuda/cutil_matrix.h>
#include "CPFToolsGPU.h"

#define M_PI 3.14159265359

namespace nsCPFToolsGPU {

	float3* vectorsA;
	float3* vectorsB;
	int angle_bins = 12;

	float4* RefFrames;

}

using namespace texpert;
using namespace nsCPFToolsGPU;

void vecToPointerF(float3* dst, vector<Eigen::Vector3f>& src)
{
	Eigen::Vector3f curVec;
	for (int i = 0; i < src.size(); i++) {
		curVec = src.at(i);
		dst[i] = make_float3(curVec(0), curVec(1), curVec(2));
	}
}



//static
__host__ __device__
float AngleBetween(const float3 a, const float3 b)
{
	float3 a_norm = normalize(a);
	float3 b_norm = normalize(b);
	float3 c = cross(a, b);
	return atan2f(sqrtf(powf(c.x, 2) + powf(c.y, 2) + powf(c.z, 2)), dot(a_norm, b_norm));
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
	float theta =	AngleBetween(normal, make_float3(1, 0, 0)); //Angle between the surface normal and the x axis

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
Eigen::Affine3f CPFToolsGPU::GetRefFrame(vector<Eigen::Vector3f>& p, vector<Eigen::Vector3f>& n)
{
	//Map Eigen vectors to float3 pointers
	vecToPointerF(vectorsA, p);
	vecToPointerF(vectorsB, n);


}

