#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda/cutil_math.h>
#include <cuda/cutil_matrix.h>
#include "CPFToolsGPU.h"

#define M_PI 3.14159265359

namespace nsCPFTools {

	float3* vectorsA;
	float3* vectorsB;
	int angle_bins = 12;

}

using namespace texpert;
using namespace nsCPFTools;

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
void GetRefFrame()
{

}

void vecToPointerF(float3& dst, vector<Eigen::Vector3f>& src)
{

}