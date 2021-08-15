#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda/cutil_math.h>
#include "CPFToolsGPU.h"
#include <chrono>

namespace nsCPFToolsGPU {
	int angle_bins = 12;
	float* max_ang_value;
	float* min_ang_value;

	int* pc_size;
	float3* pcN;
	float3* pcP;
	Matches* pt_matches;

	//GetReferenceFrames
	float4* RefFrames;

	//DiscretizeCurvature
	double2* curvature_pairs;
	int* discretized_curvatures;

	//DiscretizeCPF
	CPFDiscreet* discretized_cpfs;

	//Times for testing
	std::chrono::steady_clock::time_point start;
	std::chrono::steady_clock::time_point stop;
}

using namespace texpert;
using namespace nsCPFToolsGPU;

/*
---------------Memory Allocation Functions-------------------
*/

void CPFToolsGPU::AllocateMemory(uint32_t size)
{
	cudaMallocManaged(&pc_size, sizeof(int));
	cudaMallocManaged(&max_ang_value, sizeof(int));
	cudaMallocManaged(&min_ang_value, sizeof(int));
	*pc_size = size;
	*max_ang_value = 0.0;
	*min_ang_value = 10000000.0;

	cudaMallocManaged(&pcN, size * sizeof(float3));
	cudaMallocManaged(&pcP, size * sizeof(float3));
	cudaMallocManaged(&pt_matches, size * sizeof(Matches));

	cudaMallocManaged(&RefFrames, size * 4 * sizeof(float4));

	cudaMallocManaged(&curvature_pairs, size * 21 * sizeof(double2));
	cudaMallocManaged(&discretized_curvatures, size * sizeof(int));
	
	cudaMallocManaged(&discretized_cpfs, size * KNN_MATCHES_LENGTH * sizeof(CPFDiscreet));

	cudaError error = cudaGetLastError();
	if (error)
		cout << "ERROR: AllocateMemory: " << cudaGetErrorString(error) << endl;
}

void CPFToolsGPU::DeallocateMemory()
{
	cudaFree(max_ang_value);
	cudaFree(min_ang_value);

	cudaFree(pcN);
	cudaFree(pt_matches);

	cudaFree(RefFrames);

	cudaFree(curvature_pairs);
	cudaFree(discretized_curvatures);

	cudaFree(discretized_cpfs);

	cudaError error = cudaGetLastError();
	if (error)
		cout << "ERROR: AllocateMemory: " << cudaGetErrorString(error) << endl;
}

/*
--------------CPU-GPU Translation Functions-----------------
*/
void vecToPointer3F(float3* dst, const vector<Eigen::Vector3f>& src)
{
	cudaMemcpy(dst, src.data(), src.size() * sizeof(float3), cudaMemcpyHostToDevice);
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
	cudaMemcpy(dst, src.data(), src.size() * sizeof(int), cudaMemcpyDeviceToHost);
}

void pointerToVecI(std::vector<uint32_t>& dst, int* src, int num_e)
{
	dst.resize(num_e);
	cudaMemcpy(dst.data(), src, num_e * sizeof(int), cudaMemcpyDeviceToHost);
}

void CPFToolsGPU::AssignPointCloud(PointCloud& pc)
{
	vecToPointer3F(pcN, pc.normals);
	vecToPointer3F(pcP, pc.points);
}

void CPFToolsGPU::AssignMatches(vector<Matches>& matches)
{
	cudaMemcpy(pt_matches, matches.data(), matches.size() * sizeof(Matches), cudaMemcpyHostToDevice);
}

/*
-----------------Main Class Functions--------------------
*/

//static
__host__ __device__
float AngleBetweenGPU(float3 a, float3 b)
{
	if (a.x == b.x && a.y == b.y && a.z == b.z)
		return 0.0f;

	float an = sqrtf(powf(a.x, 2) + powf(a.y, 2) + powf(a.z, 2));
	float bn = sqrtf(powf(b.x, 2) + powf(b.y, 2) + powf(b.z, 2));
	float3 a_norm;
	float3 b_norm;

	if (an == 0)
		a_norm = a;
	else
		a_norm = make_float3(a.x / an, a.y / an, a.z / an);

	if (bn == 0)
		b_norm = b;
	else
		b_norm = make_float3(b.x / bn, b.y / bn, b.z / bn);

	float3 c = cross(a_norm, b_norm);


	double c_norm = sqrt(powf(c.x, 2) + powf(c.y, 2) + powf(c.z, 2));
	double ab_dot = (a_norm.x * b_norm.x) + (a_norm.y * b_norm.y) + (a_norm.z * b_norm.z);

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
	float3 rot1 = make_float3(cost + (powf(axis.x, 2) * omcost), (xy * omcost) - (axis.z * sint), (xz * omcost) + (axis.y * sint));
	float3 rot2 = make_float3((xy * omcost) + (axis.z * sint), cost + (powf(axis.y, 2) * omcost), (yz * omcost) - (axis.x * sint));
	float3 rot3 = make_float3((xz * omcost) - (axis.y * sint), (yz * omcost) + (axis.x * sint), cost + (powf(axis.z, 2) * omcost));

	//Translate the point to fit in the new rotation frame
	float3 tinf = make_float3(rot1.x * point.x + rot1.y * point.y + rot1.z * point.z,
		rot2.x * point.x + rot2.y * point.y + rot2.z * point.z,
		rot3.x * point.x + rot3.y * point.y + rot3.z * point.z);

	// create the reference frame
	res[i * 4] =		make_float4(rot1.x, rot1.y, rot1.z, tinf.x);
	res[(i * 4) + 1] =  make_float4(rot2.x, rot2.y, rot2.z, tinf.y);
	res[(i * 4) + 2] =  make_float4(rot3.x, rot3.y, rot3.z, tinf.z);
	res[(i * 4) + 3] =  make_float4(0, 0, 0, 1);
}

//static
void CPFToolsGPU::GetRefFrames(vector<Eigen::Affine3f>& dst, vector<Eigen::Vector3f>& p, vector<Eigen::Vector3f>& n)
{
	//Get reference frames
	int threads = 32;
	int blocks = ceil((float) p.size() / threads);
	GetRefFrameGPU<<<blocks, threads>>>(pcP, pcN, p.size(), RefFrames);
	cudaDeviceSynchronize();

	cudaError error = cudaGetLastError();
	if (error)
		cout << "ERROR: CPFToolsGPU: GetRefFrames: " << cudaGetErrorString(error) << endl;
}


__global__
void DiscretizeCurvatureGPU(double2* dst, float3* n, Matches* matches, int num_pts, double range) 
{
	int i = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (i > num_pts * 21) return;

	int ii = i / 21;
	int it = i % 21;
	if (matches[ii].matches[it].distance > 0.0)
	{
		int id = matches[ii].matches[it].second;
		double curAng = AngleBetweenGPU(n[ii], n[id]) * range;

		dst[i].x = curAng; 
		dst[i].y = 1;
	}
	else
	{
		dst[i].x = 0;
		dst[i].y = 0;
	}
}

__global__
void CalculateDiscCurve(int* dst, double2* src, int num_curve)
{
	int i = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (i > num_curve) return;
	double2* begin = src + (i * 21);
	double2 funct;

	funct.x = begin[0].x + begin[1].x + begin[2].x + begin[3].x + begin[4].x + begin[5].x + begin[6].x +
		begin[7].x + begin[8].x + begin[9].x + begin[10].x + begin[11].x + begin[12].x + begin[13].x +
		begin[14].x + begin[15].x + begin[16].x + begin[17].x + begin[18].x + begin[19].x + begin[20].x;

	funct.y = begin[0].y + begin[1].y + begin[2].y + begin[3].y + begin[4].y + begin[5].y + begin[6].y +
		begin[7].y + begin[8].y + begin[9].y + begin[10].y + begin[11].y + begin[12].y + begin[13].y +
		begin[14].y + begin[15].y + begin[16].y + begin[17].y + begin[18].y + begin[19].y + begin[20].y;

	dst[i] = funct.x / funct.y;
}

//static
void CPFToolsGPU::DiscretizeCurvature(vector<uint32_t>& dst, const vector<Eigen::Vector3f>& n1, PointCloud& pc, vector<Matches> matches, const double range)
{
	cudaMemcpy(pt_matches, matches.data(), matches.size() * sizeof(Matches), cudaMemcpyHostToDevice);

	int threads = 64;
	int blocks = ceil(((float) pc.normals.size() * 21.0f) / threads);

	//Store number of valid matches and the sum of the angle between their normals
	DiscretizeCurvatureGPU<<<blocks, threads>>>(curvature_pairs, pcN, pt_matches, pc.normals.size(), range);
	cudaDeviceSynchronize();

	cudaError error = cudaGetLastError();
	if (error)
		cout << "ERROR: CPFToolsGPU: DiscretizeCurvature: DiscretizeCurvatureGPU: " << cudaGetErrorString(error) << endl;

	//curvature_pairs.x / curvature_pairs.y
	blocks = ceil((float)pc.normals.size() / threads);
	CalculateDiscCurve<<<blocks, threads>>>(discretized_curvatures, curvature_pairs, pc.normals.size());
	cudaDeviceSynchronize();
	error = cudaGetLastError();
	if (error)
		cout << "ERROR: CPFToolsGPU: DiscretizeCurvature: CalculateDiscCurve: " << cudaGetErrorString(error) << endl;

	pointerToVecI(dst, discretized_curvatures, n1.size());
}

__global__
void DiscretizeCPFGPU(CPFDiscreet* dst, uint32_t* curvatures, float4* ref_frames, float3* pts, int num_pts, Matches* matches, float* max_angle_val, float* min_angle_val, int ang_bins)
{
	int i = (blockIdx.x * blockDim.x) + threadIdx.x;

	if (i > num_pts * KNN_MATCHES_LENGTH)
		return;

	int it = i % KNN_MATCHES_LENGTH;
	int ii = i / KNN_MATCHES_LENGTH;

	if (matches[ii].matches[it].distance > 0.0) {
		int id = matches[ii].matches[it].second;
		int cur1 = curvatures[ii];
		int cur2 = curvatures[id];

		float3 pt = pts[id];
		float4* ref_frame = ref_frames + (ii * 4);

		double3 ptd = make_double3(pt.x, pt.y, pt.z);

		dst[i].point_idx = ii;


		double3 pt_trans = make_double3(
			(ref_frame[0].x * pt.x) + (ref_frame[0].y * pt.y) + (ref_frame[0].z * pt.z) + ref_frame[0].w,
			(ref_frame[1].x * pt.x) + (ref_frame[1].y * pt.y) + (ref_frame[1].z * pt.z) + ref_frame[1].w, 
			(ref_frame[2].x * pt.x) + (ref_frame[2].y * pt.y) + (ref_frame[2].z * pt.z) + ref_frame[2].w);

		//get the angle.
		//The point pt is in the frame origin.  n is aligned with the x axis.
		dst[i].alpha = atan2((double)-pt_trans.z, (double)pt_trans.y);



		pt = pts[ii];

		double pn = sqrt(powf(pt.x, 2) + powf(pt.y, 2) + powf(pt.z, 2));
		double3 p_norm;
		if (pn == 0)
			p_norm = make_double3(pt.x, pt.y, pt.z);
		else
			p_norm = make_double3(pt.x / pn, pt.y / pn, pt.z / pn);

		double ptrn = sqrt(powf(pt_trans.x, 2) + powf(pt_trans.y, 2) + powf(pt_trans.z, 2));
		double3 ptr_norm;
		if (ptrn == 0)
			ptr_norm = make_double3(pt_trans.x, pt_trans.y, pt_trans.z);
		else
			ptr_norm = make_double3(pt_trans.x / ptrn, pt_trans.y / ptrn, pt_trans.z / ptrn);

		double ang = (p_norm.x * ptr_norm.x) + (p_norm.y * ptr_norm.y) + (p_norm.z * ptr_norm.z);

		dst[i].data[0] = cur1;
		dst[i].data[1] = cur2;
		if (ii == id)
			dst[i].data[2] = (double)ang_bins / 2.0;
		else
			dst[i].data[2] = ((ang + 1.0) * ((double)ang_bins / 2.0));
		dst[i].data[3] = 0; //cur1 - cur2

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
	//For each point, find the CPFDiscreet for each match
	int threads = 128;
	int blocks = ceil((float) pts.size() * KNN_MATCHES_LENGTH / threads);
	DiscretizeCPFGPU<<<blocks, threads>>>(discretized_cpfs, (uint32_t*)discretized_curvatures, RefFrames, pcP, pts.size(), pt_matches, max_ang_value, min_ang_value, angle_bins);
	cudaDeviceSynchronize();

	cudaError error = cudaGetLastError();
	if (error)
		cout << "ERROR: CPFToolsGPU: DiscretizeCPF: " << cudaGetErrorString(error) << endl;

	//Push back all valid points
	dst.resize(pts.size() * KNN_MATCHES_LENGTH);
	cudaMemcpy(dst.data(), discretized_cpfs, pts.size() * KNN_MATCHES_LENGTH * sizeof(CPFDiscreet), cudaMemcpyDeviceToHost);
	error = cudaGetLastError();
	if (error)
		cout << "ERROR: CPFToolsGPU: MemCpy: " << cudaGetErrorString(error) << endl;
}

//static 
void CPFToolsGPU::GetMaxMinAng(float& max, float& min)
{
	max = *max_ang_value;
	min = *min_ang_value;
}

//static 
void CPFToolsGPU::Reset(void)
{
	*max_ang_value = 0.0;
	*min_ang_value = 10000000.0;
}

//static 
void CPFToolsGPU::SetParam(CPFParamGPU& param)
{
	angle_bins = param.angle_bins;
}