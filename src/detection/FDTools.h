#pragma once



// stl
#include <vector>
#include <iostream>

// Eigen
#include <Eigen/Dense>
#include <Eigen/Geometry>

// local 
#include "FDTypes.h"

using namespace Eigen;

namespace isu_ar {

	class FDTools
	{
	public:

		/*
		Calculate an x-axis aligned reference frame for points.
		The vector n is aligned with e_x 
		@param p - a point as (x, y, z)
		@param n - the normal vector associated to p. 
		*/
		static Affine3f getRefFrame(Vector3f& p, Vector3f& n);



		/*
		Calculate the angle between two vectors
		@param a - the first vector as vec3
		@param b - the second vector as vec3
		@return - the angle in rad. 
		*/
		static float angleBetween(const Eigen::Vector3f& a, const Eigen::Vector3f& b);


		/*
		Discretize the ppf dataset.
		This function returns a ppf feature descriptors. 
		Read 
		Bertram Drost et al., Model Globally, Match Locally: Efficient and Robust 3D Object Recognition
		http://campar.in.tum.de/pub/drost2010CVPR/drost2010CVPR.pdf
		for details.
		@param p1 - the first point as vec3 (x, y, z)
		@param n1 - the associated normal vector.
		@param p2 - the second point as vec3 (x, y, z)
		@param n2 - the  normal vector associated to p2.
		@param distance_step - the distance to discretize the delta between p1 and p2 -> ||p1 - p2|| / distance_step.
		@param angle_step - the angle step to discretize the anlges between normal vectors, e.g. ang(n1, n2) / angle_step
		@return a PPF feature as type PPFDiscreet.
		*/
		static PPFDiscreet DiscretizePPF(const Vector3f& p1, const Vector3f& n1, const Vector3f& p2, const Vector3f& n2,
										 const float distance_step,
										 const float angle_step);

	private:





	};

}