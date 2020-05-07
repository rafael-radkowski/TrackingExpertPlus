#pragma once

// stl
#include <iostream>
#include <string>
#include <vector>


// Eigen
#include <Eigen\Dense>
#include <Eigen/Geometry>



class MatrixTransform
{
public:

	static Eigen::Matrix4f CreateAffineMatrix(const Eigen::Vector3f translation, const Eigen::Vector3f rotation);


	

};