#include "MatrixUtils.h"


using namespace texpert;

/*
Converts an Eigen::AAffine3f transformation to a glm4 matrix
Matrices are in column major order.
@param matrix - affine matrix of type Affine3f (4x4).
@return glm mat4 matrix. 
*/
//static 
glm::mat4 MatrixUtils::Affine3f2Mat4(Eigen::Affine3f& matrix)
{
	glm::mat4 m;
	for (int i = 0; i < 16; i++) {
		m[i/4][i%4] =  matrix.data()[i];
	}
	return m;
}


/*
Print an affine3 Eigen matrix.
@param matrix - the matrix in column-major order
*/
//static 
void MatrixUtils::PrintAffine3f(Eigen::Affine3f& matrix)
{
	for (int i = 0; i < 4; i++) {
		for (int j = 0; j < 4; j++) {
			cout << matrix.data()[i * 4 + j] << ", ";
		}
		cout << "\n";
	}
	cout << "\n";
}


/*
Print an a glm::mat4  matrix.
@param matrix - the matrix in column-major order
*/
//static 
void MatrixUtils::PrintGlm4(glm::mat4& matrix)
{
	for (int i = 0; i < 4; i++) {
		for (int j = 0; j < 4; j++) {
			cout << matrix[i][j] << ", ";
		}
		cout << "\n";
	}
	cout << "\n";
}
