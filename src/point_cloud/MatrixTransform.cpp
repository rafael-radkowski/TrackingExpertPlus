#include "MatrixTransform.h"



using namespace Eigen;


//static 
Eigen::Matrix4f MatrixTransform::CreateAffineMatrix(const Eigen::Vector3f translation, const Eigen::Vector3f rotation)
{
	const float deg2rad = 3.141592653589793238463 / 180.0;

	Transform<float, 3, Eigen::Affine> t;
	t = Translation<float, 3>(Eigen::Vector3f(translation));
	t.rotate(AngleAxis<float>(deg2rad * rotation.x(), Vector3f::UnitX()));
	t.rotate(AngleAxis<float>(deg2rad * rotation.y(), Vector3f::UnitY()));
	t.rotate(AngleAxis<float>(deg2rad * rotation.z(), Vector3f::UnitZ()));
  
   
  
	//t = Translation<float, 3>(translation);
	//t.translate(translation);
    
	return t.matrix();
}



