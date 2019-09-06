#include "NoiseFilter.h"




/*
Add Gaussian noise to a point cloud.  
*/
//static 
int NoiseFilter::ApplyGaussianNoise(PointCloud& src, PointCloud& dst, GausianParams param, bool verbose)
{
	if (src.size() == 0)
	{
		cout << "[NoiseFilter] Error - the point cloud contains no points." << endl;
		return -1;
	}

    int size = src.size();
    dst.resize(size);

    //--------------
    // find min and max values 
    float maxX =  std::numeric_limits<float>::min();
    float maxY =  std::numeric_limits<float>::min();
    float maxZ =  std::numeric_limits<float>::min();
    float minX =  std::numeric_limits<float>::max();
    float minY =  std::numeric_limits<float>::max();
    float minZ =  std::numeric_limits<float>::max();

    for( auto p : src.points){
       if(p.x() > maxX) maxX = p.x();
       if(p.x() < minX) minX = p.x();
       if(p.y() > maxY) maxY = p.y();
       if(p.y() < minY) minY = p.y();
       if(p.z() > maxZ) maxZ = p.z();
       if(p.z() < minZ) minZ = p.z();
    }

    float dimX  = (maxX - minX)/2.0;
    float dimY  = (maxY - minY)/2.0;
    float dimZ  = (maxZ - minZ)/2.0;
    

    std::default_random_engine generator;
    std::uniform_real_distribution<double> distributionX(-dimX,dimX);
    std::uniform_real_distribution<double> distributionY(-dimY,dimY);
    std::uniform_real_distribution<double> distributionZ(-dimX,dimZ);


    double c = 1 / (param.sigma * std::sqrt(2*3.1415) );

    int count = 0;

    vector<Eigen::Vector3f> p = src.points;
    vector<Eigen::Vector3f> n = src.normals;
    for( int i=0; i<size; i++)
    {
        double valuex = distributionX(generator);
        double valuey = distributionY(generator);
        double valuez = distributionZ(generator);

        double x = pow(double(p[i].x() - param.mean), 2.0) / (2.0 * param.mean * param.mean);
        double y = pow(double(p[i].y() - param.mean), 2.0) / (2.0 * param.mean * param.mean);
        double z = pow(double(p[i].z() - param.mean), 2.0) / (2.0 * param.mean * param.mean);

        dst.points[i].x() =  p[i].x() + c * exp( -x);
        dst.points[i].y() =  p[i].y() + c * exp( -y);
        dst.points[i].z() =  p[i].z() + c * exp( -z);

        dst.points[i].x() =  p[i].x() + valuex * param.sigma;
        dst.points[i].y() =  p[i].y() + valuey * param.sigma;
        dst.points[i].z() =  p[i].z() + valuez * param.sigma;

        dst.normals[i] =  n[i];
        count++;
    }

	return count;

}
