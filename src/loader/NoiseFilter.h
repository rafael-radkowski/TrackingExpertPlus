/*
class NoiseFilter

Rafael Radkowski
Iowa State University
rafael@iastate.edu
MIT License
---------------------------------------------------------------
*/

#ifndef __GAUSSIAN_NOISE__
#define __GAUSSIAN_NOISE__


#include <vector>
#include <iostream>
#include <string>
#include <fstream>
#include <random>
#ifdef __WIN32
	#include <conio.h>
#endif

// eigen
#include <Eigen/Dense>

// local
#include "Types.h"

using namespace std;


namespace texpert {

class NoiseFilter
{
public:

    /*
    */
    typedef struct _GaussianParams
    {
        double sigma;
        double mean;

        _GaussianParams()
        {
            sigma = 0.01;
            mean = 0.0;
        }

    }GausianParams;


    /*
    Add Gaussian noise to a point cloud.  
    */
    static int ApplyGaussianNoise(PointCloud& src, PointCloud& dst, GausianParams param, bool verbose = false);




};

} //texpert 
#endif