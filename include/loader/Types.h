/*
class Types

Rafael Radkowski
Iowa State University
rafael@iastate.edu
MIT License
---------------------------------------------------------------

Last edits:

June 3, 2020, RR
- Added a vector to store the centroid of the point cloud. 
*/

#ifndef __TYPES__
#define __TYPES__

#include <vector>
#include <iostream>
#include <string>
#include <fstream>

// eigen
#include <Eigen/Dense>


using namespace std;

typedef struct PointCloud
{
    vector<Eigen::Vector3f>     points;
    vector<Eigen::Vector3f>     normals;

    int                         N; // size
    int                         id; // index

    Eigen::Matrix4f             pose;

	Eigen::Vector3f				centroid0; // the centroid of the object as loaded or as initially added. 

    PointCloud(){
        N = 0;
        id = 0;
        pose << 1,0,0,0,
                0,1,0,0,
                0,0,1,0,
                0,0,0,1;

		centroid0 << 0.0, 0.0, 0.0;
    }

    int size(void){
        N = (int)points.size();
        return N;
    }

    void resize(int size)
    {
        if(size <0 && size == N ) return;
        points.resize(size);
        normals.resize(size);
    }

}PointCloud;



#endif