#include "Sampling.h"


namespace Sampling_ns{

    SamplingParam   curr_param;

    SamplingMethod  curr_method;


    bool    g_verbose = false;
	int		g_verbose_level = 0;
}


using namespace Sampling_ns;
using namespace texpert;

/*
Sample the point cloud with a uniform sampling filter. 
This filter is voxel based and will put one point into a vocel 
@param src - location of the the source point cloud.
@param dst - location of the the destination  point cloud.
@param param - the sampling parameters
*/
//static 
void Sampling::Uniform( PointCloud& src, PointCloud& dst, SamplingParam param)
{
	// make a local copy in case src and dst point to the same memory
	PointCloud temp_src = src;

    int size = src.size();
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

    if(g_verbose && g_verbose_level == 2){
        cout << "[INFO] Sampling - Min x: " << minX << ", max x: " << maxX << endl;
        cout << "[INFO] Sampling - Min y: " << minY << ", max y: " << maxY << endl;
        cout << "[INFO] Sampling - Min z: " << minZ << ", max z: " << maxZ << endl;
    }

    //--------------
    // calculate the number of voxels for the given grid size. 
    float dimX  = maxX - minX;
    float dimY  = maxY - minY;
    float dimZ  = maxZ - minZ;
    float voxX = param.grid_x;
    float voxY = param.grid_y;
    float voxZ = param.grid_z;

    // size check
    if(voxX  > dimX ) voxX = dimX;
    if(voxY  > dimY ) voxY = dimY;
    if(voxZ  > dimZ ) voxZ = dimZ;

    // number of cells
    int vx = (int)std::ceil ( dimX / voxX );
    int vy = (int)std::ceil ( dimY / voxY );
    int vz = (int)std::ceil ( dimZ / voxZ );

    if(g_verbose  && g_verbose_level == 2){
        cout << "[INFO] Sampling - Num cells Vx: " << vx << endl;
        cout << "[INFO] Sampling - Num cells Vy: " << vy << endl;
        cout << "[INFO] Sampling - Num cells Vz: " << vz << endl;
    }


    //--------------
    // Create the cube. It stores indices of points. -1 means no point (empty)
    vector<vector<vector<int>>> cube(vx+1, vector<vector<int> > (vy+1, vector<int> (vz+1, -1)));

	// for min is negative
    float offset_x = dimX - maxX;
    float offset_y = dimY - maxY;
    float offset_z = dimZ - maxZ ;


    int count = 0;
    vector<Eigen::Vector3f> p_ = src.points;
    for( int i=0; i<size; i++){
		float v = std::max((p_[i].x() + offset_x), 0.0f);
        int idx = (int)ceil((p_[i].x() + offset_x) / voxX );
        int idy = (int)ceil((p_[i].y() + offset_y) / voxY );
        int idz = (int)ceil((p_[i].z() + offset_z) / voxZ );

        if(cube[idx][idy][idz] == -1)
        {
            cube[idx][idy][idz] = i;
            count++;
        }
    }

    if(g_verbose && g_verbose_level == 2){
        cout << "[INFO] - Downsampled fr0m " << src.N << " to " << count << " points. " << endl;
    }
	dst.points.clear();
	dst.normals.clear();
    dst.points.resize(count);
    dst.normals.resize(count);

    int i = 0;
    vector<vector<vector<int>>>::iterator itr_x;
    vector<vector<int>>::iterator itr_y;
    vector<int>::iterator itr_z;

    // Copy back.
    // ToDo: That is computational expensive. Is there a better way?
    for(itr_x = cube.begin(); itr_x != cube.end(); itr_x++ ){
        for(itr_y = itr_x->begin(); itr_y != itr_x->end(); itr_y++ ){
            for(itr_z = itr_y->begin(); itr_z != itr_y->end(); itr_z++ ){
                int idx = (*itr_z);
                if( idx != -1 ){
                    dst.points[i] =  temp_src.points[idx];
                    dst.normals[i] =  temp_src.normals[idx];
                    i++;
                }
            }
        }
    }

	// to reset the variable N
	dst.size();

    if(g_verbose && g_verbose_level == 1){ 
        cout << "[INFO] Sampling - Sampling successfull; output contains " << i << " points and normals. "  << endl;
    }
}



/*
Set the sampling method along with the sampling parameters
@param method - the sampling method. Can be RAW, UNIFORM, and RANDOM
@param param - sampling parameters of type SamplingParam. The parameter must belong to
the set SamplingMethod.
*/
//static 
void Sampling::SetMethod(SamplingMethod method, SamplingParam param)
{
	if(g_verbose && g_verbose_level == 0){ 
       switch(method){
		 case RAW:
			std::cout << "[INFO] Sampling - Set method to Raw"   << std::endl;
            break;
        case UNIFORM:
            std::cout << "[INFO] Sampling - Set method to UNIFORM"   << std::endl;
            break;
        case RANDOM:
			std::cout << "[ERROR] Sampling - Set method to RANDOM - METHOD CURRENTLY NOT SUPPORTED."   << std::endl;
            break;
        default:
            break;
		}
	}
    curr_method = method;
    curr_param = param;
}

/*
Start the sampling procedure
@param src - location of the the source point cloud.
@param dst - location of the the destination  point cloud.
*/
//static 
void Sampling::Run(PointCloud& src, PointCloud& dst, bool verbose)
{
    g_verbose = verbose;
    switch(curr_method){
        case RAW:
            break;
        case UNIFORM:
            Uniform(src, dst, curr_param);
            break;
        case RANDOM:
            break;
        default:
            break;
    }

	dst.size(); // calculates its size

}



/*!
Enable more output information.
@param verbose - true enables output information. 
@param level - set the verbose level. It changes the amount of information.
	level 0: essential information and parameter changes, 
	level 1: additional warnings
	level 2: frame-by-frame information. 
*/
//static 
void Sampling::SetVerbose(bool verbose, int level)
{
	g_verbose = verbose;
	g_verbose_level = level;
}