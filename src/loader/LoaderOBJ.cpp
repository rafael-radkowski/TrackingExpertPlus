#include "LoaderOBJ.h"




//Helper functions
//static 
bool LoaderObj::is_number(std::string s)
{
    return !s.empty() && s.find_first_not_of("0123456789.-") == std::string::npos;
}


/*!
Load a point cloud .obj object from file
@param file = The file
@param loadedNormals = The output location of the loaded normals
@return cloud = The output of the loaded point cloud object
*/
//static 
bool LoaderObj::Read(string file, vector<Eigen::Vector3f >* dst_points, vector<Eigen::Vector3f >* dst_normals, bool invert_z,  bool verbose)
{
    dst_points->clear();
    dst_normals->clear();

    std::ifstream infile(file);

    if(!infile.is_open()){
        #ifdef __WIN32
        _cprintf("[LoaderObj] - Error: could not open file %s.\n", file.c_str())
        #else
         cout << "[LoaderObj] - Error: could not open file " <<  file.c_str() << "." << endl;
        #endif
        return false;
    }

    int count=0;
    while(infile)
    {
        string str;
        infile >> str;
        if(str.compare("vn") == 0){
            count++;
            string nx, ny, nz;
            infile >> nx; infile >> ny; infile >> nz;
            // remove werid meshlab things and invalid data
            if(is_number(nx) && is_number(ny) && is_number(nz)){
                double dnx=std::stod(nx);
                double dny=std::stod(ny);
                double dnz=std::stod(nz);
                if (invert_z) dnz = -dnz;
                Eigen::Vector3f v((float)dnx,(float)dny,(float)dnz);
                v.normalize();
                dst_normals->push_back(v);
            }else{
                //Mark as invalid
                dst_normals->push_back(Eigen::Vector3f(0.0f,0.0f,0.0f));
            }
        }
        else if(str.compare("v") == 0){
            count++;
                double x, y, z;
                infile >> x; infile >> y; infile >> z;
                //_cprintf("v %d %f %f %f\n",count,x,y,z);
                if (invert_z) z = -z;
                Eigen::Vector3f p((float)x,(float)y,(float)z);
                dst_points->push_back(p);
        }
    }




    //Verifiy data////////////////////////
	if(dst_points->size()>0){
		//If the normals do not match the points clear the normals
		//to force them to be recalculated later.
		if(dst_points->size()!= dst_normals->size()){
            dst_normals->clear();
        }

		std::vector<Eigen::Vector3f>::iterator nitr = dst_normals->begin();
		std::vector<Eigen::Vector3f>::iterator pitr = dst_points->begin();

		//check for bad points
		while(nitr!=dst_normals->end() && pitr!=dst_points->end()){
			if((*pitr)[0]==0 && (*pitr)[1]==0 && (*pitr)[2]==0){
				//bad point delete
				pitr=dst_points->erase(pitr);
				if(dst_normals->size()>0){
					nitr=dst_normals->erase(nitr);
				}
			}
            else{
				nitr++;
				pitr++;
			}
		}
		nitr = dst_normals->begin();
		pitr = dst_points->begin();
		int ncount=0;
		//count the number of undefined normals
		while(nitr!=dst_normals->end()){
			if((*nitr).x()==0 && (*nitr).y()==0 && (*nitr).z()==0){
				ncount++;
			}
			nitr++;
		}

		//If more then 10% of the normals are undefined delete all of the normals
		// and recalculate them later
		if(ncount>dst_normals->size()*.1){
			dst_normals->clear();
			//MessageBox(0, "More then 10% of the normals are undefined!", "Error:", MB_OK);
		}
        else{
			//Remove the points with undefined normals
			nitr = dst_normals->begin();
			pitr = dst_points->begin();
			//check for bad normals

			int count_e = 0;
			while(nitr!=dst_normals->end() && pitr!=dst_points->end()){
				if((*nitr).x()==0 && (*nitr).y()==0 && (*nitr).z()==0){
					//bad normal delete

					pitr = dst_points->erase(pitr);
					nitr = dst_normals->erase(nitr);
					// RR, Dec. 16, 2015: that's better

				}else{
					nitr++;
					pitr++;
				}
				count_e++;
			}
		}

    
    }

	if (verbose) {
		#ifdef __WIN32
		_cprintf("[LoaderObj] - Info: loaded %d points and %d normals from %s.\n", dst_points->size(), dst_normals->size(), file.c_str())
			#else
		cout << "[LoaderObj] - Info:  loaded " << dst_points->size() << " points and " <<  dst_normals->size() << " normals from" <<  file.c_str() << "." << endl;
        #endif
	}

    return true;

}