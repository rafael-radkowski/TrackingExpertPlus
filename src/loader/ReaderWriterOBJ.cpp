#include "ReaderWriterOBJ.h"



	/*!
Load a point cloud object from a file
@param file - The file
@param loadedNormals = The output location of the loaded normals
@return cloud = The output of the loaded point cloud object
*/
//virtual 
//static 
bool ReaderWriterOBJ::Read(const std::string file, std::vector<Eigen::Vector3f>& dst_points, std::vector<Eigen::Vector3f>& dst_normals, const bool normalize, const bool invert_z)
{
	std::string found_file;
	if (!texpert::FileUtils::Search(file, found_file)) {
		std::cout << "[ERROR] - ReaderWriterOBJ: the file " << found_file << " does not exist." << std::endl;
		return false;
	}

	if (!check_type(found_file, "obj")) {
		std::cout << "[ERROR] - ReaderWriterOBJ: the file " << found_file << " is not obj wavefront file. This loader only supports obj." << std::endl;
		return false;
	}


	std::ifstream infile(found_file);

    if(!infile.is_open()){
        #ifdef _WIN32
        _cprintf("[ERROR] - ReaderWriterPLY: could not open file %s.\n", found_file.c_str());
        #else
         cout << "[ERROR] - ReaderWriterPLY: could not open file " <<  file << "." << endl;
        #endif
        return false;
    }

	dst_points.clear();
	dst_normals.clear();

	
	
	int count_n  = 0;
	int count_n_err  = 0;
	int count_p  = 0;
	int count_p_err  = 0;
	std::string str = "";
	while(std::getline(infile, str))
    {
		std::vector<string> e = ReaderWriter::split(str, ' ');
		if(e.size() == 0) continue;

        if(e[0].compare("vn") == 0){
           

			if(e.size() == 4){
				string nx, ny, nz;
				nx = e[1]; ny = e[2]; nz = e[3];
				// remove werid meshlab things and invalid data
				if(is_number(nx) && is_number(ny) && is_number(nz)){
					float dnx = std::stof(nx);
					float dny = std::stof(ny);
					float dnz = std::stof(nz);

					if (invert_z) dnz = -dnz;
					Eigen::Vector3f v((float)dnx,(float)dny,(float)dnz);
					if(v.norm() > 1.001)
						v.normalize();
					dst_normals.push_back(v);

					 count_n++;

				}else{
					//Mark as invalid
					dst_normals.push_back(Eigen::Vector3f(0.0f,0.0f,0.0f));
					count_n_err++;
				}
			}
        }
        else if(e[0].compare("v") == 0){
				
                string x, y, z;
				x = e[1]; y = e[2]; z = e[3];

				if(is_number(x) && is_number(y) && is_number(z)){
					float dx = std::stof(x);
					float dy = std::stof(y);
					float dz = std::stof(z);
               
					if (invert_z) dz = -dz;

					Eigen::Vector3f p((float)dx,(float)dy,(float)dz);
					dst_points.push_back(p);

					count_p++;
				}else
				{
					dst_points.push_back(Eigen::Vector3f(0.0f,0.0f,0.0f));
					count_p_err++;
				}
        }
    }



	infile.close();

	std::cout << "[INFO] - ReaderWriterOBJ: loaded " << count_p << " points and " << count_n << " normal vectors from file " << found_file << "." << std::endl;
	if (count_n_err > 0 || count_p_err > 0) {
		std::cout << "[INFO] - ReaderWriterOBJ: detected " << count_p_err << " invalid points and " << count_n_err << " invalid normal vectors in file " << found_file << "." << std::endl;
	}

	return true;
}


/*
Write the point cloud data to a file
@param file - string containing path and name
@param dst_points - vector of vector3f points containing x, y, z coordinates
@param dst_normals - vector of vector3f normal vectors index-aligned to the points.
@param scale_points - float value > 0.0 that scales all points and normal vectors. 
*/
//virtual 
//static 
bool ReaderWriterOBJ::Write(std::string file, std::vector<Eigen::Vector3f>& src_points, std::vector<Eigen::Vector3f>& src_normals, const float scale_points)
{
	
	// check if the number of points matches the number of normals
	if (src_points.size() != src_normals.size()) {
		std::cout << "[WARNING] - ReaderWriterOBJ: number of points and normals does not match: " << src_points.size() << " != " << src_normals.size() << std::endl;
		//return false;
	}


	// append a obj ending 
	int index = file.find_last_of(".");
	std::string outfile;
	if (index != -1)
	{
		outfile = file.substr(0, index);
	}
	outfile.append(".obj");


	std::ofstream of;
	of.open(outfile, std::ofstream::out);

	size_t size = src_points.size();

	if (!of.is_open()) {
		std::cout << "[ERROR] - ReaderWriterOBJ: cannot open file " << outfile << " for writing." << std::endl;
		return false;
	}
	
	of << "# Created by SurfExtract point cloud generation.\n";
	of << "# Size: " << size << "\n\n";

	Eigen::Matrix3f T = Eigen::Matrix3f::Identity();
	T(0,0) = scale_points;
	T(1,1) = scale_points;
	T(2,2) = scale_points;

	// inverse transpose
	// not really required now. But we most likely rotate the object in future.
	// so it should already be inside.
	Eigen::Matrix3f Tit = (T.inverse()).transpose(); 

	int inv_cout = 0; // counts invalid normal vectors 
	for (int i = 0; i < size; i++) {

		Eigen::Vector3f o_p =  T * src_points[i];
		Eigen::Vector3f o_n =  Tit * src_normals[i];
		//o_n.normalize();

	
		if(o_n.norm() == 0.0){
			inv_cout++;
		}

		of << std::fixed << "v " << o_p[0] << " " << o_p[1] << " " << o_p[2] << "\n";
		of << std::fixed << "vn " << o_n[0] << " " << o_n[1] << " " << o_n[2] << "\n";
	}


	of.close();
	

	std::cout << "[INFO] - ReaderWriterOBJ: saved " << size << " points and normal vectors to file " << outfile << "." << std::endl;


	return true;
}


//static 
void  ReaderWriterOBJ::ErrorMsg(std::string msg)
{
	std::cout << "[ERROR] - ReaderWriterOBJ: " << msg << "." << std::endl;
}