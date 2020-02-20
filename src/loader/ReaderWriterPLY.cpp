#include "ReaderWriterPLY.h"



	/*!
Load a point cloud object from a file
@param file - The file
@param loadedNormals = The output location of the loaded normals
@return cloud = The output of the loaded point cloud object
*/
//virtual 
//static 
bool ReaderWriterPLY::Read(const std::string file, std::vector<Eigen::Vector3f>& dst_points, std::vector<Eigen::Vector3f>& dst_normals, const bool normalize, const bool invert_z)
{
	if (!texpert::FileUtils::Exists(file)) {
		std::cout << "[ERROR] - ReaderWriterPLY: the file " << file << " does not exist." << std::endl;
		return false;
	}


	// append a ply ending 
	int index = file.find_last_of(".");
	std::string outfile;

	if (index != -1)
	{
		outfile = file.substr(0, index);
	}
	outfile.append(".ply");


	std::ifstream infile(file);

    if(!infile.is_open()){
        #ifdef _WIN32
        _cprintf("[ERROR] - ReaderWriterPLY: could not open file %s.\n", file.c_str());
        #else
         cout << "[ERROR] - ReaderWriterPLY: could not open file " <<  file << "." << endl;
        #endif
        return false;
    }

	dst_points.clear();
	dst_normals.clear();

	int count = 0;
	int size = 0;
	bool found_data = false;
	std::string str;
    while(std::getline(infile, str))
    {
		std::vector<string> e = ReaderWriter::split(str, ' ');
		if(e.size() == 0) continue;

		if (!found_data) {
	
			if (e[0].compare("element") == 0) {
				if (e.size() == 3) {
					if (e[1].compare("vertex") == 0) {
						string s = e[2];
						if(ReaderWriter::is_number(s))
						{
							size = std::atoi(s.c_str());
							if (size > 0) {
								dst_points.reserve(size);
								dst_normals.reserve(size);	
							}
						}
					}
				}
			}
			else if (e[0].compare("end_header") == 0) {
				found_data = true;
			}
		}
		else{
			
			float x, y, z, nx, ny, nz;
			if (e.size() == 6) {
				if(ReaderWriter::is_number(e[0]))
					x = std::atof(e[0].c_str());
				else ErrorMsg("coordinate x is NaN at vertex " + std::to_string(count));

				if(ReaderWriter::is_number(e[1]))
					y = std::atof(e[1].c_str());
				else ErrorMsg("coordinate y is NaN at vertex " + std::to_string(count));

				if(ReaderWriter::is_number(e[2]))
					z = std::atof(e[2].c_str());
				else ErrorMsg("coordinate z is NaN at vertex " + std::to_string(count));
						
				if(ReaderWriter::is_number(e[3]))
					nx = std::atof(e[3].c_str());
				else ErrorMsg("coordinate nx is NaN at vertex " + std::to_string(count));

				if(ReaderWriter::is_number(e[4]))
					ny = std::atof(e[4].c_str());
				else ErrorMsg("coordinate ny is NaN at vertex " + std::to_string(count));

				if(ReaderWriter::is_number(e[5]))
					nz = std::atof(e[5].c_str());
				else ErrorMsg("coordinate nz is NaN at vertex " + std::to_string(count));

				dst_points.push_back(Eigen::Vector3f(x, y, z));
				dst_normals.push_back(Eigen::Vector3f(nx, ny, nz));

				count++;
			}
		}


	}

	infile.close();

	std::cout << "[INFO] - ReaderWriterPLY: loaded " << count << " points and normal vectors from file " << file << "." << std::endl;

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
bool ReaderWriterPLY::Write(std::string file, std::vector<Eigen::Vector3f>& src_points, std::vector<Eigen::Vector3f>& src_normals, const float scale_points)
{
	
	// check if the number of points matches the number of normals
	if (src_points.size() != src_normals.size()) {
		std::cout << "[ERROR] - ReaderWriterPLY: number of points and normals does not match: " << src_points.size() << " != " << src_normals.size() << std::endl;
		return false;
	}


	// append a pcd ending 
	int index = file.find_last_of(".");
	std::string outfile;

	if (index != -1)
	{
		outfile = file.substr(0, index);
	}
	outfile.append(".ply");


	std::ofstream of;
	of.open(outfile, std::ofstream::out);

	size_t size = src_points.size();

	if (!of.is_open()) {
		std::cout << "[ERROR] - ReaderWriterPLY: cannot open file " << outfile << " for writing." << std::endl;
		return false;
	}
	

	of << "ply\n";
	of << "format ascii 1.0\n";
	of << "comment SurfExtract output\n";
	of << "element vertex " << size << "\n";
	of << "property float x\n";
	of << "property float y\n";
	of << "property float z\n";
	of << "property float nx\n";
	of << "property float ny\n";
	of << "property float nz\n";
	of << "end_header\n";

	Eigen::Matrix3f T = Eigen::Matrix3f::Identity();
	T(0,0) = scale_points;
	T(1,1) = scale_points;
	T(2,2) = scale_points;
	Eigen::Matrix3f Tit = (T.inverse()).transpose(); 



	for (int i = 0; i < size; i++) {

		Eigen::Vector3f p = src_points[i];
		Eigen::Vector3f n = src_normals[i];

		Eigen::Vector3f o_p =  T * p;
		Eigen::Vector3f o_n =  Tit * n;
				
		if (p[0] || p[1] || p[2]) {
			of << std::fixed << scale_points * o_p.x() << " " << scale_points * o_p.y() << " " << scale_points * o_p.z() << " "  << o_n.x() << " " << o_n.y() << " " << o_n.z() << "\n";
		}
	}
	of.close();
	

	std::cout << "[INFO] - ReaderWriterPLY: saved " << size << " points and normal vectors to file " << outfile << "." << std::endl;


	return true;
}


//static 
void  ReaderWriterPLY::ErrorMsg(std::string msg)
{
	std::cout << "[ERROR] - ReaderWriterPLY: " << msg << "." << std::endl;
}