#include "DataReaderWriter.h"

using namespace texpert;

/*
Write a complete dataset to a folder. 
@param path - string containing the path
@param name - label for all images. 
@param point_cloud - reference to the point cloud object. 
@param rgb_img - cv::Mat containing the rgb image of type CV_8UC3
@param depth_map - cv::Mat containing the depth image of type CV_16UC1
@return true - if the image was sucessfully written.
*/
bool DataReaderWriter::Write(std::string path, std::string name, PointCloud& point_cloud, cv::Mat& rgb_img, cv::Mat& depth_map )
{
	bool ret = FileUtils::Exists(path);

	if (!ret) {
		//static 
                #if (_MSC_VER >= 1920 && _MSVC_LANG  == 201703L) || (__GNUC__ >= 8) 
			return std::filesystem::create_directories(path);
		#else
			return std::experimental::filesystem::create_directories(path);
		#endif
	}

	// write the obj file
	ReaderWriterOBJ::Write(name, point_cloud.points, point_cloud.normals);

	// write the ply file
	ReaderWriterPLY::Write(name, point_cloud.points, point_cloud.normals);



	return true;
}
