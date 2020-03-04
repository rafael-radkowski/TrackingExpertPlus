#pragma once

#include <iostream>
#include <string>
#include <vector>
#include <iostream>
#include <filesystem>


class ReadFiles
{
public:

	/*
	Read all files of a particular type from a folder
	@param folder - the path to the folder. 
	@param type - the ending, e.g., obj or ply.
	@param dst_path - a std::vector container with all files as sting.
	*/
	static bool	GetFileList(const std::string folder, const std::string type, std::vector<std::string>& dst_path);

};