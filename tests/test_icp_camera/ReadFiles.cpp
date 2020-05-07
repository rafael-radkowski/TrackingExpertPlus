#include "ReadFiles.h"



/*
	Read all files of a particular type from a folder
	@param folder - the path to the folder. 
	@param type - the ending, e.g., obj or ply.
	@param dst_path - a std::vector container with all files as sting.
	*/
	bool ReadFiles::GetFileList(const std::string folder, const std::string type, std::vector<std::string>& dst_path)
	{
		dst_path.clear();

#ifdef _WIN32
	#if _MSC_VER >= 1920 && _MSVC_LANG  == 201703L 
		for (const auto& entry : std::experimental::filesystem::directory_iterator(folder)) {
			//std::cout << entry.path() << std::endl;
			std::string file = entry.path().string();
 

			int idx = file.find_last_of(".");
			if(idx == std::string::npos)
				continue;

			std::string local_type = file.substr(idx+1, 3);
			if (local_type.compare(type) == 0) {
				//std::cout << file << std::endl;
				dst_path.push_back(file);
			}
		}
		
	#else
		#define _SILENCE_ALL_CXX17_DEPRECATION_WARNINGS
		#define _SILENCE_EXPERIMENTAL_FILESYSTEM_DEPRECATION_WARNING
		
		

		for (const auto& entry : std::experimental::filesystem::directory_iterator(folder)) {
			//std::cout << entry.path() << std::endl;
			std::string file = entry.path().string();


			int idx = file.find_last_of(".");
			if(idx == std::string::npos)
				continue;

			std::string local_type = file.substr(idx+1, 3);
			if (local_type.compare(type) == 0) {
				//std::cout << file << std::endl;
				dst_path.push_back(file);
			}
		}
				
	#endif

	
#else
	
#endif

		return  (dst_path.size() > 0) ?  true :  false;
		
	}