#include "FileUtils.h"



 /*
Check if a file exits. 
@param path_and_file - string containing the path and file, relative or absoulte. 
@return true, if the file exits. 
*/
//static 
bool FileUtils::Exists(string path_and_file)
{

#ifdef _WIN32
#if _MSC_VER >= 1920 && _MSVC_LANG  == 201703L 
	return std::filesystem::exists(path_and_file);
#else
	return std::experimental::filesystem::exists(path_and_file);
#endif
#else
    int res = access(path_and_file.c_str(), R_OK);
    if (res < 0) {
        if (errno == ENOENT) {
            // file does not exist
            return false;
        } else if (errno == EACCES) {
            // file exists but is not readable
            return false;
        } else {
             return false;
        }
    }
    return true;
#endif

}




//static 
bool FileUtils::CreateDirectories(string path)
{
#if _MSC_VER >= 1920 && _MSVC_LANG  == 201703L 
	return std::filesystem::create_directories(path);
#else
	return std::experimental::filesystem::create_directories(path);
#endif
}


/*
Return the names of all files at the given path
@param path  - string containing the path
@return std::vector with the file names. 
*/
//static
vector<string> FileUtils::GetFileList(string path)
{
	std::vector<std::string> files;
#if _MSC_VER >= 1920 && _MSVC_LANG  == 201703L 
	std::filesystem::directory_iterator itr(path);

	for (const auto & entry : itr) {
		//std::cout << entry.path().string() << std::endl;
		files.push_back(entry.path().string());
	}
#else

	std::experimental::filesystem::directory_iterator itr(path);

	for (const auto & entry : itr) {
		//std::cout << entry.path().string() << std::endl;
		files.push_back(entry.path().string());
	}
#endif

	return files;
}


//static 
bool FileUtils::Remove(string path)
{
#if _MSC_VER >= 1920 && _MSVC_LANG  == 201703L 
	return std::filesystem::remove(path);
#else
	return std::experimental::filesystem::remove(path);
#endif
}


	
//static 
bool FileUtils::CreateDirectory(string path)
{
#if _MSC_VER >= 1920 && _MSVC_LANG  == 201703L 
	return std::filesystem::create_directory(path);
#else
	return std::experimental::filesystem::create_directory(path);
#endif

}