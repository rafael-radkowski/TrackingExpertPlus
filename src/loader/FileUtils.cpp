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
	return std::experimental::filesystem::exists(path_and_file);
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


