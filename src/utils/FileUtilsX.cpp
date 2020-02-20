#include "FileUtilsX.h"


using namespace texpert;

#ifdef _WIN32
	#if defined(_MSVC2017)
		namespace fs = std::filesystem;
	#else
		namespace fs = std::experimental::filesystem;
	#endif
#endif

/*
Check whether or not a file exists at the given path.
@param path_and_file - relative or absolute location and name of the file.
@return true, if the file exists, false otherwise. 
*/
//static
bool FileUtils::Exists(string path_and_file)
{
 #ifdef _WIN32
	#if defined(_MSVC2017)
		if (std::filesystem::exists(path_and_file)) {
	#else
		if (std::experimental::filesystem::exists(path_and_file)) {
	#endif
			return true;
		} else {
			return false;
		}
#else
        struct stat buffer;
        return (stat (path_and_file.c_str(), &buffer) == 0);
#endif
}


/*
Search for a file in adjacent folders
@param path_and_file - relative or absolute location and name of the file.
@param new_path_and_file - relative or absolute location and name of the located file or "" if no file exists. 
@return true, if the file exists, false otherwise. 
*/
//static 
bool FileUtils::Search(string path_and_file, string& new_path_and_file)
{
	// Check if the file exists at this location. 
	if (Exists(path_and_file)) {
		new_path_and_file = path_and_file;
		return true;
	}

 #ifdef _WIN32
	// replace all \\ with /
	// ToDo: replace all \\ with /


	// Extract the filename from the path. 
	new_path_and_file = "";
	int idx = path_and_file.find_last_of("/");

	string file = "";
	if(idx == -1 ){
		file = path_and_file;
	}
	else {
		file = path_and_file.substr(idx+1, path_and_file.length() - idx -1);
	}

	// Identify the search paths. 
	std::list<string> search_path_list;
	search_path_list.push_back(".");
	search_path_list.push_back("../");
	search_path_list.push_back("../../");
	
	// traverse to all adjacent folders. 
	while(search_path_list.size() > 0){
		string e = search_path_list.front();
		search_path_list.pop_front();
		for (const auto & entry : fs::directory_iterator(e)){
	#if defined(_MSVC2017)
			if(entry.is_directory()){
				search_path_list.push_back(entry.path().string());
			}
	#else
			if(fs::is_directory(entry)){
				search_path_list.push_back(entry.path().string());
			}
	#endif
		}

		// search for all files in these folder. 
		// Return if the search path exists. 
		string p = e + "/" + file;
		if(Exists(p)){
			new_path_and_file = p;
			cout << "[INFO] - Changed path for " << path_and_file << " to " << new_path_and_file << "." << endl;
			return true;
		}

	}

	new_path_and_file = "";
	return false;
 #else
	// Linux version is not implemented. The code returns false if the file does not exists. 
	if (FileUtils::Exists(path_and_file)) {
		new_path_and_file = path_and_file;
		return true;
	}else{
		new_path_and_file = "";
		return false;
	}
 #endif
}
