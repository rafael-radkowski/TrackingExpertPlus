#include "ProcedureLoader.h"

bool ProcedureLoader::loadProcedure(const std::string& path, Sequence& _sequence)
{
	// determine the file type
	int idx = path.find_last_of(".");
	std::string sub = path.substr(idx + 1, 3);
	std::transform(sub.begin(), sub.end(), sub.begin(), ::tolower);

	if (!(sub.compare("jso") == 0))
	{
		std::cerr << "[ProcedureLoader] Error, cannot load " << path << ". Wrong file format. json is required." << std::endl;
		return false;
	}

	//Set up sequence, load .json file
	_sequence = Sequence();
	cv::FileStorage fs(path, cv::FileStorage::READ);

	if (fs.isOpened())
	{
		for (auto& m : fs["models"])
		{
			Model* model = new Model(m["name"], m["path"]);
			_sequence._models.insert(std::make_pair(m["name"], model));
		}
	}
}