#include "ProcedureLoader.h"

bool ProcedureLoader::loadProcedure(const std::string& path, Procedure& _procedure)
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
	_procedure = Procedure();
	cv::FileStorage fs(path, cv::FileStorage::READ);

	bool main_found = false;
	std::unordered_map<std::string, Procedure> subproc = std::unordered_map<std::string, Procedure>();

	if (fs.isOpened())
	{
		//Load models
		for (auto& m : fs["models"])
		{
			cs557::OBJModel model = cs557::OBJModel();
			model.create(m["path"].string());
			_procedure._models->insert(std::make_pair(m["name"].string(), model));
		}

		//Load procedures
		for (auto& proc : fs["procedures"])
		{
			Procedure temp_proc = Procedure();
			temp_proc.name = proc["id"];

			for (auto& s : proc["instructions"])
			{
				if (s["subprocedure"].string().compare("false") == 0)
				{
					std::vector<std::string> prereqs = std::vector<std::string>();
					for (auto& prereq : s["prerequisites"])
					{
						prereqs.push_back(prereq.string());
					}

					glm::vec3 trans = glm::vec3();
					trans.x = s["trans"][0].real();
					trans.y = s["trans"][1].real();
					trans.z = s["trans"][2].real();

					glm::vec3 rot = glm::vec3();
					rot.x = s["rot"][0].real();
					rot.y = s["rot"][1].real();
					rot.z = s["rot"][2].real();

					_procedure._steps.insert(std::make_pair(s["id"].string(), Step(prereqs, s["model"].string(), trans, rot)));
				}
				else
				{

				}
			}

			//Handle main process and subprocesses
			if (proc["id"].string().compare("main") == 0)
			{
				main_found = true;
				temp_proc._subprocs = new std::unordered_map<std::string, Procedure>(subproc);
				_procedure = Procedure(temp_proc);
			}
			else 
			{
				if (main_found)
				{
					_procedure._subprocs->insert(make_pair(temp_proc.name, Procedure(temp_proc)));
				}
				else
				{
					subproc.insert(make_pair(temp_proc.name, Procedure(temp_proc)));
				}
			}
		}
	}
	else 
	{
		std::cerr << "[ProcedureLoader] Error, unable to open file " << path << "." << std::endl;
	}

	if (!main_found)
	{
		std::cerr << "[ProcedureLoader] Error, unable to find main procedure.  Please assign one procedure id 'main'." << std::endl;
		return false;
	}

	return true;
}