#include "PartDatabase.h"

PartDatabase::PartDatabase()
{
	models = std::unordered_map<int, Model>(50);
}

PartDatabase::~PartDatabase()
{
}


bool PartDatabase::loadObjsFromFile(const char* path)
{
	//models.clear();

	fstream objList = fstream();

	objList.open(path, ios_base::in);
	if (objList.bad())
	{
		cout << "ERROR: PartDatabase: loadObjsFromFile: Failed to open file: " << path << "\n";
		return false;
	}

	char* fileLine = (char*)malloc(100 * sizeof(char));

	int idx = 0;
	Model nModel;
	while (!objList.eof())
	{
		nModel = Model();
		objList.getline(fileLine, 100);
		
		if (std::experimental::filesystem::exists(fileLine))
		{
			nModel.model->create(fileLine);
			nModel.visible = false;
			nModel.name = std::experimental::filesystem::path(fileLine).filename().string();
		}

		else
		{
			nModel.name = "null";
			nModel.visible = false;
			cout << "WARNING: PartDatabase: File " << fileLine << " does not exist.  It will be ignored.\n";
		}

		models.insert({ idx, nModel });

		idx++;
	}

	return true;
}

Model PartDatabase::getObj(int id)
{
	return models.at(id);
}