#include "PartDatabase.h"

PartDatabase::PartDatabase()
{
	models = std::vector<cs557::OBJModel*>();
}

PartDatabase::~PartDatabase()
{
}


bool PartDatabase::loadObjsFromFile(const char* path)
{
	models.clear();

	fstream objList = fstream();

	objList.open(path, ios_base::in);
	if (objList.bad())
	{
		cout << "ERROR: PartDatabase: loadObjsFromFile: Failed to open file: " << path << "\n";
		return false;
	}

	char* fileLine = (char*)malloc(100 * sizeof(char));
	while (!objList.eof())
	{
		cs557::OBJModel* model = new cs557::OBJModel();
		objList.getline(fileLine, 100);
		
		if (std::experimental::filesystem::exists(fileLine))
		{
			model->create(fileLine);
		}

		else
		{
			cout << "WARNING: PartDatabase: File " << fileLine << " does not exist.  It will be ignored.\n";
		}

		models.push_back(model);
	}

	return true;
}

cs557::OBJModel* PartDatabase::getObj(int id)
{
	return models.at(id);
}