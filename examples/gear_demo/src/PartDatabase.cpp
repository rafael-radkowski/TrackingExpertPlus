#include "PartDatabase.h"

PartDatabase::PartDatabase()
{
	models = std::unordered_map<int, Model*>();
	size = 0;
}

PartDatabase::~PartDatabase()
{
}


bool PartDatabase::loadObjsFromFile(const char* path)
{
	models.clear();

	ifstream objList(path);

	if (!objList)
	{
		cout << "ERROR: PartDatabase: loadObjsFromFile: Failed to open file: " << path << "\n";
		return false;
	}

	char fileLine[100];

	int idx = 0;
	Model* nModel;
	while (!objList.eof())
	{
		nModel = new Model();
		objList.getline(fileLine, 100);
		
		if (std::experimental::filesystem::exists(fileLine))
		{
			nModel->model->create(fileLine);
			nModel->visible = false;
			nModel->name = std::experimental::filesystem::path(fileLine).filename().string();
			GLenum err = glGetError();
			while (err != GL_NO_ERROR)
			{
				err = glGetError();
			}
		}

		else
		{
			nModel->name = "null";
			nModel->visible = false;
			cout << "WARNING: PartDatabase: File " << fileLine << " does not exist.  It will be ignored.\n";
		}

		models.insert(std::make_pair(idx, nModel));

		idx++;
	}

	size = idx;

	return true;
}