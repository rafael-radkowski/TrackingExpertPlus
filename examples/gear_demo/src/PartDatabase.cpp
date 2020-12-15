#include "PartDatabase.h"

Model* findModel(std::unordered_map<int, Model*> models, std::string name)
{
	for (std::pair<int, Model*> p : models)
	{
		if (p.second->name.compare(name) == 0)
		{
			return p.second;
		}
	}

	return NULL;
}

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
			nModel->duplicates = 0;
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

bool PartDatabase::setNumDuplicates(const char* name, int num_dupes)
{
	if (num_dupes < 0)
	{
		cout << "WARNING: PartDatabase: setNumDuplicates: Cannot set a negative amount of duplicates.  Number of duplicates not set." << endl;
		return false;
	}

	Model* ref_model = findModel(models, name);

	if (ref_model == NULL) return false;
	if (num_dupes == ref_model->duplicates) return false;

	if (num_dupes > ref_model->duplicates)
	{
		Model* dModel;
		char* new_name = (char*)malloc(100 * sizeof(char));
		for (int i = ref_model->duplicates; i <= num_dupes; i++)
		{
			sprintf(new_name, "%s-%d", name, i);

			if (findModel(models, new_name) != NULL) continue;

			dModel = new Model();
			dModel->model = new cs557::OBJModel(*ref_model->model);
			dModel->name = std::string(new_name);

			models.insert(std::make_pair(size, dModel));
			size++;
		}
	}

	ref_model->duplicates = num_dupes;

	return true;
}