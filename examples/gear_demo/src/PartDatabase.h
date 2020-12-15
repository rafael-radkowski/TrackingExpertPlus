#pragma once
/*

*/

//std
#include <unordered_map>

#include "FileUtils.h"

#include "Sequence.h"

class PartDatabase
{
private:
	std::unordered_map<int, Model*> models;

	int size;

public:
	PartDatabase();
	~PartDatabase();

	/*
	Load objects from .obj files into PartDatabase given a .txt file that
	lists, line by line, the paths and filenames of each .obj file to be loaded
	(e.g. D:/ModelsPath/OBJModels/model_to_be_loaded.obj).  The models previously
	stored in this PartDatabase are cleared when calling this function.

	@param path - the path of the .txt file to load .obj files from.
	*/
	bool loadObjsFromFile(const char* path);

	bool setNumDuplicates(const char* name, int num_dupes);

	int getNumModels() { return size; };

	/*
	Get the object specified by the given line of the .txt file used to
	load .obj files from.

	@param id - the id of the object to be returned, given by its line number in the .txt file.
	
	@return a pointer to the OBJModel stored at the given id
	*/
	Model* getObj(int id) { return models.at(id); };
};