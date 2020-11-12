#pragma once
/*

*/

//std
#include <vector>

#include "FileUtils.h"

//gl_support
#include "ModelOBJ.h"

#include "PartDatabase.h"

class GearBoxRenderer
{
private:
	std::vector<cs557::OBJModel*> models;

public:
	GearBoxRenderer();
	~GearBoxRenderer();

	/*
	Load models into the stored PartDatabase

	@param filepath - the path and name of the .txt file with the list of .objs to load
	*/
	bool loadModelsIntoDatabase(const char* filepath);

	/*
	Add a model from the stored PartDatabase into this GearBoxRenderer

	@param id - the id of the object from the PartDatabase to load into this GearBoxRenderer
	*/
	void addModel(cs557::OBJModel* model);

	/*
	Set the transformation of an object at the given index

	@param id - the id of the object whose transformation is being changed.
	@param transform - the transformation to be applied to the object
	*/
	void setTransform(int id, glm::mat4 transform);

	/*
	Calls all rendering processes.

	@param proj - the projection matrix
	@param vm - the view matrix
	*/
	void draw(glm::mat4 proj, glm::mat4 vm);
};
