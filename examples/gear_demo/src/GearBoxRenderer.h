#pragma once
/*

*/

//std
#include <vector>

#include "FileUtils.h"

#include "Sequence.h"
#include "AssemblySequence.h"



class GearBoxRenderer
{
private:
	Sequence asm_seq;

public:
	GearBoxRenderer();
	~GearBoxRenderer();

	/*
	Add a model from the stored PartDatabase into this GearBoxRenderer

	@param id - the id of the object from the PartDatabase to load into this GearBoxRenderer
	*/
	void addModel(Model* model, string model_name);

	void clearModels();

	/*
	Set the transformation of an object at the given index

	@param id - the id of the object whose transformation is being changed.
	@param transform - the transformation to be applied to the object
	*/
	void setTransform(string model_name, glm::mat4 transform);

	void progress(bool forward);

	void updateInPlace();

	/*
	Calls all rendering processes.

	@param proj - the projection matrix
	@param vm - the view matrix
	*/
	void draw(glm::mat4 proj, glm::mat4 vm);
};
