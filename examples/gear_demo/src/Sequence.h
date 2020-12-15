#pragma once

#include <string>
#include <unordered_map>

#include "ModelOBJ.h"

typedef struct Model
{
	std::string name = "null";
	bool visible = false;
	cs557::OBJModel* model;
	int duplicates = 0;

	Model()
	{
		name = "null";
		visible = false;
		model = new cs557::OBJModel();
		duplicates = 0;
	}
};

struct Sequence
{
	int _current_state = 0;
	int* _forward = 0;
	int* _backward = 0;

	std::unordered_map<std::string, Model*> _models;

	Sequence()
	{
		_current_state = 0;
		_forward = 0;
		_backward = 0;

		_models = std::unordered_map<std::string, Model*>();
	}
};