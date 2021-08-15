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
	Model(std::string _name, std::string path)
	{
		name = _name;
		visible = false;
		model = new cs557::OBJModel();
		model->create(path);
		duplicates = 0;
	}
};

struct Sequence
{
	int _current_state = 0;
	int* _forward = 0;
	int* _backward = 0;

	std::unordered_map<std::string, Model*> _models;
	std::vector<std::vector<std::pair<string, glm::mat4>>> _steps;

	Sequence()
	{
		_current_state = 0;
		_forward = 0;
		_backward = 0;

		_models = std::unordered_map<std::string, Model*>();
		_steps = std::vector<std::vector<std::pair<string, glm::mat4>>>();
	}

	Sequence(int num_steps)
	{
		_current_state = 0;
		_forward = 0;
		_backward = 0;

		_models = std::unordered_map<std::string, Model*>();
		_steps = std::vector<std::vector<std::pair<string, glm::mat4>>>(num_steps);
	}
};