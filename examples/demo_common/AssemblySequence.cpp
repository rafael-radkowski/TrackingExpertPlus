#include "AssemblySequence.h"
#include <iterator>

#define PI 3.1415926535

bool setModelVisible(string name, Sequence& sequence)
{
	if (sequence._models.find(name) != sequence._models.end())
	{
		sequence._models.find(name)->second->visible = true;
		return true;
	}

	return false;
}

bool setModelPos(string name, Sequence& sequence, glm::mat4 pos)
{
	if (sequence._models.find(name) != sequence._models.end())
	{
		sequence._models.find(name)->second->model->setModelMatrix(pos);
		return true;
	}

	return false;
}

void AssemblySequence::update(Sequence& sequence)
{
	for (auto i = sequence._models.begin(); i != sequence._models.end(); i++)
	{
		i->second->visible = false;
	}

	for (std::pair<string, glm::mat4> p : sequence._steps.at(sequence._current_state))
	{
		setModelPos(p.first, sequence, p.second);
		setModelVisible(p.first, sequence);
	}
}

//static
void AssemblySequence::process(Sequence& sequence, glm::mat4 proj, glm::mat4 vm)
{
	Model* curModel;

	for (auto i = sequence._models.begin(); i != sequence._models.end(); i++)
	{
		curModel = i->second;
		if (curModel->name.compare("null") != 0 && curModel->visible)
			curModel->model->draw(proj, vm);
	}
}

//static
bool AssemblySequence::nextStage(Sequence& sequence)
{
	sequence._current_state = sequence._forward[sequence._current_state];

	update(sequence);

	return true;
}

//static
bool AssemblySequence::prevStage(Sequence& sequence)
{
	sequence._current_state = sequence._backward[sequence._current_state];

	update(sequence);

	return true;
}

//static
void AssemblySequence::setSeq(std::vector<int> order, Sequence& sequence)
{
	sequence._forward = (int*)malloc(order.size() * sizeof(int));
	sequence._backward = (int*)malloc(order.size() * sizeof(int));

	for (int i = 0; i < order.size() - 1; i++)
	{
		sequence._forward[i] = order.at(i + 1);
	}
	sequence._forward[order.size() - 1] = order.at(0);

	sequence._backward[0] = order.at(order.size() - 1);
	for (int i = 1; i < order.size(); i++)
	{
		sequence._backward[i] = order.at(i - 1);
	}
}
