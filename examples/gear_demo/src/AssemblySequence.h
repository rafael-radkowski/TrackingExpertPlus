#pragma once

#include <vector>
#include <unordered_map>
#include "GearBoxRenderer.h"

class AssemblySequence
{
private:
	int _current_state;
	int* _forward;
	int* _backward;
public:

	AssemblySequence();
	~AssemblySequence();

	void stage00();
	void stage01();
	void stage02();
	void stage03();


	void process(std::unordered_map<int, Model*> models);
	bool nextStage();
	bool prevStage();

};