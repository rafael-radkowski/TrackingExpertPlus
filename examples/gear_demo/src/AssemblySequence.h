#pragma once

#include <vector>
#include <unordered_map>
#include "GearBoxRenderer.h"

class AssemblySequence
{
public:

	AssemblySequence();
	~AssemblySequence();


	static void process(std::unordered_map<string, Model*> models, glm::mat4 proj, glm::mat4 vm);
	static bool nextStage();
	static bool prevStage();

	static void setSeq(std::vector<int> sequence);

};