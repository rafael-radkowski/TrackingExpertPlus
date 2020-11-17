#pragma once

#include <vector>
#include <unordered_map>
#include "Sequence.h"

class AssemblySequence
{
public:

	static void process(Sequence& sequence, glm::mat4 proj, glm::mat4 vm);
	static bool nextStage(Sequence& sequence);
	static bool prevStage(Sequence& sequence);

	static void setSeq(std::vector<int> order, Sequence& sequence);
};