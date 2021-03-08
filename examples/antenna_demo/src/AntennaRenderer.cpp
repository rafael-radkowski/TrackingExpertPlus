#include "AntennaRenderer.h"

#define PI 3.1415926535

AntennaRenderer::AntennaRenderer()
{
	asm_seq = Sequence(8);

	asm_seq._models = std::unordered_map<string, Model*>();

	std::vector<int> order = std::vector<int>();
	for (int i = 0; i < 8; i++)
	{
		order.push_back(i);
	}

	AssemblySequence::setSeq(order, asm_seq);
}

AntennaRenderer::~AntennaRenderer()
{
}

void AntennaRenderer::addModel(Model* model, string model_name)
{
	asm_seq._models.insert(std::make_pair(model_name, model));
}

void AntennaRenderer::clearModels()
{
	asm_seq._models.clear();
}

void AntennaRenderer::setTransform(string model_name, glm::mat4 transform)
{
	asm_seq._models.at(model_name)->model->setModelMatrix(transform);
}

void AntennaRenderer::progress(bool forward)
{
	if (forward)
		AssemblySequence::nextStage(asm_seq);
	else
		AssemblySequence::prevStage(asm_seq);
}

void AntennaRenderer::updateInPlace()
{
	AssemblySequence::update(asm_seq);
}

void AntennaRenderer::draw(glm::mat4 proj, glm::mat4 vm)
{
	AssemblySequence::process(asm_seq, proj, vm);
}

void AntennaRenderer::setSteps()
{
	std::vector<std::pair<string, glm::mat4>> refModels = std::vector<std::pair<string, glm::mat4>>();

	refModels.push_back(make_pair<string, glm::mat4>("PartA_pc_gfx.obj", glm::translate(glm::vec3(0.0f, 0.0f, 0.0f))));
	refModels.push_back(make_pair<string, glm::mat4>("PartB_pc_gfx.obj", glm::rotate(glm::translate(glm::vec3(-0.77f, 0.1f, 0.0f)), (float)PI, glm::vec3(1, 0, 0))));
	asm_seq._steps.at(0) = refModels;
	refModels.clear();

	refModels.push_back(make_pair<string, glm::mat4>("PartA_pc_gfx.obj", glm::translate(glm::vec3(0.0f, 0.0f, 0.0f))));
	refModels.push_back(make_pair<string, glm::mat4>("PartB_pc_gfx.obj", glm::rotate(glm::translate(glm::vec3(0.34f, 0.1f, 0.0f)), (float)PI, glm::vec3(1, 0, 0))));
	asm_seq._steps.at(1) = refModels;
	refModels.clear();

	refModels.push_back(make_pair<string, glm::mat4>("PartA_pc_gfx.obj", glm::translate(glm::vec3(0.0f, 0.0f, 0.0f))));
	refModels.push_back(make_pair<string, glm::mat4>("PartB_pc_gfx.obj", glm::rotate(glm::translate(glm::vec3(0.34f, 0.1f, 0.0f)), (float)PI, glm::vec3(1, 0, 0))));
	refModels.push_back(make_pair<string, glm::mat4>("PartC_pc_gfx.obj", glm::translate(glm::vec3(-0.77f, 0.1f, 0.0f)) * glm::rotate((float)PI / 2, glm::vec3(0, 1, 0)) * glm::rotate((float)PI / 2, glm::vec3(0, 0, 1))));
	asm_seq._steps.at(2) = refModels;
	refModels.clear();

	refModels.push_back(make_pair<string, glm::mat4>("PartA_pc_gfx.obj", glm::translate(glm::vec3(0.0f, 0.0f, 0.0f))));
	refModels.push_back(make_pair<string, glm::mat4>("PartB_pc_gfx.obj", glm::rotate(glm::translate(glm::vec3(0.34f, 0.1f, 0.0f)), (float)PI, glm::vec3(1, 0, 0))));
	refModels.push_back(make_pair<string, glm::mat4>("PartC_pc_gfx.obj", glm::translate(glm::vec3(0.01f, 0.135f, 0.0f)) * glm::rotate((float)PI / 2, glm::vec3(0, 1, 0)) * glm::rotate((float)PI / 2, glm::vec3(0, 0, 1))));
	asm_seq._steps.at(3) = refModels;
	refModels.clear();

	refModels.push_back(make_pair<string, glm::mat4>("PartA_pc_gfx.obj", glm::translate(glm::vec3(0.0f, 0.0f, 0.0f))));
	refModels.push_back(make_pair<string, glm::mat4>("PartB_pc_gfx.obj", glm::rotate(glm::translate(glm::vec3(0.34f, 0.1f, 0.0f)), (float)PI, glm::vec3(1, 0, 0))));
	refModels.push_back(make_pair<string, glm::mat4>("PartC_pc_gfx.obj", glm::translate(glm::vec3(0.01f, 0.135f, 0.0f)) * glm::rotate((float)PI / 2, glm::vec3(0, 1, 0)) * glm::rotate((float)PI / 2, glm::vec3(0, 0, 1))));
	refModels.push_back(make_pair<string, glm::mat4>("PartD_pc_gfx.obj", glm::translate(glm::vec3(-0.77f, 0.1f, 0.0f)) * glm::rotate((float)-PI / 3, glm::vec3(0, 1, 0))));
	asm_seq._steps.at(4) = refModels;
	refModels.clear();

	refModels.push_back(make_pair<string, glm::mat4>("PartA_pc_gfx.obj", glm::translate(glm::vec3(0.0f, 0.0f, 0.0f))));
	refModels.push_back(make_pair<string, glm::mat4>("PartB_pc_gfx.obj", glm::rotate(glm::translate(glm::vec3(0.34f, 0.1f, 0.0f)), (float)PI, glm::vec3(1, 0, 0))));
	refModels.push_back(make_pair<string, glm::mat4>("PartC_pc_gfx.obj", glm::translate(glm::vec3(0.01f, 0.135f, 0.0f)) * glm::rotate((float)PI / 2, glm::vec3(0, 1, 0)) * glm::rotate((float)PI / 2, glm::vec3(0, 0, 1))));
	refModels.push_back(make_pair<string, glm::mat4>("PartD_pc_gfx.obj", glm::translate(glm::vec3(-0.3f, 0.1f, -0.125f)) * glm::rotate((float)-PI / 3, glm::vec3(0, 1, 0))));
	asm_seq._steps.at(5) = refModels;
	refModels.clear();

	refModels.push_back(make_pair<string, glm::mat4>("PartA_pc_gfx.obj", glm::translate(glm::vec3(0.0f, 0.0f, 0.0f))));
	refModels.push_back(make_pair<string, glm::mat4>("PartB_pc_gfx.obj", glm::rotate(glm::translate(glm::vec3(0.34f, 0.1f, 0.0f)), (float)PI, glm::vec3(1, 0, 0))));
	refModels.push_back(make_pair<string, glm::mat4>("PartC_pc_gfx.obj", glm::translate(glm::vec3(0.01f, 0.135f, 0.0f)) * glm::rotate((float)PI / 2, glm::vec3(0, 1, 0)) * glm::rotate((float)PI / 2, glm::vec3(0, 0, 1))));
	refModels.push_back(make_pair<string, glm::mat4>("PartD_pc_gfx.obj", glm::translate(glm::vec3(-0.3f, 0.1f, -0.125f)) * glm::rotate((float)-PI / 3, glm::vec3(0, 1, 0))));
	refModels.push_back(make_pair<string, glm::mat4>("PartE_pc_gfx.obj", glm::translate(glm::vec3(-0.77f, 0.1f, 0.0f)) * glm::rotate((float)-PI / 2, glm::vec3(1, 0, 0)) * glm::rotate((float)-PI / 2, glm::vec3(0, 1, 0))));
	asm_seq._steps.at(6) = refModels;
	refModels.clear();

	refModels.push_back(make_pair<string, glm::mat4>("PartA_pc_gfx.obj", glm::translate(glm::vec3(0.0f, 0.0f, 0.0f))));
	refModels.push_back(make_pair<string, glm::mat4>("PartB_pc_gfx.obj", glm::rotate(glm::translate(glm::vec3(0.34f, 0.1f, 0.0f)), (float)PI, glm::vec3(1, 0, 0))));
	refModels.push_back(make_pair<string, glm::mat4>("PartC_pc_gfx.obj", glm::translate(glm::vec3(0.01f, 0.135f, 0.0f)) * glm::rotate((float)PI / 2, glm::vec3(0, 1, 0)) * glm::rotate((float)PI / 2, glm::vec3(0, 0, 1))));
	refModels.push_back(make_pair<string, glm::mat4>("PartD_pc_gfx.obj", glm::translate(glm::vec3(-0.3f, 0.1f, -0.125f)) * glm::rotate((float)-PI / 3, glm::vec3(0, 1, 0))));
	refModels.push_back(make_pair<string, glm::mat4>("PartE_pc_gfx.obj", glm::translate(glm::vec3(-0.3f, 0.1f, 0.03f)) * glm::rotate((float)-PI / 2, glm::vec3(1, 0, 0)) * glm::rotate((float)-PI / 2, glm::vec3(0, 1, 0))));
	asm_seq._steps.at(7) = refModels;
	refModels.clear();
}