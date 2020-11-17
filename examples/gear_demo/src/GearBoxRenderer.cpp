#include "GearBoxRenderer.h"

GearBoxRenderer::GearBoxRenderer()
{
	asm_seq = Sequence();

	asm_seq._models = std::unordered_map<string, Model*>();

	std::vector<int> order = std::vector<int>();
	for (int i = 0; i < 24; i++)
	{
		order.push_back(i);
	}

	AssemblySequence::setSeq(order, asm_seq);
}

GearBoxRenderer::~GearBoxRenderer()
{
}

void GearBoxRenderer::addModel(Model* model, string model_name)
{
	asm_seq._models.insert(std::make_pair(model_name, model));
}

void GearBoxRenderer::clearModels()
{
	asm_seq._models.clear();
}

void GearBoxRenderer::setTransform(string model_name, glm::mat4 transform)
{
	asm_seq._models.at(model_name)->model->setModelMatrix(transform);
}

void GearBoxRenderer::draw(glm::mat4 proj, glm::mat4 vm)
{
	AssemblySequence::process(asm_seq, proj, vm);
}