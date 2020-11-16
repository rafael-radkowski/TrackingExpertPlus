#include "GearBoxRenderer.h"

GearBoxRenderer::GearBoxRenderer()
{
	_models = std::unordered_map<string, Model*>();
}

GearBoxRenderer::~GearBoxRenderer()
{
}

void GearBoxRenderer::addModel(Model* model, string model_name)
{
	_models.insert(std::make_pair(model_name, model));
}

void GearBoxRenderer::clearModels()
{
	_models.clear();
}

void GearBoxRenderer::setTransform(string model_name, glm::mat4 transform)
{
	_models.at(model_name)->model->setModelMatrix(transform);
}

void GearBoxRenderer::draw(glm::mat4 proj, glm::mat4 vm)
{
	AssemblySequence::process(_models, proj, vm);
}