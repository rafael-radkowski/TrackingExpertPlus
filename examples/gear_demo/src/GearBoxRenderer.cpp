#include "GearBoxRenderer.h"

GearBoxRenderer::GearBoxRenderer()
{
	models = std::vector<Model*>();
}

GearBoxRenderer::~GearBoxRenderer()
{
}

void GearBoxRenderer::addModel(Model* model)
{
	models.push_back(model);
}

void GearBoxRenderer::claerModels()
{
	models.clear();
}

void GearBoxRenderer::setTransform(int id, glm::mat4 transform)
{
	models.at(id)->model->setModelMatrix(transform);
}

void GearBoxRenderer::draw(glm::mat4 proj, glm::mat4 vm)
{
	AssemblySequence::process(models, proj, vm);
}