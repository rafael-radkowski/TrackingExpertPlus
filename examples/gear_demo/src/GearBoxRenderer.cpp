#include "GearBoxRenderer.h"

GearBoxRenderer::GearBoxRenderer()
{
	models = std::vector<cs557::OBJModel*>();
}

GearBoxRenderer::~GearBoxRenderer()
{
}

void GearBoxRenderer::addModel(cs557::OBJModel* model)
{
	models.push_back(model);
}

void GearBoxRenderer::setTransform(int id, glm::mat4 transform)
{
	models.at(id)->setModelMatrix(transform);
}

void GearBoxRenderer::draw(glm::mat4 proj, glm::mat4 vm)
{
	for (int i = 0; i < models.size(); i++) 
	{
		models.at(i)->draw(proj, vm);
	}
}