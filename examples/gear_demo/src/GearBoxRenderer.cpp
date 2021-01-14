#include "GearBoxRenderer.h"

#define PI 3.1415926535

GearBoxRenderer::GearBoxRenderer()
{
	asm_seq = Sequence(24);

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

void GearBoxRenderer::progress(bool forward)
{
	if (forward)
		AssemblySequence::nextStage(asm_seq);
	else
		AssemblySequence::prevStage(asm_seq);
}

void GearBoxRenderer::updateInPlace()
{
	AssemblySequence::update(asm_seq);
}

void GearBoxRenderer::draw(glm::mat4 proj, glm::mat4 vm)
{
	AssemblySequence::process(asm_seq, proj, vm);
}

void GearBoxRenderer::setSteps()
{
	std::vector<std::pair<string, glm::mat4>> refModels = std::vector<std::pair<string, glm::mat4>>();

	/*
	Phase 1
	*/
	refModels.push_back(make_pair<string, glm::mat4>("N1-001_pc_gfx.obj", glm::translate(glm::mat4(1.0f), glm::vec3(0.0f, 0.0f, 0.12f))));
	asm_seq._steps.at(0) = refModels;
	refModels.clear();

	refModels.push_back(make_pair<string, glm::mat4>("N1-002_pc_gfx.obj", glm::translate(glm::vec3(0.0f, 0.0f, 0.0f))));
	asm_seq._steps.at(1) = refModels;

	refModels.push_back(make_pair<string, glm::mat4>("N1-001_pc_gfx.obj", glm::translate(glm::mat4(1.0f), glm::vec3(0.0f, 0.0f, 0.12f))));
	refModels.push_back(make_pair<string, glm::mat4>("N1-002_pc_gfx.obj", glm::translate(glm::vec3(0.0f, 0.0f, 0.0f))));
	refModels.push_back(make_pair<string, glm::mat4>("N1-002_pc_gfx.obj-1", glm::translate(glm::mat4(1.0f), glm::vec3(0.0f, 0.3f, 0.0f))));
	asm_seq._steps.at(2) = refModels;
	refModels.clear();

	refModels.push_back(make_pair<string, glm::mat4>("N1-003_pc_gfx.obj", glm::translate(glm::vec3(0.0f, 0.0f, 0.0f))));
	asm_seq._steps.at(3) = refModels;
	refModels.clear();

	refModels.push_back(make_pair<string, glm::mat4>("N1-001_pc_gfx.obj", glm::translate(glm::mat4(1.0f), glm::vec3(0.0f, 0.0f, 0.12f))));
	refModels.push_back(make_pair<string, glm::mat4>("N1-002_pc_gfx.obj", glm::translate(glm::vec3(0.0f, 0.0f, 0.0f))));
	refModels.push_back(make_pair<string, glm::mat4>("N1-002_pc_gfx.obj-1", glm::translate(glm::mat4(1.0f), glm::vec3(0.0f, 0.3f, 0.0f))));
	refModels.push_back(make_pair<string, glm::mat4>("N1-003_pc_gfx.obj", glm::rotate(glm::translate(glm::vec3(0.0f, -0.1f, 0.0f)), (float)-PI / 2, glm::vec3(0, 0, 1))));
	asm_seq._steps.at(4) = refModels;
	refModels.clear();

	refModels.push_back(make_pair<string, glm::mat4>("N0-000_pc_gfx.obj", glm::translate(glm::vec3(0.0f, 0.0f, 0.0f))));
	refModels.push_back(make_pair<string, glm::mat4>("N1-001_pc_gfx.obj", glm::rotate(glm::translate(glm::vec3(0.5f, -0.3f, -0.12f)), (float)PI / 2, glm::vec3(0, 0, 1))));
	refModels.push_back(make_pair<string, glm::mat4>("N1-002_pc_gfx.obj", glm::rotate(glm::translate(glm::vec3(0.5f, -0.3f, -0.24f)), (float)PI / 2, glm::vec3(0, 0, 1))));
	refModels.push_back(make_pair<string, glm::mat4>("N1-002_pc_gfx.obj-1", glm::rotate(glm::translate(glm::vec3(0.2f, -0.3f, -0.24f)), (float)PI / 2, glm::vec3(0, 0, 1))));
	refModels.push_back(make_pair<string, glm::mat4>("N1-003_pc_gfx.obj", glm::translate(glm::vec3(0.6f, -0.3f, -0.24f))));
	asm_seq._steps.at(5) = refModels;
	refModels.clear();

	/*
	Phase 2
	*/
	refModels.push_back(make_pair<string, glm::mat4>("N4-001_pc_gfx.obj", glm::translate(glm::vec3(0.0f, 0.0f, 0.0f))));
	asm_seq._steps.at(6) = refModels;
	refModels.clear();

	refModels.push_back(make_pair<string, glm::mat4>("N4-002_pc_gfx.obj", glm::translate(glm::vec3(0.0f, 0.0f, 0.0f))));
	asm_seq._steps.at(7) = refModels;
	refModels.clear();

	refModels.push_back(make_pair<string, glm::mat4>("N4-001_pc_gfx.obj", glm::translate(glm::vec3(0.0f, 0.0f, 0.0f))));
	refModels.push_back(make_pair<string, glm::mat4>("N4-002_pc_gfx.obj", glm::translate(glm::vec3(0.0f, -0.15f, 0.0f))));
	asm_seq._steps.at(8) = refModels;
	refModels.clear();

	refModels.push_back(make_pair<string, glm::mat4>("N3-002_pc_gfx.obj", glm::translate(glm::vec3(0.0f, 0.0f, 0.0f))));
	asm_seq._steps.at(9) = refModels;
	refModels.clear();

	refModels.push_back(make_pair<string, glm::mat4>("N4-001_pc_gfx.obj", glm::translate(glm::vec3(0.0f, 0.0f, 0.0f))));
	refModels.push_back(make_pair<string, glm::mat4>("N4-002_pc_gfx.obj", glm::translate(glm::vec3(0.0f, -0.15f, 0.0f))));
	refModels.push_back(make_pair<string, glm::mat4>("N3-002_pc_gfx.obj", glm::rotate(glm::translate(glm::vec3(0.0f, -0.25f, 0.0f)), (float)-PI / 2, glm::vec3(0, 0, 1))));
	asm_seq._steps.at(10) = refModels;
	refModels.clear();

	refModels.push_back(make_pair<string, glm::mat4>("N4-004_pc_gfx.obj", glm::translate(glm::vec3(0.0f, 0.0f, 0.0f))));
	asm_seq._steps.at(11) = refModels;
	refModels.clear();

	refModels.push_back(make_pair<string, glm::mat4>("N4-001_pc_gfx.obj", glm::translate(glm::vec3(0.0f, 0.0f, 0.0f))));
	refModels.push_back(make_pair<string, glm::mat4>("N4-002_pc_gfx.obj", glm::translate(glm::vec3(0.0f, -0.15f, 0.0f))));
	refModels.push_back(make_pair<string, glm::mat4>("N3-002_pc_gfx.obj", glm::rotate(glm::translate(glm::vec3(0.0f, -0.25f, 0.0f)), (float)-PI / 2, glm::vec3(0, 0, 1))));
	refModels.push_back(make_pair<string, glm::mat4>("N4-004_pc_gfx.obj", glm::translate(glm::vec3(0.0f, 0.0f, 0.0f))));
	asm_seq._steps.at(12) = refModels;
	refModels.clear();

	refModels.push_back(make_pair<string, glm::mat4>("N4-003_pc_gfx.obj", glm::translate(glm::vec3(0.0f, 0.0f, 0.0f))));
	asm_seq._steps.at(13) = refModels;
	refModels.clear();

	refModels.push_back(make_pair<string, glm::mat4>("N4-001_pc_gfx.obj", glm::translate(glm::vec3(0.0f, 0.0f, 0.0f))));
	refModels.push_back(make_pair<string, glm::mat4>("N4-002_pc_gfx.obj", glm::translate(glm::vec3(0.0f, -0.15f, 0.0f))));
	refModels.push_back(make_pair<string, glm::mat4>("N3-002_pc_gfx.obj", glm::rotate(glm::translate(glm::vec3(0.0f, -0.25f, 0.0f)), (float)-PI / 2, glm::vec3(0, 0, 1))));
	refModels.push_back(make_pair<string, glm::mat4>("N4-004_pc_gfx.obj", glm::translate(glm::vec3(0.0f, 0.0f, 0.0f))));
	refModels.push_back(make_pair<string, glm::mat4>("N4-003_pc_gfx.obj", glm::translate(glm::vec3(0.0f, 0.15f, 0.0f))));
	asm_seq._steps.at(14) = refModels;
	refModels.clear();

	refModels.push_back(make_pair<string, glm::mat4>("N0-000_pc_gfx.obj", glm::translate(glm::vec3(0.0f, 0.0f, 0.0f))));
	refModels.push_back(make_pair<string, glm::mat4>("N1-001_pc_gfx.obj", glm::rotate(glm::translate(glm::vec3(0.5f, -0.3f, -0.12f)), (float)PI / 2, glm::vec3(0, 0, 1))));
	refModels.push_back(make_pair<string, glm::mat4>("N1-002_pc_gfx.obj", glm::rotate(glm::translate(glm::vec3(0.5f, -0.3f, -0.24f)), (float)PI / 2, glm::vec3(0, 0, 1))));
	refModels.push_back(make_pair<string, glm::mat4>("N1-002_pc_gfx.obj-1", glm::rotate(glm::translate(glm::vec3(0.2f, -0.3f, -0.24f)), (float)PI / 2, glm::vec3(0, 0, 1))));
	refModels.push_back(make_pair<string, glm::mat4>("N1-003_pc_gfx.obj", glm::translate(glm::vec3(0.6f, -0.3f, -0.24f))));

	refModels.push_back(make_pair<string, glm::mat4>("N4-001_pc_gfx.obj", glm::rotate(glm::translate(glm::vec3(-0.4f, -0.3f, -0.24f)), (float)-PI / 2, glm::vec3(0, 0, 1))));
	refModels.push_back(make_pair<string, glm::mat4>("N4-002_pc_gfx.obj", glm::rotate(glm::translate(glm::vec3(-0.15f, -0.3f, -0.24f)), (float)-PI / 2, glm::vec3(0, 0, 1))));
	refModels.push_back(make_pair<string, glm::mat4>("N3-002_pc_gfx.obj", glm::rotate(glm::translate(glm::vec3(-0.65f, -0.3f, -0.24f)), (float)-PI, glm::vec3(0, 0, 1))));
	refModels.push_back(make_pair<string, glm::mat4>("N4-003_pc_gfx.obj", glm::rotate(glm::translate(glm::vec3(-0.45f, -0.3f, -0.24f)), (float)-PI / 2, glm::vec3(0, 0, 1))));
	refModels.push_back(make_pair<string, glm::mat4>("N4-004_pc_gfx.obj", glm::rotate(glm::translate(glm::vec3(-0.3f, -0.3f, -0.24f)), (float)-PI / 2, glm::vec3(0, 0, 1))));
	asm_seq._steps.at(15) = refModels;
	refModels.clear();

	/*
	Phase 3
	*/
	refModels.push_back(make_pair<string, glm::mat4>("N2-001_pc_gfx.obj", glm::translate(glm::vec3(0.0f, 0.0f, 0.19f))));
	asm_seq._steps.at(16) = refModels;
	refModels.clear();

	refModels.push_back(make_pair<string, glm::mat4>("N3-001_pc_gfx.obj", glm::mat4(1.0)));
	asm_seq._steps.at(17) = refModels;
	refModels.clear();

	refModels.push_back(make_pair<string, glm::mat4>("N2-001_pc_gfx.obj", glm::translate(glm::vec3(0.0f, 0.0f, 0.19f))));
	refModels.push_back(make_pair<string, glm::mat4>("N3-001_pc_gfx.obj", glm::translate(glm::vec3(0.0f, 0.3f, 0.0f))));
	asm_seq._steps.at(18) = refModels;
	refModels.clear();

	refModels.push_back(make_pair<string, glm::mat4>("N3-002_pc_gfx.obj-1", glm::mat4(1.0)));
	asm_seq._steps.at(19) = refModels;
	refModels.clear();

	refModels.push_back(make_pair<string, glm::mat4>("N2-001_pc_gfx.obj", glm::translate(glm::vec3(0.0f, 0.0f, 0.19f))));
	refModels.push_back(make_pair<string, glm::mat4>("N3-001_pc_gfx.obj", glm::translate(glm::vec3(0.0f, 0.3f, 0.0f))));
	refModels.push_back(make_pair<string, glm::mat4>("N3-002_pc_gfx.obj-1", glm::rotate(glm::translate(glm::vec3(0.0f, 0.6f, 0.0f)), (float)PI / 2, glm::vec3(0, 0, 1))));
	asm_seq._steps.at(20) = refModels;
	refModels.clear();

	refModels.push_back(make_pair<string, glm::mat4>("N2-001_pc_gfx.obj", glm::translate(glm::vec3(0.0f, 0.0f, 0.19f))));
	refModels.push_back(make_pair<string, glm::mat4>("N2-002_pc_gfx.obj", glm::translate(glm::vec3(0.0f, -0.55f, 0.0f))));
	refModels.push_back(make_pair<string, glm::mat4>("N3-001_pc_gfx.obj", glm::translate(glm::vec3(0.0f, 0.3f, 0.0f))));
	refModels.push_back(make_pair<string, glm::mat4>("N3-002_pc_gfx.obj-1", glm::rotate(glm::translate(glm::vec3(0.0f, 0.6f, 0.0f)), (float)PI / 2, glm::vec3(0, 0, 1))));
	asm_seq._steps.at(21) = refModels;
	refModels.clear();

	refModels.push_back(make_pair<string, glm::mat4>("N2-001_pc_gfx.obj", glm::translate(glm::vec3(0.0f, 0.0f, 0.19f))));
	refModels.push_back(make_pair<string, glm::mat4>("N2-002_pc_gfx.obj", glm::translate(glm::vec3(0.0f, -0.55f, 0.0f))));
	refModels.push_back(make_pair<string, glm::mat4>("N2-003_pc_gfx.obj", glm::translate(glm::vec3(0.0f, -0.65f, 0.0f))));
	refModels.push_back(make_pair<string, glm::mat4>("N3-001_pc_gfx.obj", glm::translate(glm::vec3(0.0f, 0.3f, 0.0f))));
	refModels.push_back(make_pair<string, glm::mat4>("N3-002_pc_gfx.obj-1", glm::rotate(glm::translate(glm::vec3(0.0f, 0.6f, 0.0f)), (float)PI / 2, glm::vec3(0, 0, 1))));
	asm_seq._steps.at(22) = refModels;
	refModels.clear();

	refModels.push_back(make_pair<string, glm::mat4>("N0-000_pc_gfx.obj", glm::translate(glm::vec3(0.0f, 0.0f, 0.0f))));
	refModels.push_back(make_pair<string, glm::mat4>("N1-001_pc_gfx.obj", glm::rotate(glm::translate(glm::vec3(0.5f, -0.3f, -0.12f)), (float)PI / 2, glm::vec3(0, 0, 1))));
	refModels.push_back(make_pair<string, glm::mat4>("N1-002_pc_gfx.obj", glm::rotate(glm::translate(glm::vec3(0.5f, -0.3f, -0.24f)), (float)PI / 2, glm::vec3(0, 0, 1))));
	refModels.push_back(make_pair<string, glm::mat4>("N1-002_pc_gfx.obj-1", glm::rotate(glm::translate(glm::vec3(0.2f, -0.3f, -0.24f)), (float)PI / 2, glm::vec3(0, 0, 1))));
	refModels.push_back(make_pair<string, glm::mat4>("N1-003_pc_gfx.obj", glm::translate(glm::vec3(0.6f, -0.3f, -0.24f))));

	refModels.push_back(make_pair<string, glm::mat4>("N4-001_pc_gfx.obj", glm::rotate(glm::translate(glm::vec3(-0.4f, -0.3f, -0.24f)), (float)-PI / 2, glm::vec3(0, 0, 1))));
	refModels.push_back(make_pair<string, glm::mat4>("N4-002_pc_gfx.obj", glm::rotate(glm::translate(glm::vec3(-0.15f, -0.3f, -0.24f)), (float)-PI / 2, glm::vec3(0, 0, 1))));
	refModels.push_back(make_pair<string, glm::mat4>("N3-002_pc_gfx.obj", glm::rotate(glm::translate(glm::vec3(-0.65f, -0.3f, -0.24f)), (float)-PI, glm::vec3(0, 0, 1))));
	refModels.push_back(make_pair<string, glm::mat4>("N4-003_pc_gfx.obj", glm::rotate(glm::translate(glm::vec3(-0.45f, -0.3f, -0.24f)), (float)-PI / 2, glm::vec3(0, 0, 1))));
	refModels.push_back(make_pair<string, glm::mat4>("N4-004_pc_gfx.obj", glm::rotate(glm::translate(glm::vec3(-0.3f, -0.3f, -0.24f)), (float)-PI / 2, glm::vec3(0, 0, 1))));

	refModels.push_back(make_pair<string, glm::mat4>("N2-001_pc_gfx.obj", glm::rotate(glm::translate(glm::vec3(-0.25f, -0.3f, 0.43f)), (float)PI / 2, glm::vec3(0, 0, 1))));
	refModels.push_back(make_pair<string, glm::mat4>("N2-002_pc_gfx.obj", glm::rotate(glm::translate(glm::vec3(0.33f, -0.3f, 0.25f)), (float)PI / 2, glm::vec3(0, 0, 1))));
	refModels.push_back(make_pair<string, glm::mat4>("N2-003_pc_gfx.obj", glm::rotate(glm::translate(glm::vec3(0.48f, -0.3f, 0.25f)), (float)PI / 2, glm::vec3(0, 0, 1))));
	refModels.push_back(make_pair<string, glm::mat4>("N3-001_pc_gfx.obj", glm::rotate(glm::translate(glm::vec3(-0.55f, -0.3f, 0.25f)), (float)PI / 2, glm::vec3(0, 0, 1))));
	refModels.push_back(make_pair<string, glm::mat4>("N3-002_pc_gfx.obj-1", glm::rotate(glm::translate(glm::vec3(-0.65f, -0.3f, 0.25f)), (float)PI, glm::vec3(0, 0, 1))));
	asm_seq._steps.at(23) = refModels;
	refModels.clear();
}