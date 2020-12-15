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


/*
TODO: Set the positions for all of the elements when they are put into the gearbox frame.
*/

/*
Phase 1
*/
void stage00(Sequence& sequence)
{
	setModelPos("N1-001_pc_gfx.obj", sequence, glm::translate(glm::mat4(1.0f), glm::vec3(0.0f, 0.0f, 0.12f)));
	setModelVisible("N1-001_pc_gfx.obj", sequence);
}
void stage01(Sequence& sequence)
{
	setModelPos("N1-002_pc_gfx.obj", sequence, glm::translate(glm::vec3(0.0f, 0.0f, 0.0f)));
	setModelVisible("N1-002_pc_gfx.obj", sequence);
}
void stage02(Sequence& sequence)
{
	setModelPos("N1-001_pc_gfx.obj", sequence, glm::translate(glm::mat4(1.0f), glm::vec3(0.0f, 0.0f, 0.12f)));
	setModelPos("N1-002_pc_gfx.obj-1", sequence, glm::translate(glm::mat4(1.0f), glm::vec3(0.0f, 0.3f, 0.0f)));
	setModelVisible("N1-001_pc_gfx.obj", sequence);
	setModelVisible("N1-002_pc_gfx.obj", sequence);
	setModelVisible("N1-002_pc_gfx.obj-1", sequence);
}
void stage03(Sequence& sequence)
{
	setModelPos("N1-001_pc_gfx.obj", sequence, glm::translate(glm::mat4(1.0f), glm::vec3(0.0f, 0.0f, 0.12f)));
	setModelPos("N1-002_pc_gfx.obj-1", sequence, glm::translate(glm::mat4(1.0f), glm::vec3(0.0f, 0.3f, 0.0f)));
	setModelPos("N1-003_pc_gfx.obj", sequence, glm::translate(glm::vec3(0.0f, 0.0f, 0.0f)));
	setModelVisible("N1-003_pc_gfx.obj", sequence);
}
void stage04(Sequence& sequence)
{
	setModelPos("N1-001_pc_gfx.obj", sequence, glm::translate(glm::mat4(1.0f), glm::vec3(0.0f, 0.0f, 0.12f)));
	setModelPos("N1-002_pc_gfx.obj-1", sequence, glm::translate(glm::mat4(1.0f), glm::vec3(0.0f, 0.3f, 0.0f)));
	setModelPos("N1-003_pc_gfx.obj", sequence, glm::rotate(glm::translate(glm::vec3(0.0f, -0.1f, 0.0f)), (float)-PI / 2, glm::vec3(0, 0, 1)));
	setModelVisible("N1-001_pc_gfx.obj", sequence);
	setModelVisible("N1-002_pc_gfx.obj", sequence);
	setModelVisible("N1-002_pc_gfx.obj-1", sequence);
	setModelVisible("N1-003_pc_gfx.obj", sequence);
}


void stage05(Sequence& sequence)
{
	setModelPos("N1-001_pc_gfx.obj", sequence, glm::rotate(glm::translate(glm::vec3(0.3f, -0.1f, 0.2f)), (float)PI / 2, glm::vec3(0, 0, 1)));
	setModelPos("N1-002_pc_gfx.obj-1", sequence, glm::translate(glm::vec3(0.0f, 0.3f, 0.0f)));
	setModelPos("N1-003_pc_gfx.obj", sequence, glm::rotate(glm::translate(glm::vec3(0.0f, -0.1f, 0.0f)), (float)-PI / 2, glm::vec3(0, 0, 1)));
	setModelVisible("N0-000_pc_gfx.obj", sequence);
	setModelVisible("N1-001_pc_gfx.obj", sequence);
	setModelVisible("N1-002_pc_gfx.obj", sequence);
	setModelVisible("N1-002_pc_gfx.obj-1", sequence);
	setModelVisible("N1-003_pc_gfx.obj", sequence);
}

/*
Phase 2
*/
void stage06(Sequence& sequence)
{
	setModelVisible("N4-001_pc_gfx.obj", sequence);
}
void stage07(Sequence& sequence)
{
	setModelVisible("N4-002_pc_gfx.obj", sequence);
}
void stage08(Sequence& sequence)
{
	setModelPos("N4-002_pc_gfx.obj", sequence, glm::translate(glm::vec3(0.0f, -0.15f, 0.0f)));
	setModelVisible("N4-001_pc_gfx.obj", sequence);
	setModelVisible("N4-002_pc_gfx.obj", sequence);
}
void stage09(Sequence& sequence)
{
	setModelPos("N3-002_pc_gfx.obj", sequence, glm::translate(glm::vec3(0.0f, 0.0f, 0.0f)));
	setModelVisible("N3-002_pc_gfx.obj", sequence);
}
void stage10(Sequence& sequence)
{
	setModelPos("N4-002_pc_gfx.obj", sequence, glm::translate(glm::vec3(0.0f, -0.15f, 0.0f)));
	setModelPos("N3-002_pc_gfx.obj", sequence, glm::rotate(glm::translate(glm::vec3(0.0f, -0.25f, 0.0f)), (float)-PI / 2, glm::vec3(0, 0, 1)));
	setModelVisible("N4-001_pc_gfx.obj", sequence);
	setModelVisible("N4-002_pc_gfx.obj", sequence);
	setModelVisible("N3-002_pc_gfx.obj", sequence);
}
void stage11(Sequence& sequence)
{
	setModelVisible("N4-004_pc_gfx.obj", sequence);
}
void stage12(Sequence& sequence)
{
	setModelPos("N4-002_pc_gfx.obj", sequence, glm::translate(glm::vec3(0.0f, -0.15f, 0.0f)));
	setModelPos("N3-002_pc_gfx.obj", sequence, glm::rotate(glm::translate(glm::vec3(0.0f, -0.25f, 0.0f)), (float)-PI / 2, glm::vec3(0, 0, 1)));
	setModelVisible("N4-004_pc_gfx.obj", sequence);
	setModelVisible("N4-001_pc_gfx.obj", sequence);
	setModelVisible("N4-002_pc_gfx.obj", sequence);
	setModelVisible("N3-002_pc_gfx.obj", sequence);
}
void stage13(Sequence& sequence)
{
	setModelPos("N4-003_pc_gfx.obj", sequence, glm::translate(glm::vec3(0.0f, 0.0f, 0.0f)));
	setModelVisible("N4-003_pc_gfx.obj", sequence);
}
void stage14(Sequence& sequence)
{
	setModelPos("N4-002_pc_gfx.obj", sequence, glm::translate(glm::vec3(0.0f, -0.15f, 0.0f)));
	setModelPos("N3-002_pc_gfx.obj", sequence, glm::rotate(glm::translate(glm::vec3(0.0f, -0.25f, 0.0f)), (float)-PI / 2, glm::vec3(0, 0, 1)));
	setModelPos("N4-003_pc_gfx.obj", sequence, glm::translate(glm::vec3(0.0f, 0.15f, 0.0f)));
	setModelVisible("N4-003_pc_gfx.obj", sequence);
	setModelVisible("N4-004_pc_gfx.obj", sequence);
	setModelVisible("N4-001_pc_gfx.obj", sequence);
	setModelVisible("N4-002_pc_gfx.obj", sequence);
	setModelVisible("N3-002_pc_gfx.obj", sequence);
}


void stage15(Sequence& sequence)
{
	setModelVisible("N0-000_pc_gfx.obj", sequence);
	setModelVisible("N1-001_pc_gfx.obj", sequence);
	setModelVisible("N1-002_pc_gfx.obj", sequence);
	setModelVisible("N1-002_pc_gfx.obj-1", sequence);
	setModelVisible("N1-003_pc_gfx.obj", sequence);
	setModelVisible("N4-003_pc_gfx.obj", sequence);
	setModelVisible("N4-004_pc_gfx.obj", sequence);
	setModelVisible("N4-001_pc_gfx.obj", sequence);
	setModelVisible("N4-002_pc_gfx.obj", sequence);
	setModelVisible("N3-002_pc_gfx.obj", sequence);
}

/*
Phase 3
*/
void stage16(Sequence& sequence)
{
	setModelPos("N2-001_pc_gfx.obj", sequence, glm::translate(glm::vec3(0.0f, 0.0f, 0.19f)));
	setModelVisible("N2-001_pc_gfx.obj", sequence);
}
void stage17(Sequence& sequence)
{
	setModelPos("N3-001_pc_gfx.obj", sequence, glm::mat4(1.0));
	setModelVisible("N3-001_pc_gfx.obj", sequence);
}
void stage18(Sequence& sequence)
{
	setModelPos("N2-001_pc_gfx.obj", sequence, glm::translate(glm::vec3(0.0f, 0.0f, 0.19f)));
	setModelPos("N3-001_pc_gfx.obj", sequence, glm::translate(glm::vec3(0.0f, 0.3f, 0.0f)));
	setModelVisible("N2-001_pc_gfx.obj", sequence);
	setModelVisible("N3-001_pc_gfx.obj", sequence);
}
void stage19(Sequence& sequence)
{
	setModelPos("N2-001_pc_gfx.obj", sequence, glm::translate(glm::vec3(0.0f, 0.0f, 0.19f)));
	setModelPos("N3-001_pc_gfx.obj", sequence, glm::translate(glm::vec3(0.0f, 0.3f, 0.0f)));
	setModelPos("N3-002_pc_gfx.obj-1", sequence, glm::mat4(1.0));
	setModelVisible("N3-002_pc_gfx.obj-1", sequence);
}
void stage20(Sequence& sequence)
{
	setModelPos("N2-001_pc_gfx.obj", sequence, glm::translate(glm::vec3(0.0f, 0.0f, 0.19f)));
	setModelPos("N3-001_pc_gfx.obj", sequence, glm::translate(glm::vec3(0.0f, 0.3f, 0.0f)));
	setModelPos("N3-002_pc_gfx.obj-1", sequence, glm::rotate(glm::translate(glm::vec3(0.0f, 0.6f, 0.0f)), (float)PI / 2, glm::vec3(0, 0, 1)));
	setModelVisible("N2-001_pc_gfx.obj", sequence);
	setModelVisible("N3-001_pc_gfx.obj", sequence);
	setModelVisible("N3-002_pc_gfx.obj-1", sequence);
}
void stage21(Sequence& sequence)
{
	setModelPos("N2-001_pc_gfx.obj", sequence, glm::translate(glm::vec3(0.0f, 0.0f, 0.19f)));
	setModelPos("N3-001_pc_gfx.obj", sequence, glm::translate(glm::vec3(0.0f, 0.3f, 0.0f)));
	setModelPos("N3-002_pc_gfx.obj-1", sequence, glm::rotate(glm::translate(glm::vec3(0.0f, 0.6f, 0.0f)), (float)PI / 2, glm::vec3(0, 0, 1)));
	setModelPos("N2-002_pc_gfx.obj", sequence, glm::translate(glm::vec3(0.0f, -0.55f, 0.0f)));
	setModelVisible("N2-001_pc_gfx.obj", sequence);
	setModelVisible("N2-002_pc_gfx.obj", sequence);
	setModelVisible("N3-001_pc_gfx.obj", sequence);
	setModelVisible("N3-002_pc_gfx.obj-1", sequence);
}
void stage22(Sequence& sequence)
{
	setModelPos("N2-001_pc_gfx.obj", sequence, glm::translate(glm::vec3(0.0f, 0.0f, 0.19f)));
	setModelPos("N3-001_pc_gfx.obj", sequence, glm::translate(glm::vec3(0.0f, 0.3f, 0.0f)));
	setModelPos("N3-002_pc_gfx.obj-1", sequence, glm::rotate(glm::translate(glm::vec3(0.0f, 0.6f, 0.0f)), (float)PI / 2, glm::vec3(0, 0, 1)));
	setModelPos("N2-002_pc_gfx.obj", sequence, glm::translate(glm::vec3(0.0f, -0.55f, 0.0f)));
	setModelPos("N2-003_pc_gfx.obj", sequence, glm::translate(glm::vec3(0.0f, -0.65f, 0.0f)));
	setModelVisible("N2-001_pc_gfx.obj", sequence);
	setModelVisible("N2-002_pc_gfx.obj", sequence);
	setModelVisible("N2-003_pc_gfx.obj", sequence);
	setModelVisible("N3-001_pc_gfx.obj", sequence);
	setModelVisible("N3-002_pc_gfx.obj-1", sequence);
}


void stage23(Sequence& sequence)
{
	setModelVisible("N0-000_pc_gfx.obj", sequence);
	setModelVisible("N1-001_pc_gfx.obj", sequence);
	setModelVisible("N1-002_pc_gfx.obj", sequence);
	setModelVisible("N1-002_pc_gfx.obj-1", sequence);
	setModelVisible("N1-003_pc_gfx.obj", sequence);
	setModelVisible("N4-003_pc_gfx.obj", sequence);
	setModelVisible("N4-004_pc_gfx.obj", sequence);
	setModelVisible("N4-001_pc_gfx.obj", sequence);
	setModelVisible("N4-002_pc_gfx.obj", sequence);
	setModelVisible("N3-002_pc_gfx.obj", sequence);
	setModelVisible("N2-001_pc_gfx.obj", sequence);
	setModelVisible("N2-002_pc_gfx.obj", sequence);
	setModelVisible("N2-003_pc_gfx.obj", sequence);
	setModelVisible("N3-001_pc_gfx.obj", sequence);
	setModelVisible("N3-002_pc_gfx.obj", sequence);
	setModelVisible("N3-002_pc_gfx.obj-1", sequence);
}

void AssemblySequence::update(Sequence& sequence)
{
	for (auto i = sequence._models.begin(); i != sequence._models.end(); i++)
	{
		i->second->visible = false;
	}

	switch (sequence._current_state)
	{
	case 0:
		stage00(sequence);
		break;
	case 1:
		stage01(sequence);
		break;
	case 2:
		stage02(sequence);
		break;
	case 3:
		stage03(sequence);
		break;
	case 4:
		stage04(sequence);
		break;
	case 5:
		stage05(sequence);
		break;
	case 6:
		stage06(sequence);
		break;
	case 7:
		stage07(sequence);
		break;
	case 8:
		stage08(sequence);
		break;
	case 9:
		stage09(sequence);
		break;
	case 10:
		stage10(sequence);
		break;
	case 11:
		stage11(sequence);
		break;
	case 12:
		stage12(sequence);
		break;
	case 13:
		stage13(sequence);
		break;
	case 14:
		stage14(sequence);
		break;
	case 15:
		stage15(sequence);
		break;
	case 16:
		stage16(sequence);
		break;
	case 17:
		stage17(sequence);
		break;
	case 18:
		stage18(sequence);
		break;
	case 19:
		stage19(sequence);
		break;
	case 20:
		stage20(sequence);
		break;
	case 21:
		stage21(sequence);
		break;
	case 22:
		stage22(sequence);
		break;
	case 23:
		stage23(sequence);
		break;
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
	if (sequence._forward[sequence._current_state] > 23) return false;
	sequence._current_state = sequence._forward[sequence._current_state];

	update(sequence);

	return true;
}

//static
bool AssemblySequence::prevStage(Sequence& sequence)
{
	if (sequence._backward[sequence._current_state] < 0) return false;
	sequence._current_state = sequence._backward[sequence._current_state];

	update(sequence);

	return true;
}

//static
void AssemblySequence::setSeq(std::vector<int> order, Sequence& sequence)
{
	delete[] sequence._forward;
	delete[] sequence._backward;

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
