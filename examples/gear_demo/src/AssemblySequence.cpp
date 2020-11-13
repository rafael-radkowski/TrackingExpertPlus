#include "AssemblySequence.h"

namespace ns_AssemblySequence
{
	int _current_state = 0;
	int* _forward;
	int* _backward;
}

using namespace ns_AssemblySequence;

void stage00()
{

}
void stage01()
{

}
void stage02()
{

}
void stage03()
{

}
void stage04()
{

}
void stage05()
{

}
void stage06()
{

}
void stage07()
{

}
void stage08()
{

}
void stage09()
{

}
void stage10()
{

}
void stage11()
{

}
void stage12()
{

}
void stage13()
{

}
void stage14()
{

}
void stage15()
{

}
void stage16()
{

}
void stage17()
{

}
void stage18()
{

}
void stage19()
{

}
void stage20()
{

}
void stage21()
{

}
void stage22()
{

}
void stage23()
{

}

//static
void AssemblySequence::process(std::vector<Model*> models, glm::mat4 proj, glm::mat4 vm)
{
	Model* curModel;
	for (int i = 0; i < models.size(); i++)
	{
		curModel = models.at(i);
		if (curModel->name.compare("null") != 0 && curModel->visible)
			curModel->model->draw(proj, vm);
	}
}