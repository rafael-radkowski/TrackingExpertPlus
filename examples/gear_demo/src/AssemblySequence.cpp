#include "AssemblySequence.h"

namespace ns_AssemblySequence
{
	int _current_state = 0;
	int* _forward;
	int* _backward;

	std::unordered_map<string, Model*> _models;
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
void AssemblySequence::process(std::unordered_map<string, Model*> models, glm::mat4 proj, glm::mat4 vm)
{
	_models = models;

	switch (_current_state)
	{
	case 0:
		stage00();
		break;
	case 1:
		stage01();
		break;
	case 2:
		stage02();
		break;
	case 3:
		stage03();
		break;
	case 4:
		stage04();
		break;
	case 5:
		stage05();
		break;
	case 6:
		stage06();
		break;
	case 7:
		stage07();
		break;
	case 8:
		stage08();
		break;
	case 9:
		stage09();
		break;
	case 10:
		stage10();
		break;
	case 11:
		stage11();
		break;
	case 12:
		stage12();
		break;
	case 13:
		stage13();
		break;
	case 14:
		stage14();
		break;
	case 15:
		stage15();
		break;
	case 16:
		stage16();
		break;
	case 17:
		stage17();
		break;
	case 18:
		stage18();
		break;
	case 19:
		stage19();
		break;
	case 20:
		stage20();
		break;
	case 21:
		stage21();
		break;
	case 22:
		stage22();
		break;
	case 23:
		stage23();
		break;
	}

	for (auto i = _models.begin(); i != _models.end(); i++)
	{
		if (i->second->name.compare("null") != 0 && i->second->visible)
			i->second->model->draw(proj, vm);
	}
}

//static
bool AssemblySequence::nextStage()
{
	if (_forward[_current_state] > 23) return false;
	_current_state = _forward[_current_state];
	return true;
}

//static
bool AssemblySequence::prevStage()
{
	if (_backward[_current_state] < 0) return false;
	_current_state = _backward[_current_state];
	return true;
}

//static
void AssemblySequence::setSeq(std::vector<int> sequence)
{
	delete[] _forward;
	delete[] _backward;

	_forward = (int*)malloc(sequence.size() * sizeof(int));
	_backward = (int*)malloc(sequence.size() * sizeof(int));

	for (int i = 0; i < sequence.size() - 1; i++)
	{
		_forward[i] = sequence.at(i + 1);
	}
	_forward[sequence.size() - 1] = sequence.at(0);

	_backward[0] = sequence.at(sequence.size() - 1);
	for (int i = 1; i < sequence.size(); i++)
	{
		_backward[i] = sequence.at(i - 1);
	}
}
