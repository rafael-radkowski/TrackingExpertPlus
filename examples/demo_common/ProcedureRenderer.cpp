#include "ProcedureRenderer.h"

#define PI 3.14159265

ProcedureRenderer::ProcedureRenderer()
{
}

void ProcedureRenderer::init(const std::string path_and_name, std::vector<std::string>& order)
{
	_procedure = Procedure();
	ProcedureLoader::loadProcedure(path_and_name, _procedure);

	_current_state = 0;
	_steps = order;

	Step firstStep = _procedure.steps.at(_steps.at(0));

	//if (firstStep.subproc.compare("main") != 0)
	//{
	//	_procedure.subprocs->at(firstStep.model_name).steps.at(_steps.at(1)).completed = true;
	//	_current_subproc = firstStep.model_name;
	//}
	//else
	//{
	//	firstStep.completed = true;
	//	_current_subproc = "NULL";
	//}

	_current_subproc = firstStep.subproc;
	_procedure.steps.at(_steps.at(0)).completed = true;
}

void ProcedureRenderer::progress(bool forward)
{
	if (_steps.size() == 0) return;

	if (forward)
	{
		_current_state++;

		//If the sequence wraps
		if (_current_state >= _steps.size())
		{
			_current_state = 0;

			for (std::string step_name : _steps)
			{
				_procedure.steps.at(step_name).completed = false;
			}
		}

		Step curStep = _procedure.steps.at(_steps.at(_current_state));

		//If switching to a subprocedure
		if (curStep.subproc.compare("main") != 0 && _current_subproc.compare("main") == 0)
		{
			for (int i = 0; i < _current_state; i++)
			{
				_procedure.steps.at(_steps.at(i)).completed = false;
			}
		}
		//If switching back to main
		else if (_current_subproc.compare("main") != 0 && curStep.subproc.compare("main") == 0)
		{
			for (int i = 0; i < _current_state; i++)
			{
				_procedure.steps.at(_steps.at(i)).completed = true;
			}
		}

		_current_subproc = curStep.subproc;
		_procedure.steps.at(_steps.at(_current_state)).completed = true;
	}
	else
	{
		_procedure.steps.at(_steps.at(_current_state)).completed = false;

		_current_state--;

		//If the sequence wraps
		if (_current_state < 0)
		{
			_current_state = _steps.size() - 1;

			for (std::string step_name : _steps)
			{
				_procedure.steps.at(step_name).completed = true;
			}
		}

		Step curStep = _procedure.steps.at(_steps.at(_current_state));

		//If changing subprocedures or going back into main
		if (curStep.subproc.compare(_current_subproc) != 0)
		{
			if (curStep.subproc.compare("main") != 0)
			{
				_current_subproc = curStep.subproc;
				for (int i = 0; i < _current_state; i++)
				{
					if (_procedure.steps.at(_steps.at(i)).subproc.compare(_current_subproc) != 0)
					{
						_procedure.steps.at(_steps.at(i)).completed = false;
					}
				}
			}

			else if (curStep.subproc.compare("main") == 0)
			{
				_current_subproc = curStep.subproc;
				for (int i = 0; i <= _current_state; i++)
				{
					_procedure.steps.at(_steps.at(i)).completed = true;
				}
			}
		}
	}
}


float mult_factor = (PI / 180);

void ProcedureRenderer::draw(glm::mat4 proj, glm::mat4 vm)
{

	for (auto step = _procedure.steps.begin(); step != _procedure.steps.end(); step++)
	{
		Step cur = step->second;

		if (cur.completed && _procedure.models->find(cur.model_name) != _procedure.models->end())
		{
			//Generate model matrix
			glm::vec3 rotRad = cur.rot * mult_factor;
			glm::mat4 m_mat = glm::translate(cur.trans) *
				glm::rotate(rotRad.x, glm::vec3(1, 0, 0)) *
				glm::rotate(rotRad.y, glm::vec3(0, 1, 0)) *
				glm::rotate(rotRad.z, glm::vec3(0, 0, 1));

			_procedure.models->at(step->second.model_name).draw(proj, vm, m_mat);
		}
	}

}