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

	Step firstStep = _procedure._steps.at(_steps.at(0));

	if (firstStep.is_subproc)
	{
		_procedure._subprocs->at(firstStep.model_name)._steps.at(_steps.at(1)).completed = true;
		_current_subproc = firstStep.model_name;
	}
	else
	{
		firstStep.completed = true;
		_current_subproc = "NULL";
	}
}

void ProcedureRenderer::progress(bool forward)
{
		if (forward)
		{
			_current_state++;

			//If the sequence wraps
			if (_current_state >= _steps.size())
			{
				_current_state = 0;

				for (std::string step_name : _steps)
				{
					_procedure._steps.at(step_name).completed = false;
				}
			}

			Step curStep = _procedure._steps.at(_steps.at(_current_state));

			//If switching to a subprocedure
			if (curStep.is_subproc)
			{
				for (int i = 0; i < _current_state; i++)
				{
					_procedure._steps.at(_steps.at(i)).completed = false;
				}

				_current_subproc = curStep.model_name;

				_current_state++;
				curStep = _procedure._steps.at(_steps.at(_current_state));
			}

			//If switching back to main
			if (_current_subproc.compare("NULL") != 0 && !curStep.is_subproc)
			{
				for (int i = 0; i < _current_state; i++)
				{
					_procedure._steps.at(_steps.at(i)).completed = true;
				}
			}

			curStep.completed = true;
		}
		else
		{
			_procedure._steps.at(_steps.at(_current_state)).completed = false;

			_current_state--;

			//If the sequence wraps
			if (_current_state < 0)
			{
				_current_state = _steps.size() - 1;

				for (std::string step_name : _steps)
				{
					_procedure._steps.at(step_name).completed = true;
				}
			}

			Step curStep = _procedure._steps.at(_steps.at(_current_state));

			//If going back into a subprocedure

			//TODO: Finish subprocedure conditions
		}
}


float mult_factor = (PI / 180);

void ProcedureRenderer::draw(glm::mat4 proj, glm::mat4 vm)
{

	for (auto step = _procedure._steps.begin(); step != _procedure._steps.end(); step++)
	{
		Step cur = step->second;

		if (cur.completed == true)
		{
			//Generate model matrix
			glm::vec3 rotRad = cur.rot * mult_factor;
			glm::mat4 m_mat = glm::translate(cur.trans) *
				glm::rotate(rotRad.x, glm::vec3(1, 0, 0)) *
				glm::rotate(rotRad.y, glm::vec3(0, 1, 0)) *
				glm::rotate(rotRad.z, glm::vec3(0, 0, 1));

			_procedure._models->at(step->second.model_name).draw(proj, vm, m_mat);
		}
	}

}