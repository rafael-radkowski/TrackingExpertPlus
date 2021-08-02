#pragma once

/*!

*/

#include "Procedure.h"

#include "ProcedureLoader.h"

#include <stack>

class ProcedureRenderer
{
private:
	int _current_state;
	std::string _current_subproc;
	std::stack<std::string> _subproc_stack;
	std::vector<std::string> _steps;
	Procedure _procedure;

	void handleSubproc(Step curStep);

public:
	ProcedureRenderer();
	~ProcedureRenderer();

	void init(const std::string path_and_name, std::vector<std::string>& order);

	void progress(bool forward);

	void draw(glm::mat4 proj, glm::mat4 vm);

};