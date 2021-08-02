#pragma once

/*!

*/

#include <string>
#include <stdio.h>
#include <iostream>

#include <opencv2/core.hpp>

#include "Procedure.h"

class ProcedureLoader 
{
public:
	static bool loadProcedure(const std::string& path, Procedure& _procedure);
};