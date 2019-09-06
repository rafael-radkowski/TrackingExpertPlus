#pragma once
/*
class TimeUtils

The class returns the current time and date as a string.
Use
	TimeUtils::GetCurrentDateTime();

Features:
- Return the current time as string

Rafael Radkowski
Iowa State University
rafael@iastate.edu
+1 (515) 294 7044
Jan 2, 2015
MIT License
------------------------------------------------------
Last changes:

*/

#include <iostream>
#include <time.h>
#include <string>

using namespace std;

class TimeUtils
{
public:

	/*
	Return the current time as a string.
	@return - a string containing the current data and time 
	formated as year-month-day_hours-minutes-seconds
	*/
	static string GetCurrentDateTime(void);

}; 
