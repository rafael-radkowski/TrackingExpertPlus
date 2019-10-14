#include "TimeUtils.h"

using namespace texpert; 


//const 
std::string TimeUtils::GetCurrentDateTime()
{
	time_t     now = time(0);
	struct tm  tstruct;
	char       buf[80];
	tstruct = *localtime(&now);

	// Visit http://en.cppreference.com/w/cpp/chrono/c/strftime
	// for more information about date/time format
	strftime(buf, sizeof(buf), "%Y-%m-%d_%I-%M-%S", &tstruct);

	return buf;
}
