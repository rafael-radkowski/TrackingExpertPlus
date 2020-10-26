#pragma once

#include <chrono>
#include <vector>
#include <string>
/*!
@class   EasyTimer

@brief  Any easy way of timing things such as framerate, or time taken by a function


Features:
- STart or stop multiple timers at once

Tyler Ingebrand Iowa State University
tyleri@iastate.edu


*/
class EasyTimer {
private:
	int timerCount;
	//time variables
	std::vector<std::chrono::time_point<std::chrono::system_clock>> startFrame, endFrame;
	std::chrono::duration<double> elapsed_seconds;
	double dif; //difference from start to end loop in ms
public:
	/*!
	Constructor
	@param numTimers - Creates a timer with numTimers timers at once
	*/
	EasyTimer(int numTimers);
	/*!
	Starts timer at index given
	@param index - The index to start
	*/
	void startTimer(int index);
	/*!
	Stops timer at index given
	@param index - The index to stop
	*/
	void endTimer(int index);
	/*!
	Prints timer at index given with message
	@param index - The index to print
	@param message - The message to print infront of the time
	*/
	void print(int index, std::string message = "");
	/*!
	Returns change in time for given index in seconds
	@param index - The index to return time for
	@return the quantity of time
	*/
	double getDifference(int index);


};