
#include <chrono>
#include <vector>
#include <string>
#include "EasyTimer.h"

	EasyTimer::EasyTimer(int numTimers)
	{
		timerCount = numTimers;
		for (int i = 0; i < timerCount; i++)
		{
			startFrame.push_back(std::chrono::time_point < std::chrono::system_clock>());
			endFrame.push_back(std::chrono::time_point < std::chrono::system_clock>());
		}
	}


	void EasyTimer::startTimer(int index)
	{
		startFrame[index] = std::chrono::system_clock::now();
	}
	
	void EasyTimer::endTimer(int index)
	{
		endFrame[index] = std::chrono::system_clock::now();
	}

	void EasyTimer::print(int index, std::string message)
	{
		elapsed_seconds = endFrame[index] - startFrame[index];
		dif = elapsed_seconds.count();//calculates difference in seconds(float)
		
		if (dif > 60.0)
			printf("%s %lf minute(s).\n", message.c_str(), dif/60);
		else if (dif > 1.0)//ms
			printf("%s %lf second(s).\n", message.c_str(), dif);
		else if (dif > .001)//ms
			printf("%s %lf millisecond(s).\n", message.c_str(), dif*1000);

		else if (dif > .000001) //micro s
			printf("%s %lf microsecond(s).\n", message.c_str(), dif * 1000000);
		else //nano s
			printf("%s %lf nanosecond(s).\n", message.c_str(), dif * 1000000000);

	}
	//get difference in seconds as a double
	double EasyTimer::getDifference(int index)
	{
		elapsed_seconds = endFrame[index] - startFrame[index];
		dif = elapsed_seconds.count();//calculates difference in seconds(float)
		return dif;
	}

