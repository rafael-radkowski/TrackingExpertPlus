#pragma once

#define _CRT_SECURE_NO_WARNINGS

#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <fstream>
#include <filesystem>
#include <string>
#include <ctime>
#include <chrono>
#include <ShlObj.h>

using namespace std;

/**-------------------------------------------------------------
						LogFileWriter.h
	This header file allows the user to write logs to certain log
	files.  If the user has admin privelages, they are able to
	use the LogAdmin class to create a new file to write logs in.
	Otherwise, the user can only write to specific files that were
	already created.

*/

/*  ------------------------------------------------------------
							Classes
	-----------------------------------------------------------*/

	/**
	LogFileWriter: allows a user to create a file for logs to
		logs to be written in
	*/

class LogFileWriter {
private:
	string filename;
	bool write_log;
	int nextPos;
	bool fileExists();

public:
	LogFileWriter(string path);
	LogFileWriter();
	~LogFileWriter();
	void enableLog(bool makeOpen);
	bool addLog(string log);
};

/**
Log: the normal user's version of a LogFileWriter
*/
class Log
{
private:
	static LogFileWriter myLog;
public:
	Log(string logName);
	static bool write(string log);
	static void enableLog(bool open);
};

/**
LogAdmin: a system administrator's version
	of a LogFileWriter.
*/
class LogAdmin {
private:
	static bool isAdmin();
public:
	LogAdmin();
	static bool startNewLog(string logName);
};