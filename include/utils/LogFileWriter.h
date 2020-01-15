#pragma once

#define _CRT_SECURE_NO_WARNINGS
#define _SILENCE_EXPERIMENTAL_FILESYSTEM_DEPRECATION_WARNING

#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <fstream>

#if _MSVC_LANG==201703L
#include <filesystem>
#else
#include <experimental/filesystem>
#endif

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
	string strFilename;
	bool bWriteLog;

public:
	LogFileWriter(string path);
	LogFileWriter();
	~LogFileWriter();
	void enableLog(bool makeOpen);
	bool addLog(string log);
	bool fileExists(string pathname);
};

/**
Log: the normal user's version of a LogFileWriter
*/
class Log
{
private:
	LogFileWriter* lfwMyLog;
	string logName;
public:
	Log(string strlogName);
	bool write(string log);
	void enableLog(bool open);
};

/**
LogAdmin: a system administrator's version
	of a LogFileWriter.
*/
class LogAdmin {
private:
	static bool isAdmin();
public:
	static bool startNewLog(string logName);
};