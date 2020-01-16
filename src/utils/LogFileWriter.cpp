#include "LogFileWriter.h"

#define _SILENCE_EXPERIMENTAL_FILESYSTEM_DEPRECATION_WARNING

using namespace std;

/*  ------------------------------------------------------------
			LogFileWriter Function Definitions
	-----------------------------------------------------------*/

	/**
	Constructor: assigns the file directory to this LogFileWriter
		and throws a failure when there isn't a file to write to in
		that directory.

	@param path - the path to the file being written to
	*/

string temp = "";

LogFileWriter::LogFileWriter(string strPath) {
	strFilename = strPath + ".log";
	bWriteLog = false;
}

LogFileWriter::LogFileWriter() :strFilename(temp) {
	strFilename = " ";
	bWriteLog = false;
}



LogFileWriter::~LogFileWriter() {
}

/**
Enables/disables this LogFileWriter to start writing to
	a particular file.

@param makeOpen - true if the file is being open, false otherwise.
*/

void LogFileWriter::enableLog(bool bMakeOpen) {
	bWriteLog = bMakeOpen;
}

/**
If this LogFileWriter is open, adds a line to the log file
	with the date and time of the log appended to the beginning.

@param log - the log being written to the file.
*/

bool LogFileWriter::addLog(string strLog) {
	if (!bWriteLog) return false;

	try {
		if (true) {
			//Opens log this LogFileWriter is assigned to
			ofstream osFile;
			osFile.open(strFilename, ofstream::app);


			//Finding system time
			time_t tm = time(0);
			string now = ctime(&tm);
			now.pop_back();


			osFile << now << ":  " << strLog << endl;
			osFile.close();
			return true;
		}

		return false;
	}
	catch (ofstream::failure e) {
		return false;
	}
}

bool LogFileWriter::fileExists(string pathname) {
#if _MSVC_LANG==201703L
	return std::filesystem::exists(pathname);
#else
	return std::experimental::filesystem::exists(pathname);
#endif
}



/*  ------------------------------------------------------------
			Log Function Definitions
	-----------------------------------------------------------*/

	/**
	Stores the name of the log file and initializes
		the enabling variable.
	*/

Log::Log(string strLogName) {
	lfwMyLog = &LogFileWriter(strLogName);
}

/**
Enables/disables this log from being written to.

@param open - true if enabling the file, false if disabling the file
*/
void Log::enableLog(bool bOpen) {

	LogFileWriter* lfwLog = lfwMyLog;

	if (bOpen) {
		lfwMyLog->enableLog(true);
		lfwMyLog->addLog("Open file...");
	}
	else {
		lfwMyLog->addLog("Close file...");
		lfwMyLog->enableLog(false);
	}
}

/**
Writes to this particular log file by assigning a LogFileWriter
	to the file name and writing the log to that file.

@param log - the log being written to the file.
*/
bool Log::write(string strLog) {
	LogFileWriter* lfwLog = lfwMyLog;
	return lfwMyLog->addLog(strLog);
}


/*  ------------------------------------------------------------
			LogAdmin Function Definitions
	-----------------------------------------------------------*/

	/**
	A function specific to LogAdmin that allows the user
		to create a new log file.  A timestamp along with a *FILE CREATED*
		message is written to the file before closing it again.

	@param logName - the name of the log file being created.
	*/
bool LogAdmin::startNewLog(string logName) {
		try {
			ofstream file;

			file.open(logName + ".log");

			time_t tm = time(0);

			string now = ctime(&tm);

			now.pop_back();

			file << now << ": " << "*FILE CREATED - " << logName << "*" << endl;

			file.close();
			return true;
		}
		catch (...) {
			return false;
		}
	
}

bool LogAdmin::isAdmin() {
	BOOL fIsElevated = FALSE;
	HANDLE hToken = NULL;
	TOKEN_ELEVATION elevation;
	DWORD dwSize;

	if (!OpenProcessToken(GetCurrentProcess(), TOKEN_QUERY, &hToken))
	{
		printf("\n Failed to get Process Token :%d.", GetLastError());
		goto Cleanup;  // if Failed, we treat as False
	}


	if (!GetTokenInformation(hToken, TokenElevation, &elevation, sizeof(elevation), &dwSize))
	{
		printf("\nFailed to get Token Information :%d.", GetLastError());
		goto Cleanup;// if Failed, we treat as False
	}

	fIsElevated = elevation.TokenIsElevated;

Cleanup:
	if (hToken)
	{
		CloseHandle(hToken);
		hToken = NULL;
	}
	return fIsElevated;
}