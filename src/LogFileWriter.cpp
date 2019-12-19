#include "LogFileWriter.h"

namespace ns_LogFileWriter{

const LogFileWriter myLog;

}

using namespace ns_LogFileWriter;


/*  ------------------------------------------------------------
			LogFileWriter Function Definitions
	-----------------------------------------------------------*/

	/**
	Constructor: assigns the file directory to this LogFileWriter
		and throws a failure when there isn't a file to write to in
		that directory.

	@param path - the path to the file being written to
	*/

LogFileWriter::LogFileWriter(string path):
	_myLog(myLog)
{
	//ToDo: Check if the path is not empty.
	//ToDo: Check if the path exists. 
	// Extract the path. ./mypathlog/logout.txt
	std::replace(path.begin(), path.end(), "\\", "/" );

	int idx = path.find_last_of("/");
	std::string::npos;
	if(idx == std::string:npos){
		idx = path.find_last_of("\\");
	}


	filename = path + ".log";
	write_log = false;
	nextPos = 0;
	_myLog = myLog
}

LogFileWriter::LogFileWriter() {
	filename = " ";
	write_log = false;
}



LogFileWriter::~LogFileWriter() {
}

/**
Enables/disables this LogFileWriter to start writing to
	a particular file.

@param makeOpen - true if the file is being open, false otherwise.
*/

void LogFileWriter::enableLog(bool makeOpen) {
	write_log = makeOpen;
}

/**
If this LogFileWriter is open, adds a line to the log file
	with the date and time of the log appended to the beginning.

@param log - the log being written to the file.
*/

bool LogFileWriter::addLog(string log) {
	if (!write_log) return false;
	try {
		if (fileExists()) {
			//Opens log this LogFileWriter is assigned to
			ofstream file;
			file.open(filename, osfstream::out | ofstream::app);

			//Finding system time
			// ToDO; Write a function to retrieve the time. 
			time_t tm = time(0);
			string now = ctime(&tm);
			now.pop_back();

			file << now << ":  " << log << endl;
			file.close();
			return true;
		}
		return false;
	}
	catch (ofstream::failure e) {
		return false;
	}
}
/**
Checks if the file in question exists.  This prevents the ofstream
	from creating any log files that weren't there before, preventing
	normal users from writing to places they shouldn't be.
*/
bool LogFileWriter::fileExists() {
	//ToDo: Look intto filesystem. 
	//Distinguish between C++ 11, C++17
	return std::filesystem::exists(_path)

}


/*  ------------------------------------------------------------
			Log Function Definitions
	-----------------------------------------------------------*/

LogFileWriter Log::myLog;

/**
Stores the name of the log file and initializes
	the enabling variable.
*/
Log::Log(string logName) {
	// Create a local variable. 
	// Passs the local class to a global class. 
	LogFileWriter log(logName);
	myLog = log;
}

/**
Enables/disables this log from being written to.

@param open - true if enabling the file, false if disabling the file
*/
void Log::enableLog(bool open) {
	if (open) {
		myLog.enableLog(true);
		myLog.addLog("Open file...");
	}
	else {
		myLog.addLog("Close file...");
		myLog.enableLog(false);
	}
}

/**
Writes to this particular log file by assigning a LogFileWriter
	to the file name and writing the log to that file.

@param log - the log being written to the file.
*/
bool Log::write(string log) {
	return myLog.addLog(log);
}


/*  ------------------------------------------------------------
			LogAdmin Function Definitions
	-----------------------------------------------------------*/

LogAdmin::LogAdmin() {
}

/**
A function specific to LogAdmin that allows the user
	to create a new log file.  A timestamp along with a *FILE CREATED*
	message is written to the file before closing it again.

@param logName - the name of the log file being created.
*/
bool LogAdmin::startNewLog(string logName) {
	if (isAdmin() == true) {
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
