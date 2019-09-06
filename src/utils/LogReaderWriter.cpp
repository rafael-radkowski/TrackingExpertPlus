#include "LogReaderWriter.h"


namespace LogReaderWriter_
{
	string	_path_and_file = "";
};


using namespace isu_ar;
using namespace LogReaderWriter_;
using namespace std;

/*
Start a new log file
@param path_and_file - string with the relative or absolute file. 
@return - true if file was successfully generated. 
*/
bool LogReaderWriter::Create(string path_and_file)
{
	ExistsAndCreate(path_and_file);

	_path_and_file = path_and_file;

	std::ofstream of(_path_and_file, std::ofstream::out);

	if (of.is_open()) {
		of.close();

		WriteHeader(path_and_file);

		return true;
	}
	else {
		cout << "[ERROR] - Cannot open " << path_and_file << " to create a log file." << endl;
		return false;
	}

	
}

/*
Append a dataset to a file. 
@param data - the dataset to write 
@return - true if file was successfully generated. 
*/
bool LogReaderWriter::Write(LogData& data)
{

	if (!Exists(_path_and_file))return false;

	std::ofstream of(_path_and_file, std::ofstream::out | std::ofstream::app);

	if (!of.is_open()) {
		cout << "[ERROR] - Cannot open " << _path_and_file << " to create a log file." << endl;
		return false;

	}

	of << data.iteration << ", " << data.rms << ", " << data.votes << ", " << data.x << ", " << data.y << ", " << data.z << ", " << data.rx << ", " << data.ry << ", " << data.rz << "\n";
	of.close();

	return true;
}


/*
Read and create the nodes for the Balanced Pose Tree
@param path_and_file - string with the relative or absolute file. 
@param root - the root node of the tree. 
@param node_repository - reference to the node repository. 
*/
//static 
bool LogReaderWriter::Read(string path_and_file, std::vector<LogData>& data)
{
	return true;
}



vector<string>  LogReaderWriter::Split(string str, char delimiter)
{
	vector<string> s;

	return s;
}

/*
Write the header file
*/
bool LogReaderWriter::WriteHeader(string path_and_file)
{
	if (!Exists(path_and_file))return false;

	std::ofstream of(_path_and_file, std::ofstream::out| std::ofstream::app);

	if (!of.is_open()) {
		cout << "[ERROR] - Cannot open " << path_and_file << " to create a log file." << endl;
		return false;

	}

	of << "Feature matching evaluation log file\n";
	of << "Rafael Radkowski\n";
	of << TimeUtils::GetCurrentDateTime() << "\n";
	of.close();
	return true;
}


/*
Log all the metadata
*/
//static 
bool LogReaderWriter::WriteMetaData(LogMetaData& data)
{
	if (!Exists(_path_and_file))return false;

	std::ofstream of(_path_and_file, std::ofstream::out| std::ofstream::app);

	if (!of.is_open()) {
		cout << "[ERROR] - Cannot open " << _path_and_file << " to create a log file." << endl;
		return false;

	}

	of << "\nMETA\n";
	of << "File_ref, " << data.file_ref  << "\n";
	of << "File_test, " << data.file_test  << "\n";
	of << "N_ref, " << data.num_points_ref  << "\n";
	of << "N_test, " << data.num_points_test  << "\n";
	of << "Type, " << data.matching_type  << "\n";
	of << "Dist_step, " << data.distance_step  << "\n";
	of << "Ang_step, " << data.angle_step  << "\n";
	of << "Noise, " << data.noise  << "\n";
	of << "N_tests, " << data.N_tests  << "\n";
	of << "RMS_TH, " << data.rms_th  << "\n";
	of << "Sampling_type, " << data.sampling_type  << "\n";
	of << "Sampling_grid, " << data.sampling_grid  << "\n";


	of << "\nIdx, rms, votes, x, y, z, tx, ry, rz\n";
	of << "DATA\n";


	of.close();
}

/*
Log all the metadata
*/
//static 
bool LogReaderWriter::WriteResults(LogMetaData& data)
{
	if (!Exists(_path_and_file))return false;

	std::ofstream of(_path_and_file, std::ofstream::out| std::ofstream::app);

	if (!of.is_open()) {
		cout << "[ERROR] - Cannot open " << _path_and_file << " to create a log file." << endl;
		return false;

	}

	of << "\nRESULTS\n";
	of << "N_tests, " << data.N_tests  << "\n";
	of << "N_good, " << data.N_good  << "\n";
	of << "PR, " << double(data.N_good)/double(data.N_tests)  << "\n";
	of << "RMS_TH, " << data.rms_th  << "\n";
	of << "END_RESULTS\n";


	of.close();
}

/*
Check whether the file exits.
*/
//static 
bool LogReaderWriter::Exists(string path_and_file)
{
	if (!std::experimental::filesystem::exists(path_and_file)) {
		cout << "[ERROR] - Cannot find file " << path_and_file << "." << endl;
		return false;
	}
	return true;
}

/*
Check whether the file exits and create if if not
*/
//static 
bool LogReaderWriter::ExistsAndCreate(string path_and_file)
{
	if (std::experimental::filesystem::exists(path_and_file)) {
		return true;
	}

	int idx = path_and_file.find_last_of("/");

	if (idx > 0) {
		string path = path_and_file.substr(0, idx);
		std::experimental::filesystem::create_directories(path);
	}

	cout << "[Info] - Created file " << path_and_file << "." << endl;

	return true;
}


/*
Write a string into a new file
*/
//static 
bool LogReaderWriter::FlashWrite(string path_and_file, string output)
{
	std::ofstream of(path_and_file, std::ofstream::out| std::ofstream::app);

	if (!of.is_open()) {
		cout << "[ERROR] - Cannot open " << _path_and_file << " to create a log file." << endl;
		return false;

	}
	of << output << "\n";
	of.close();

	return true;
}

/*
Write a string into a file only if this file does not exist.
*/
//static 
bool LogReaderWriter::FlashWriteHeader(string path_and_file, string header)
{
	ExistsAndCreate(path_and_file);

	if (std::experimental::filesystem::exists(path_and_file))
		return false;

	return FlashWrite( path_and_file,  header);
}