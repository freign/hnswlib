#pragma once

#include <string>
#include <iostream>
#include <filesystem>

using namespace std;
namespace fs = filesystem;

struct CommandLineOptions {
    std::string dataDir;          // Data directory
    std::string dataName;         // Data name
    std::string point_data_path;  // Path to point data
    std::string query_data_path;  // Path to query data
    int maxElements;              // Maximum number of elements

    CommandLineOptions();
    CommandLineOptions(const std::string& dir, const std::string& name, int maxElems);
};

// Declaration of the ArgParser function
CommandLineOptions ArgParser(int argc, char* argv[]);
